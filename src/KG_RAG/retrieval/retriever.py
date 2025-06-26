import os

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker

from sklearn.metrics.pairwise import cosine_similarity




from .fulltext import BM25Retriever
from KG_RAG.utils import benchmark


class Retriever:
    def __init__(self, config):
        self.config = config
        self.bm25Retriever = BM25Retriever(self.config)

    def load(self, persist_dir):
        if persist_dir is None:
            persist_dir = self.config["persist-dir"]

        vectorstore_dir = os.path.join(
            persist_dir, self.config["vectorstore-dir"])
        vectorstore = Chroma(                                                                                            # DB is reloaded every time to be up to date and allow
            # for custom directory. TODO Improve it in the future
            persist_directory=vectorstore_dir, embedding_function=OllamaEmbeddings(model=self.config["embedding-model"]))

        bm25_dir = os.path.join(persist_dir, self.config["bm25-dir"])
        bm25Index = self.bm25Retriever.load(bm25_dir)

        return vectorstore, bm25Index

    def retrieve_persist(self, prompt: str, persist_dir=None, retrieval_config={}):
        if persist_dir is None:
            persist_dir = self.config["persist-dir"]

        if self.config['separate-filetypes']:
            extensions = retrieval_config.get("extensions", ['pdf', 'csv'])
            topk = retrieval_config.get('topk', self.config["topk"])
            documents = {}
            for i, ext in enumerate(extensions):
                vectorstore_dir = os.path.join(
                    persist_dir, ext, self.config["vectorstore-dir"])
                if not os.path.isdir(vectorstore_dir):
                    continue
                vectorstore = Chroma(                                                                                            # DB is reloaded every time to be up to date and allow
                    # for custom directory. TODO Improve it in the future
                    persist_directory=vectorstore_dir, embedding_function=OllamaEmbeddings(model=self.config["embedding-model"]))

                bm25_dir = os.path.join(
                    persist_dir, ext, self.config["bm25-dir"])
                bm25Index = self.bm25Retriever.load(bm25_dir)
                documents[ext] = self.retrieve(
                    prompt, vectorstore, bm25Index, retrieval_config)
            return merge_docs(prompt, documents)[:topk]
        else:
            vectorstore_dir = os.path.join(
                persist_dir, self.config["vectorstore-dir"])
            vectorstore = Chroma(                                                                                            # DB is reloaded every time to be up to date and allow
                # for custom directory. TODO Improve it in the future
                persist_directory=vectorstore_dir, embedding_function=OllamaEmbeddings(model=self.config["embedding-model"]))

            bm25_dir = os.path.join(persist_dir, self.config["bm25-dir"])
            bm25Index = self.bm25Retriever.load(bm25_dir)

            return self.retrieve(prompt, vectorstore, bm25Index, retrieval_config, True)

    def retrieve(self, prompt: str, vectorstore, bm25Index, retrieval_config={}):
        candidate_pool_size = retrieval_config.get(
            'candidate-pool-size', self.config["candidate-pool-size"])
        topk = retrieval_config.get('topk', self.config["topk"])
        score_threshold = retrieval_config.get(
            'score_threshold', self.config["score-threshold"])

        vector_retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold", search_kwargs={"k": candidate_pool_size, "score_threshold": score_threshold})

        documents = {}
        documents["BM25"] = self.bm25Retriever.retrieve(prompt, bm25Index)[
            :candidate_pool_size]
        documents["DPR"] = vector_retriever.invoke(prompt)
        # get_relevant_documents
        return merge_docs(prompt, documents)[:topk]

    def get_similarity(self, text1, text2):
        embeddings_model = OllamaEmbeddings(
            model=self.config["embedding-model"])
        embedding1 = embeddings_model.embed_query(text1)
        embedding2 = embeddings_model.embed_query(text2)
        similarity = cosine_similarity(
            [embedding1],
            [embedding2]
        )[0][0]

        return similarity

# Merge and rerank


@benchmark
def merge_docs(query, docsExtDict):
    merged_docs = []
    for key, docs in docsExtDict.items():
        merged_docs.extend(docs)

    # Filter duplicate docs
    unique_merged_docs = []
    seen_combinations = set()
    for doc in merged_docs:
        source = doc.metadata.get("source")
        page = doc.metadata.get("page")
        uniqueness_key = (doc.page_content, source, page)

        if uniqueness_key not in seen_combinations:
            unique_merged_docs.append(doc)
            seen_combinations.add(uniqueness_key)


    hf_ce = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

    reranker = CrossEncoderReranker(model=hf_ce)

    final_docs = reranker.compress_documents(
        documents=unique_merged_docs,
        query=query,
    )

    return final_docs
