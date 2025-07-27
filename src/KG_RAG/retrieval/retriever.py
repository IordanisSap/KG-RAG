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

        topk = retrieval_config.get('topk', self.config["topk"])
        score_threshold = retrieval_config.get(
            'score_threshold', self.config["score-threshold"])

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
            merged_docs = merge_docs(documents)
            return rerank_docs(prompt, merged_docs, topk, score_threshold)
        else:
            vectorstore_dir = os.path.join(
                persist_dir, self.config["vectorstore-dir"])
            vectorstore = Chroma(                                                                                            # DB is reloaded every time to be up to date and allow
                # for custom directory. TODO Improve it in the future
                persist_directory=vectorstore_dir, embedding_function=OllamaEmbeddings(model=self.config["embedding-model"]))

            bm25_dir = os.path.join(persist_dir, self.config["bm25-dir"])
            bm25Index = self.bm25Retriever.load(bm25_dir)

            return rerank_docs(prompt, self.retrieve(prompt, vectorstore, bm25Index, retrieval_config, True), topk, score_threshold)

    def retrieve(self, prompt: str, vectorstore, bm25Index, retrieval_config={}):
        candidate_pool_size = retrieval_config.get(
            'candidate-pool-size', self.config["candidate-pool-size"])
        
        score_threshold = retrieval_config.get(
            'score_threshold', self.config["score-threshold"])

        vector_retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold", search_kwargs={"k": candidate_pool_size, "score_threshold": score_threshold})

        documents = {}
        documents["BM25"] = self.bm25Retriever.retrieve(prompt, bm25Index)[
            :candidate_pool_size]
        documents["DPR"] = vector_retriever.invoke(prompt)
        # get_relevant_documents
        return merge_docs(documents)

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
def merge_docs(docsDict):
    merged_docs = []
    for key, docs in docsDict.items():
        merged_docs.extend(docs)

    if len(merged_docs) == 0:
        return []
    # Filter duplicate docs
    unique_merged_docs = []
    seen_combinations = set()
    for doc in merged_docs:
        source = doc.metadata.get("source")
        # page = doc.metadata.get("page")
        uniqueness_key = (doc.page_content, source)

        if uniqueness_key not in seen_combinations:
            unique_merged_docs.append(doc)
            seen_combinations.add(uniqueness_key)

    return unique_merged_docs

@benchmark
def rerank_docs(query, docs, topk, score_threshold=0):
    hf_ce = HuggingFaceCrossEncoder(model_name="Alibaba-NLP/gte-reranker-modernbert-base")

    reranker = CrossEncoderReranker(model=hf_ce, top_n=topk)

    final_docs = reranker.compress_documents(
        documents=docs,
        query=query,
    )
    max_score = hf_ce.score([(query, docs[0].page_content)])[0]
    min_score = hf_ce.score([(query, docs[-1].page_content)])[0]

    def normalize_score(score):
        return (score-min_score)/(max_score-min_score)

    final_docs_threshold = [doc for doc in final_docs if normalize_score(hf_ce.score([(query, doc.page_content)])[0]) >= score_threshold] 
    final_docs_scored = [(doc, hf_ce.score([(query, doc.page_content)])) for doc in final_docs]

    

    return final_docs_threshold