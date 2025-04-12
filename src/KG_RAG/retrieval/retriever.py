from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import math
import hashlib

from .fulltext import BM25Retriever
from KG_RAG.utils import benchmark

class Retriever:
    def __init__(self, config):
        self.config = config
        self.bm25Retriever = BM25Retriever(self.config)
        
        
    def load(self, persist_dir):
        if persist_dir is None:
            persist_dir = self.config["persist-dir"]
           
        vectorstore_dir = os.path.join(persist_dir, self.config["vectorstore-dir"])
        vectorstore = Chroma(                                                                                            # DB is reloaded every time to be up to date and allow
            persist_directory=vectorstore_dir, embedding_function=OllamaEmbeddings(model=self.config["embedding-model"]))  # for custom directory. TODO Improve it in the future
    
        bm25_dir = os.path.join(persist_dir, self.config["bm25-dir"])
        bm25Index = self.bm25Retriever.load(bm25_dir)
        
        return vectorstore, bm25Index

    def retrieve_persist(self, prompt: str, persist_dir=None, retrieval_config={}):
        if persist_dir is None:
            persist_dir = self.config["persist-dir"]
        
        if self.config['separate-filetypes']:
            extensions = retrieval_config.get("extensions", ['pdf','csv'])
            topk = retrieval_config.get('topk', self.config["topk"])
            documents = []
            for i, ext in enumerate(extensions):
                vectorstore_dir = os.path.join(persist_dir, ext, self.config["vectorstore-dir"])
                if not os.path.isdir(vectorstore_dir): continue
                vectorstore = Chroma(                                                                                            # DB is reloaded every time to be up to date and allow
                    persist_directory=vectorstore_dir, embedding_function=OllamaEmbeddings(model=self.config["embedding-model"]))  # for custom directory. TODO Improve it in the future
            
                bm25_dir = os.path.join(persist_dir, ext, self.config["bm25-dir"])
                bm25Index = self.bm25Retriever.load(bm25_dir)
                documents.extend(self.retrieve(prompt, vectorstore, bm25Index, retrieval_config))
            return rerank_documents(documents, prompt, OllamaEmbeddings(model=self.config["embedding-model"]))[:topk]
        else:
            vectorstore_dir = os.path.join(persist_dir, self.config["vectorstore-dir"])
            vectorstore = Chroma(                                                                                            # DB is reloaded every time to be up to date and allow
                persist_directory=vectorstore_dir, embedding_function=OllamaEmbeddings(model=self.config["embedding-model"]))  # for custom directory. TODO Improve it in the future
        
            bm25_dir = os.path.join(persist_dir, self.config["bm25-dir"])
            bm25Index = self.bm25Retriever.load(bm25_dir)
            
            return self.retrieve(prompt, vectorstore, bm25Index, retrieval_config, True)
    
    
    def retrieve(self, prompt: str, vectorstore, bm25Index, retrieval_config={}, rerank=False):
        candidate_pool_size = retrieval_config.get('candidate-pool-size', self.config["candidate-pool-size"])
        topk = retrieval_config.get('topk', self.config["topk"])
        score_threshold = retrieval_config.get('score_threshold',self.config["score-threshold"])
        
        vector_retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold", search_kwargs={"k": candidate_pool_size , "score_threshold": score_threshold})
                   
        bm25_docs = self.bm25Retriever.retrieve(prompt, bm25Index)[:candidate_pool_size]
        vector_docs = vector_retriever.invoke(prompt)
        vector_docs.extend(bm25_docs)
        return rerank_documents(vector_docs, prompt, OllamaEmbeddings(model=self.config["embedding-model"]))[:topk] if rerank else vector_docs
        
    
    def get_similarity(self, text1, text2):
        embeddings_model = OllamaEmbeddings(model=self.config["embedding-model"])
        embedding1 = embeddings_model.embed_query(text1)
        embedding2 = embeddings_model.embed_query(text2)
        similarity = cosine_similarity(
            [embedding1],
            [embedding2]
        )[0][0]

        return similarity
        


# Reranking TODO: Improve this using a cross encoder
def cosine_similarity(vec_a, vec_b):
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

@benchmark
def rerank_documents(docs, query, embedding_model):
    """
    Re-rank a list of LangChain Documents by similarity to a given query.
    """
    # 1. Embed the query
    query_embedding = embedding_model.embed_query(query)
    
    # 2. For each document, embed and compute similarity
    doc_sim_pairs = []
    for doc in docs:
        # embed_documents() returns a list, so we take the first element
        doc_embedding = embedding_model.embed_documents([doc.page_content])[0]
        sim = cosine_similarity(query_embedding, doc_embedding)
        doc_sim_pairs.append((doc, sim))
    

    def doc_hash(doc):
        content = doc.page_content.strip()
        meta = str(doc.metadata.get('source', ''))  # customize based on your metadata
        return hashlib.md5((content + meta).encode('utf-8')).hexdigest()

    unique_docs = []
    seen_hashes = set()

    for (doc,sim) in doc_sim_pairs:
        h = doc_hash(doc)
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_docs.append((doc,sim))

    # 3. Sort descending by similarity
    unique_docs.sort(key=lambda x: x[1], reverse=True)
    
    # 4. Return the documents in the new order
    return [pair[0] for pair in unique_docs]