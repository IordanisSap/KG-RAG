from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from .fulltext import BM25Retriever

class Retriever:
    def __init__(self, config):
        self.config = config
        self.bm25_retriever = BM25Retriever(config)


    def retrieve(self, prompt: str, persist_dir=None, topk=None, score_threshold=None):
        if topk is None:
            topk = self.config["topk"]
        if score_threshold is None:
            score_threshold = self.config["score-threshold"]
        if persist_dir is None:
            persist_dir = self.config["persist-dir"]
            
        vectorstore_dir = os.path.join(persist_dir, self.config["vectorstore-dir"])
        bm25_dir = os.path.join(persist_dir, self.config["bm25-dir"])
        
        vectorstore = Chroma(                                                                                            # DB is reloaded every time to be up to date and allow
            persist_directory=vectorstore_dir, embedding_function=OllamaEmbeddings(model=self.config["embedding-model"]))  # for custom directory. TODO Improve it in the future
        
        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold", search_kwargs={"k": topk , "score_threshold": score_threshold})
        retrieved_docs = retriever.invoke(prompt)
        
        bm25_docs = self.bm25_retriever.retrieve(prompt, bm25_dir)
        
        left_docs = topk - len(retrieved_docs)
        
        return retrieved_docs + bm25_docs[0:left_docs]
    
    def get_similarity(self, text1, text2):
        embeddings_model = OllamaEmbeddings(model=self.config["embedding-model"])
        embedding1 = embeddings_model.embed_query(text1)
        embedding2 = embeddings_model.embed_query(text2)
        similarity = cosine_similarity(
            [embedding1],
            [embedding2]
        )[0][0]

        return similarity
        