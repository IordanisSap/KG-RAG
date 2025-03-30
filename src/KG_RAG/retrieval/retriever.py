from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from .fulltext import BM25Retriever

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
            num_different_ext = len(extensions)
            topk = retrieval_config.get('topk', self.config["topk"])
            topk_each_ext = distribute_k_across_num(topk, num_different_ext)
            print(topk_each_ext)
            documents = []
            for i, ext in enumerate(extensions):
                vectorstore_dir = os.path.join(persist_dir, ext, self.config["vectorstore-dir"])
                vectorstore = Chroma(                                                                                            # DB is reloaded every time to be up to date and allow
                    persist_directory=vectorstore_dir, embedding_function=OllamaEmbeddings(model=self.config["embedding-model"]))  # for custom directory. TODO Improve it in the future
            
                bm25_dir = os.path.join(persist_dir, ext, self.config["bm25-dir"])
                bm25Index = self.bm25Retriever.load(bm25_dir)
                retrieval_config["topk"] = topk_each_ext[i]
                documents.extend(self.retrieve(prompt, vectorstore, bm25Index, retrieval_config))
            return documents
        else:
            vectorstore_dir = os.path.join(persist_dir, self.config["vectorstore-dir"])
            vectorstore = Chroma(                                                                                            # DB is reloaded every time to be up to date and allow
                persist_directory=vectorstore_dir, embedding_function=OllamaEmbeddings(model=self.config["embedding-model"]))  # for custom directory. TODO Improve it in the future
        
            bm25_dir = os.path.join(persist_dir, self.config["bm25-dir"])
            bm25Index = self.bm25Retriever.load(bm25_dir)
            
            return self.retrieve(prompt, vectorstore, bm25Index, retrieval_config)
    
    
    def retrieve(self, prompt: str, vectorstore, bm25Index, retrieval_config={}):
        topk = retrieval_config.get('topk', self.config["topk"])
        print(retrieval_config)
        score_threshold = retrieval_config.get('score_threshold',self.config["score-threshold"])
        
        vector_retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold", search_kwargs={"k": topk , "score_threshold": score_threshold})
                   
        bm25_docs = self.bm25Retriever.retrieve(prompt, bm25Index)
        vector_docs = vector_retriever.invoke(prompt)
        
        
        left_docs = topk - len(vector_docs)
        
        return vector_docs + bm25_docs[0:left_docs]
    
    def get_similarity(self, text1, text2):
        embeddings_model = OllamaEmbeddings(model=self.config["embedding-model"])
        embedding1 = embeddings_model.embed_query(text1)
        embedding2 = embeddings_model.embed_query(text2)
        similarity = cosine_similarity(
            [embedding1],
            [embedding2]
        )[0][0]

        return similarity
        
def distribute_k_across_num(k, num):
    base = k // num
    remainder = k % num
    return [base + 1 if i < remainder else base for i in range(num)]