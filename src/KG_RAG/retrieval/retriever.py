from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class Retriever:
    def __init__(self, config):
        self.config = config


    def retrieve(self, prompt: str, persist_dir=None, topk=None, score_threshold=None):
        if topk is None:
            topk = self.config["topk"]
        if score_threshold is None:
            score_threshold = self.config["score-threshold"]
        if persist_dir is None:
            persist_dir = os.listdir(self.config["persist-dir"])[0]
        database_path = os.path.join(self.config["persist-dir"], persist_dir)
        vectorstore = Chroma(                                                                                            # DB is reloaded every time to be up to date and allow
            persist_directory=database_path, embedding_function=OllamaEmbeddings(model=self.config["embedding-model"]))  # for custom directory. TODO Improve it in the future
        
        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold", search_kwargs={"k": topk , "score_threshold": score_threshold})
        retrieved_docs = retriever.invoke(prompt)
        
        return retrieved_docs
    
    def get_similarity(self, text1, text2):
        embeddings_model = OllamaEmbeddings(model=self.config["embedding-model"])
        embedding1 = embeddings_model.embed_query(text1)
        embedding2 = embeddings_model.embed_query(text2)
        similarity = cosine_similarity(
            [embedding1],
            [embedding2]
        )[0][0]

        return similarity
        