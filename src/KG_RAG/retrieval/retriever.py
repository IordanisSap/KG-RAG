from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import os


class Retriever:
    def __init__(self, config):
        self.config = config


    def retrieve(self, prompt: str, topk=None, score_threshold=None, database_dir=None):
        if topk is None:
            topk = self.config["topk"]
        if score_threshold is None:
            score_threshold = self.config["score-threshold"]
        if database_dir is None:
            database_dir = os.listdir(self.config["persist-dir"])[0]
        database_path = os.path.join(self.config["persist-dir"], database_dir)
        vectorstore = Chroma(                                                                                           # DB is reloaded every time to be up to date and allow
            persist_directory=database_path, embedding_function=OllamaEmbeddings(model=self.config["embedding-model"]))  # for custom directory. TODO Improve it in the future
        
        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold", search_kwargs={"k": topk , "score_threshold": score_threshold})
        retrieved_docs = retriever.invoke(prompt)
        
        return retrieved_docs