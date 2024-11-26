from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


class Retriever:
    def __init__(self, config):
        self.config = config
        self.vectorstore = Chroma(
            persist_directory=self.config["persist-dir"], embedding_function=OllamaEmbeddings(model=config["embedding-model"]))

    def retrieve(self, prompt: str):
        retriever = self.vectorstore.as_retriever(
            search_type="similarity_score_threshold", search_kwargs={"k": self.config["topk"] , "score_threshold": self.config["score-threshold"]})
        retrieved_docs = retriever.invoke(prompt)
        return retrieved_docs

    def reload(self):
        self.vectorstore = Chroma(
            persist_directory=self.config["persist-dir"], embedding_function=OllamaEmbeddings(model=self.model))
