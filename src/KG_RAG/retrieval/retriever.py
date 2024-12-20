from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


class Retriever:
    def __init__(self, config):
        self.config = config
        self.vectorstore = Chroma(
            persist_directory=self.config["persist-dir"], embedding_function=OllamaEmbeddings(model=config["embedding-model"]))

    def retrieve(self, prompt: str, topk=None, score_threshold=None):
        if topk is None:
            topk = self.config["topk"]
        if score_threshold is None:
            score_threshold = self.config["score-threshold"]
        retriever = self.vectorstore.as_retriever(
            search_type="similarity_score_threshold", search_kwargs={"k": topk , "score_threshold": score_threshold})
        retrieved_docs = retriever.invoke(prompt)
        return retrieved_docs

    def reload(self):
        self.vectorstore = Chroma(
            persist_directory=self.config["persist-dir"], embedding_function=OllamaEmbeddings(model=self.model))
