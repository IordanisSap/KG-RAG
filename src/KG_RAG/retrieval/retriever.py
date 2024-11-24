from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


class Retriever:
    def __init__(self, config):
        self.model = config["embedding-model"]
        self.persist_directory = config["persist-dir"]
        self.topk = config["topk"]
        self.score_threshold = config["score-threshold"]
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory, embedding_function=OllamaEmbeddings(model=self.model))

    def retrieve(self, prompt: str):
        retriever = self.vectorstore.as_retriever(
            search_type="similarity_score_threshold", search_kwargs={"k": self.topk , "score_threshold": self.score_threshold})
        retrieved_docs = retriever.invoke(prompt)
        return retrieved_docs

    def reload(self):
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory, embedding_function=OllamaEmbeddings(model=self.model))
