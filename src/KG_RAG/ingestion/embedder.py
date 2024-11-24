from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from ..utils import call_in_batches, log

class Embedder:
    def __init__(self, model: str, persist_directory: str):
        self.vectorstore = Chroma(
            collection_metadata={"hnsw:space": "cosine"},
            persist_directory=persist_directory,
            embedding_function=OllamaEmbeddings(model=model)
        )

    def index_embeddings(self, text_chunks):
        log("Indexing embeddings")
        index_func = lambda documents: self.vectorstore.add_documents(documents=documents)
        call_in_batches(index_func,text_chunks, 500)
        return self.vectorstore
    
    def get_vectorstore(self):
        return self.vectorstore