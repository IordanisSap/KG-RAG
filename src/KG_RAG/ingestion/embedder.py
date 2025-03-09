from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from ..utils import call_in_batches
import logging

class Embedder:
    def __init__(self, model: str):
        self.model = model

    def index_embeddings(self, text_chunks, persist_dir):
        logging.info("Indexing embeddings")
        
        vectorstore = Chroma(
            collection_metadata={"hnsw:space": "cosine"},
            persist_directory=persist_dir,
            embedding_function=OllamaEmbeddings(model=self.model)
        )
        
        index_func = lambda documents: vectorstore.add_documents(documents=documents)
        call_in_batches(index_func,text_chunks, 500)
        return vectorstore
    
    def get_vectorstore(self):
        return self.vectorstore