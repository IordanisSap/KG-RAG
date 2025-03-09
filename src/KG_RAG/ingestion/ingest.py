from .data_loader import DataLoader
from .text_splitter import Splitter
from .embedder import Embedder

import os

class Ingestor:
    def __init__(self, config):
        self.config = config
        self.data_loader = DataLoader()
        self.splitter = Splitter()
        self.embedder = Embedder(config["embedding-model"])
        
    
    def ingest_pdfs(self, dataset_dir: str, persist_dir: str):
        documents = self.data_loader.load_pdfs(dataset_dir)
        text_chunks = self.splitter.split_documents(documents)
        
        vectorstore_dir = os.path.join(persist_dir, self.config["vectorstore-dir"])
        bm25_dir = os.path.join(persist_dir, self.config["bm25-dir"])
        os.makedirs(vectorstore_dir, exist_ok=True)
        os.makedirs(bm25_dir, exist_ok=True)

        vectorstore = self.embedder.index_embeddings(text_chunks, vectorstore_dir)
        return vectorstore
    
    def get_vectorstore(self):
        return self.embedder.get_vectorstore()