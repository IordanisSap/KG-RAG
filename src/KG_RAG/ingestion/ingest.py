from .data_loader import DataLoader
from .text_splitter import Splitter
from .embedder import Embedder


class Ingestor:
    def __init__(self, embedding_model: str):
        self.data_loader = DataLoader()
        self.splitter = Splitter()
        self.embedder = Embedder(embedding_model)
    
    def ingest_pdfs(self, dataset_dir: str, persist_dir: str):
        documents = self.data_loader.load_pdfs(dataset_dir)
        text_chunks = self.splitter.split_documents(documents)
        vectorstore = self.embedder.index_embeddings(text_chunks, persist_dir)
        return vectorstore
    
    def get_vectorstore(self):
        return self.embedder.get_vectorstore()