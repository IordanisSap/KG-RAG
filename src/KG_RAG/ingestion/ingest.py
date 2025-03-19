from .data_loader import DataLoader
from .text_splitter import Splitter
from .embedder import Embedder
from .fulltext import BM25Indexer
import logging

import os

class Ingestor:
    def __init__(self, config):
        self.config = config
        self.data_loader = DataLoader()
        self.splitter = Splitter()
        self.embedder = Embedder(config["embedding-model"])
        self.fulltextIndexer = BM25Indexer(config)
    
        
    def ingest(self, dataset_dir: str, persist_dir: str):
        documents = self.data_loader.load(dataset_dir)
        text_chunks = self.splitter.split_documents(documents)
        vectorstore_dir = os.path.join(persist_dir, self.config["vectorstore-dir"])
        bm25_dir = os.path.join(persist_dir, self.config["bm25-dir"])
        os.makedirs(vectorstore_dir, exist_ok=True)
        os.makedirs(bm25_dir, exist_ok=True)
        
        logging.info("Full-text Indexing")
        self.fulltextIndexer.index(text_chunks, bm25_dir)

        logging.info("Vector Indexing")
        vectorstore = self.embedder.index_embeddings(text_chunks, vectorstore_dir)
        return vectorstore

    def get_vectorstore(self):
        return self.embedder.get_vectorstore()