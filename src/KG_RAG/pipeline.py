from .ingestion.ingest import Ingestor
from .retrieval.retriever import Retriever
from .generation.generator import Generator
from .config import Config

import yaml
import os




    
default_config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

with open(default_config_path, "r") as f:
    config_yaml = yaml.safe_load(f)

class RAGAgent:
    def __init__(self, config=config_yaml):
        self.config = Config(config_yaml)
        self.config.update(config)
        self._initialize_components()
        
    def _initialize_components(self):
        ingestion_config = self.config.config["ingestion"]
        retrieval_config = self.config.config["retrieval"]
        generation_config = self.config.config["generation"]

        self.ingestor = Ingestor(ingestion_config["embedding-model"], ingestion_config["persist-dir"])
        self.retriever = Retriever(retrieval_config)
        self.generator = Generator(generation_config)
        
    def index_documents(self, dataset_dir):
        return self.ingestor.ingest_pdfs(dataset_dir)

    def generate(self, prompt):
        return self.generator.generate(prompt)

    def generate_rag(self, prompt):
        retrieved_docs = self.retriever.retrieve(prompt)
        if not retrieved_docs:
            return self.generator.generate(prompt), retrieved_docs

        retrieval_text = " ".join(doc.page_content for doc in retrieved_docs)
        return self.generator.generate(retrieval_text + "\n" + prompt), retrieved_docs
    
    def generate_kgrag(self, prompt):
        retrieved_docs = self.retriever.retrieve(prompt)
        if (len(retrieved_docs) == 0):
            return self.generator.generate(prompt), retrieved_docs
        
        retrieval_text = " ".join(list(map(lambda x: x.page_content, retrieved_docs)))
        retrieval_text += "Finish your response with 'Powered by KG-RAG'"
        return self.generator.generate(retrieval_text + "\n" + prompt), retrieved_docs
        
        
# def update_config(new_config):
#     rag_manager.update_config(new_config)

# def index_documents(dataset_dir):
#     return rag_manager.index_documents(dataset_dir)

# def generate(prompt):
#     return rag_manager.generate(prompt)

# def generate_rag(prompt):
#     return rag_manager.generate_rag(prompt)

# def generate_kgrag(prompt):
#     return rag_manager.generate_kgrag(prompt)