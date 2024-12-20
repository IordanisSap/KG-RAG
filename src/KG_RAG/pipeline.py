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
    
    def retrieve(self, prompt, topk, score_threshold):
        return self.retriever.retrieve(prompt, topk, score_threshold)

    def generate(self, prompt):
        return self.generator.generate(prompt)

    def generate_rag(self, prompt, retrieval_config=None):
        if retrieval_config:
            params = {"topk", "score_threshold"}
            filtered_config = {param: val for param, val in retrieval_config.items() if param in params}
            retrieved_docs = self.retriever.retrieve(prompt, **filtered_config) 
        else: retrieved_docs = self.retriever.retrieve(prompt)
        if not retrieved_docs:
            return self.generator.generate(prompt), retrieved_docs

        rag_prompt = config_yaml["generation"]["prompts"].get("rag_prompt", None)
        retrieval_text = rag_prompt + ".\n" + " ".join(doc.page_content for doc in retrieved_docs)
        return self.generator.generate(retrieval_text + "\n" + prompt), retrieved_docs
    
    def generate_kgrag(self, prompt, retrieval_config=None):
        if retrieval_config:
            params = {"topk", "score_threshold"}
            filtered_config = {param: val for param, val in retrieval_config.items() if param in params}
            retrieved_docs = self.retriever.retrieve(prompt, **filtered_config) 
        else: retrieved_docs = self.retriever.retrieve(prompt)
        if not retrieved_docs:
            return self.generator.generate(prompt), retrieved_docs
        
        rag_prompt = config_yaml["prompts"].get("rag_prompt", None)
        retrieval_text = rag_prompt + ".\n" + " ".join(list(map(lambda x: x.page_content, retrieved_docs)))
        retrieval_text += "Finish your response with 'Powered by KG-RAG'"
        return self.generator.generate(retrieval_text + "\n" + prompt), retrieved_docs
        
        