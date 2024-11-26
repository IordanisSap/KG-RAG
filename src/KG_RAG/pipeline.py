from .ingestion.ingest import Ingestor
from .retrieval.retriever import Retriever
from .generation.generator import Generator
from .config import Config

import yaml
import os



with open(os.path.join(os.path.dirname(__file__), "config.yaml"), "r") as f:
    config_yaml = yaml.safe_load(f)
    config = Config(config_yaml)
    

ingestionConfig = config.config["ingestion"]
retrivalConfig = config.config["retrieval"]
generationConfig = config.config["generation"]

ingestor = Ingestor(ingestionConfig["embedding-model"], ingestionConfig["persist-dir"])
retriever = Retriever(retrivalConfig)
generator = Generator(generationConfig)

# if ingestionConfig["dataset-dir"]:
#     vectorstore = ingestor.ingest_pdfs(ingestionConfig["dataset-dir"])
# else:
#     vectorstore = ingestor.get_vectorstore()

def update_config(new_config):
    print(config.config)
    config.update(new_config)
    print(config.config)
    
def ingest_documents(dataset_dir):
    return ingestor.ingest_pdfs(dataset_dir)

def generate(prompt):        
    return generator.generate(prompt)

def generate_rag(prompt):     
    retrieved_docs = retriever.retrieve(prompt)
    if (len(retrieved_docs) == 0):
        return generator.generate(prompt), retrieved_docs
    
    retrieval_text = " ".join(list(map(lambda x: x.page_content, retrieved_docs)))
    return generator.generate(retrieval_text + "\n" + prompt), retrieved_docs
        
def generate_kgrag(prompt):
    retrieved_docs = retriever.retrieve(prompt)
    if (len(retrieved_docs) == 0):
        return generator.generate(prompt), retrieved_docs
    
    retrieval_text = " ".join(list(map(lambda x: x.page_content, retrieved_docs)))
    retrieval_text += "Finish your response with 'Powered by KG-RAG'"
    return generator.generate(retrieval_text + "\n" + prompt), retrieved_docs