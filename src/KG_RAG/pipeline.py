from .ingestion.ingest import Ingestor
from .retrieval.retriever import Retriever
from .generation.generator import Generator

import yaml
import os



with open(os.path.join(os.path.dirname(__file__), "config.yaml"), "r") as f:
    config = yaml.safe_load(f)

ingestionConfig = config["ingestion"]
retrivalConfig = config["retrieval"]
generationConfig = config["generation"]

ingestor = Ingestor(ingestionConfig["embedding-model"], ingestionConfig["persist-dir"])
retriever = Retriever(retrivalConfig)
generator = Generator(generationConfig)

# if ingestionConfig["dataset-dir"]:
#     vectorstore = ingestor.ingest_pdfs(ingestionConfig["dataset-dir"])
# else:
#     vectorstore = ingestor.get_vectorstore()
    
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