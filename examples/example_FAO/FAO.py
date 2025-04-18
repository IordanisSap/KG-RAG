from src.KG_RAG import RAGAgent
import os
import yaml

dataset_path = os.path.join(os.path.dirname(__file__), 'dataset')
store_path = os.path.join(os.path.dirname(__file__), 'persist')

config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

with open(config_path, "r") as f:
    config_yaml = yaml.safe_load(f)

agent = RAGAgent(config_yaml)

# agent.index_documents(dataset_path, store_path)

res, docs = agent.generate_rag_persist("One sentence about palm trees?")
print(res, f" (Retrieved documents: {len(docs)})")