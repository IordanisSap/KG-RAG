from src.KG_RAG import RAGAgent
import os
import yaml

dataset_path = os.path.join(os.path.dirname(__file__), 'dataset-no-inv')

store_path = os.path.join(os.path.dirname(__file__), 'persist-no-inv')

config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

with open(config_path, "r") as f:
    config_yaml = yaml.safe_load(f)

agent = RAGAgent(config_yaml)
# agent.index_documents(dataset_path, store_path)


questions_path = os.path.join(os.path.dirname(__file__), "WC-P1.txt")
ans_path = os.path.join(os.path.dirname(__file__), "WC-P1-pred-no-inv.txt")

with open(questions_path, 'r') as qFile:
    with open(ans_path, 'w') as predFile:
        for line in qFile:
            question, ansData = line.split("?")
            print(question+" ?")
            pred, docs = agent.generate_rag_persist(question, store_path)
            predFile.write(pred.replace("\n", " ") + "\n")