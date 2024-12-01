import json
from src.KG_RAG import RAGAgent
import os
import yaml




agent_config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

with open(agent_config_path, "r") as f:
    config_yaml = yaml.safe_load(f)
    
agent = RAGAgent()


dataset_path = os.path.join(os.path.dirname(__file__), "small", "bigger.json")

with open(dataset_path, 'r') as file:
    data = json.load(file)              # TODO: Maybe use a library that does not load the whole JSON in RAM


answers = {}
for section in data["data"]:
    for paragraph in section["paragraphs"]:
        context = paragraph["context"]
        for qas in paragraph["qas"]:
            question = qas["question"]
            question_id = qas["id"]
            ans = agent.generate(f"{context}\n{question}")
            answers[question_id] = ans
            

out = os.path.join(os.path.dirname(__file__),"small", "out.json")
with open(out, "w") as file:
    json.dump(answers, file, indent=4)  # Pretty-print with 4 spaces




