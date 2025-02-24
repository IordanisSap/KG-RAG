import json
from src.KG_RAG import RAGAgent
import os
import yaml




agent_config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

with open(agent_config_path, "r") as f:
    config_yaml = yaml.safe_load(f)
    
agent = RAGAgent(config_yaml)


dataset_path = os.path.join(os.path.dirname(__file__), "hotpot_dev_distractor_v1_small.json")

answers = {}

with open(dataset_path, 'r') as file:
    data = json.load(file)
    for q in data:
        id = q["_id"]
        question = q["question"]
        context = "\n\n".join(["\n".join(context[1]) for context in q["context"]])
        print(question)
        res = agent.generate("Data:" + context + "\nQuestion:" + question)
        answers[id] = res


print(answers)


with open("output.json", "w") as outfile:
    outfile.write(json.dumps({"answer":answers, "sp": {}},indent=4))