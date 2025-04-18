from src.KG_RAG import RAGAgent
import os
import yaml
import json


config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

data_file = os.path.join(os.path.dirname(__file__), "gold.jsonl")
predictions_file =  os.path.join(os.path.dirname(__file__), "predictions.jsonl")

with open(config_path, "r") as f:
    config_yaml = yaml.safe_load(f)

agent = RAGAgent(config_yaml)

# dict_keys(['id', 'paragraphs', 'question', 'question_decomposition', 'answer', 'answer_aliases', 'answerable'])
# obj["paragraphs"][0] dict_keys(['idx', 'title', 'paragraph_text', 'is_supporting'])
persist_dir = "/mnt/10TB/iordanissapidis/SemanticRAG/retrieval_store/FAO-GRSF"
# count=0
with open(data_file, 'r', encoding='utf-8') as infile, \
        open(predictions_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        obj = json.loads(line.strip())
        prompt = obj["question"]
        docs = agent.retrieve_persist(prompt, persist_dir)
        generationFunc = lambda: agent.generate_rag(obj["question"], docs)
        agent.generate_to_file_with_facts(obj["id"], generationFunc,predictions_file)
        # count+=1
        # if count > 10: break
        


