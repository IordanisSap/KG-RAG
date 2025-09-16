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


benchmark_path = os.path.join(os.path.dirname(__file__), '1-hop', 'qa_test.txt')
pred_path_verb = os.path.join(os.path.dirname(__file__), '1-hop_pred', 'pred-verbalized2-llama-topk10.txt')

pred_file = open(pred_path_verb, 'w')
with open(benchmark_path, 'r') as benchFile:
    for line in benchFile:
        question, answers_str = line.split("\t")
        answers = answers_str.rstrip().split("|")
        print(answers)

        llm_pred_str, docs = agent.generate_rag_persist(question)
        llm_pred = llm_pred_str.rstrip().replace("\n", " ").split("|")
        llm_pred = [answer.strip() for answer in llm_pred]
        llm_pred = list(dict.fromkeys(llm_pred))  # Remove duplicates
        pred_file.write("|".join(llm_pred) + "\n")
        pred_file.flush()

pred_file.close()




# pred_path_raw = os.path.join(os.path.dirname(__file__), '1-hop_pred', 'pred.txt')

# pred_file = open(pred_path_raw, 'w')
# with open(benchmark_path, 'r') as benchFile:
#     for line in benchFile:
#         question, answers_str = line.split("\t")
#         answers = answers_str.rstrip().split("|")
#         print(answers)

#         llm_pred_str, docs = agent.generate_rag_persist(question + "?")
#         llm_pred = llm_pred_str.rstrip().replace("\n", " ").split("|")
#         llm_pred = [answer.strip() for answer in llm_pred]
#         llm_pred = list(dict.fromkeys(llm_pred))  # Remove duplicates
#         print("LLM_prediction", llm_pred)
#         pred_file.write("|".join(llm_pred) + "\n")
#         pred_file.flush()

# pred_file.close()
