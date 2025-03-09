import json
from src.KG_RAG.knowledge_graph.KG import generate_KG 
import os
import yaml




# agent_config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

# with open(agent_config_path, "r") as f:
#     config_yaml = yaml.safe_load(f)
    
# agent = RAGAgent(config_yaml)


# dataset_path = os.path.join(os.path.dirname(__file__), "hotpot_dev_distractor_v1_small.json")

# answers = {}

# with open(dataset_path, 'r') as file:
#     data = json.load(file)
#     for q in data:
#         id = q["_id"]
#         question = q["question"]
#         context = "\n\n".join(["\n".join(context[1]) for context in q["context"]])
#         print(question)
#         res = agent.generate("Data:" + context + "\nQuestion:" + question)
#         answers[id] = res


# print(answers)


# with open("output.json", "w") as outfile:
#     outfile.write(json.dumps({"answer":answers, "sp": {}},indent=4))
    
    
    
import json

        

test_case = {
    "_id": "5a8b57f25542995d1e6f1371",
    "answer": "yes",
    "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
    "supporting_facts": [
        [
            "Scott Derrickson",
            0
        ],
        [
            "Ed Wood",
            0
        ]
    ],
    "context": [
        [
            "Scott Derrickson",
            [
                "Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer.",
                " He lives in Los Angeles, California.",
                " He is best known for directing horror films such as \"Sinister\", \"The Exorcism of Emily Rose\", and \"Deliver Us From Evil\", as well as the 2016 Marvel Cinematic Universe installment, \"Doctor Strange.\""
            ]
        ],
        [
            "Ed Wood",
            [
                "Edward Davis Wood Jr. (October 10, 1924 \u2013 December 10, 1978) was an American filmmaker, actor, writer, producer, and director."
            ]
        ],
    ],
    "type": "comparison",
    "level": "hard"
}

dict = {}    

all_sentences = []
for context in test_case["context"]:
    sentences = "".join(context[1])
    all_sentences.append(sentences)


print(generate_KG(all_sentences))
        