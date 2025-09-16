from src.KG_RAG import RAGAgent
import os
import yaml

text = 'Examine all the provided data carefully and answer only according to that data.   Respond with a single word, name, or date—*only the exact answer required by the question*.   Identify the entity requested in the question and ignore all data not directly mentioning that entity.   Do not include additional details such as dates, descriptions, or extra context.   If multiple answers are needed, separate them using the character | (e.g., X | Y | Z)..\nDocument 1: Verbalization: Zardoz starred actors Sara Kestelman\n[Sara Kestelman] appears in which movies'

config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

with open(config_path, "r") as f:
    config_yaml = yaml.safe_load(f)

agent = RAGAgent(config_yaml)

res = agent.generate(text)
print(res)

