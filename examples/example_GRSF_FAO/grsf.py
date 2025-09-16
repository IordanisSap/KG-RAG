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

# res, docs = agent.generate_rag_persist("Is conservation translocation of tortoises generally recommended?")
# res, docs = agent.generate_rag_persist("Do giant tortoises live in places with predators?")
# res, docs = agent.generate_rag_persist("Are there currently tigers in Bali?")
res, docs = agent.generate_kgrag_persist("Is the freshwater sculpin protected by the EU Habitats Directive?")




print("----------------------------")
for doc in docs[0:6]:
    print(doc.page_content)
    print("----------------------------")
print(res, f" (Retrieved documents: {len(docs)})")