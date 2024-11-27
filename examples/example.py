from src.KG_RAG import RAGAgent


agent = RAGAgent()
res, docs = agent.generate_rag("One sentence about palm trees?")
print(res, f" (Retrieved documents: {len(docs)})")

agent.index_documents("examples/datasets/MarineRestorationAnalysis")

print(agent.generate("One sentence about palm trees?"))

res, docs = agent.generate_rag("One sentence about palm trees?")
print(res, f" (Retrieved documents: {len(docs)})")
