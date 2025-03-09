from src.KG_RAG import RAGAgent


agent = RAGAgent()
agent.index_documents("examples/datasets/MarineRestorationAnalysis", "examples/example1/retrieval_store")

res, docs = agent.generate_rag("One sentence about palm trees?")
print(res, f" (Retrieved documents: {len(docs)})")

