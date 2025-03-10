from src.KG_RAG import RAGAgent


agent = RAGAgent()
agent.index_documents("/mnt/10TB/iordanissapidis/datasets/large", "/mnt/10TB/iordanissapidis/SemanticRAG/retrieval_store/FAO")

