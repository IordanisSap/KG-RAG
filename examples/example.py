from src.KG_RAG import ingest_documents, generate, generate_rag, generate_kgrag

res, docs = generate_rag("One sentence about palm trees?")
print(res, docs)

ingest_documents("examples/datasets/MarineRestorationAnalysis")

print(generate("One sentence about palm trees?"))

res, docs = generate_rag("One sentence about palm trees?")
print(res, docs)
