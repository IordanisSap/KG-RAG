from KG_RAG.pipeline import RAGAgent
from KG_RAG.knowledge_graph.graph import create_graph
# from KG_RAG.knowledge_graph.nlp import lemmatize_word
import json

'''
passages = [
    {
        text: 'text',
        metadata: 'metadata'
    }
]
'''

agent = RAGAgent()


def generate_KG(passages):
    triples = []
    for passage in passages:
        generated_triples = agent.generate_triples(passage)
        for triple in generated_triples:
            triples.append(triple)
    
    create_graph(triples)
    return triples
