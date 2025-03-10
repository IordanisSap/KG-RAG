import bm25s
import Stemmer
from typing import List
from langchain_core.documents import Document


class BM25Retriever:
    def __init__(self, config):
        self.config = config
        self.stemmer = Stemmer.Stemmer("english")
        self.retriever = bm25s.BM25()

    def retrieve(self, query, persist_dir):
        retriever = bm25s.BM25.load(persist_dir, load_corpus=True)
        query_tokens = bm25s.tokenize(query, stemmer=self.stemmer)
        results, scores = retriever.retrieve(query_tokens, k=5)
        for i in range(results.shape[1]):
            doc, score = results[0, i], scores[0, i]
            print(f"{i+1}: (score: {score:.2f}): {doc["id"]}")

        return [
            Document(result["text"], metadata={
                    "source": "BM25 - NO URL", "page": "1"})
            for result, score in zip(results[0, :], scores[0, :]) if score > 1
        ]
