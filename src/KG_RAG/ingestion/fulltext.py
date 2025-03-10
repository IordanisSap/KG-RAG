import bm25s
import Stemmer
from typing import List
from langchain_core.documents import Document


class BM25Indexer:
    def __init__(self, config):
        self.config = config
        self.stemmer = Stemmer.Stemmer("english")
        self.retriever = bm25s.BM25()
    
    def index(self, documents: List[Document], persist_dir):
        corpus = list(map(lambda doc: doc.page_content, documents))
        corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=self.stemmer)
        self.retriever.index(corpus_tokens)
        self.retriever.save(persist_dir, corpus=corpus)

