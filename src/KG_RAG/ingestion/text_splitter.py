from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..utils import log

class Splitter:
    def __init__(self, chunk_size=600, chunk_overlap=0):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
        )

    def split_documents(self, docs):
        chunks = []
        for i,doc in enumerate(docs):
            splits = self.text_splitter.split_documents(doc)
            log(f"({i+1}/{len(docs)}) Splitting document into {len(splits)} chunks")
            # Add metadata to each chunk
            for split in splits:
                split.metadata['title'] = 'Paper Title'
            chunks.extend(splits)
        return chunks


    def split_text(self, text):
        return self.text_splitter.split_text(text)
    