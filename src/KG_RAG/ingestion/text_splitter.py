from langchain_text_splitters import RecursiveCharacterTextSplitter

class Splitter:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
        )

    def split_documents(self, docs):
        chunks = self.text_splitter.split_documents(docs)  # Process all docs at once
        for chunk in chunks:
            chunk.metadata['title'] = chunk.metadata.get('title', 'Unknown Title')  # Preserve metadata if exists
        return chunks


    def split_text(self, text):
        return self.text_splitter.split_text(text)
    