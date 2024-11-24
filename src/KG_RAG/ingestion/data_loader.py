
import os
from langchain_community.document_loaders.pdf import PyPDFLoader

from ..utils import log

class DataLoader:
    def load_pdfs(self, dir: str):
        documents = []
        pdf_files = [file for file in os.listdir(dir) if file.endswith(".pdf")]
        for i,file in enumerate(pdf_files):
                log("({0}/{1}) Loading {2}".format(i + 1,len(pdf_files),file))
                file_path = os.path.join(dir, file)
                doc = PyPDFLoader(file_path).load()
                documents.append(doc)
        return documents


