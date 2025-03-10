
from langchain_community.document_loaders.pdf import PyPDFLoader
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import logging

class DataLoader:
    def load_pdfs(self, dir_path: str):
        dir_path = Path(dir_path)
        pdf_files = list(dir_path.glob("*.pdf"))
        
        if not pdf_files:
            logging.warning("No PDF files found in the directory.")
            return []
        
        documents = []
        
        def load_single_pdf(file_path):
            logging.info(f"Loading {file_path.name}")
            return PyPDFLoader(str(file_path)).load()
        
        with ThreadPoolExecutor() as executor:
            results = executor.map(load_single_pdf, pdf_files)
        
        for doc in results:
            documents.extend(doc)

        return documents


