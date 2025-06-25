import csv
import re
import os
import nltk
from nltk.corpus import stopwords
import time

from KG_RAG.ingestion.data_loader import DataLoader

stopwordsPath = '/home/iordanissapidis/KG-RAG/examples/calculate_term_overlap/corenlp_stopwords.txt'
file_path = '/home/iordanissapidis/KG-RAG/examples/calculate_term_overlap/dataset/verbalized-grsf.csv'
pdfFolder = '/home/iordanissapidis/KG-RAG/examples/calculate_term_overlap/dataset'

stopwords = set()
# Load stopwords
with open(stopwordsPath, 'r') as stopwordsFile:
    for line in stopwordsFile:
        print(line.rstrip())
        stopwords.update(line)

unique_words = set()

with open(file_path, 'r', encoding='utf-8') as outfile:
    reader = csv.reader(outfile)
    for row in reader:
        for cell in row:
            words = re.findall(r'\b\w+\b', cell.lower())
            words = [word for word in words if word not in stopwords]
            unique_words.update(words)

print(len(unique_words))

dataloader = DataLoader()
docs = dataloader.load(pdfFolder, ['pdf'])

start = time.time()
with open("/home/iordanissapidis/KG-RAG/examples/calculate_term_overlap/out.txt", 'w', encoding='utf-8') as outfile:
    outfile.write(f"Started")

unique_words_doc = set()
for doc in docs:
    words = re.findall(r'\b\w+\b', doc.page_content.lower())
    words = [word for word in words if word not in stopwords]
    unique_words_doc.update(words)

# Print the count of unique words
print(f"Total unique words in KG: {len(unique_words)}")
print(f"Total unique words in docs: {len(unique_words_doc)}")
print(f"Total common words: {len(unique_words.intersection(unique_words_doc))}")
print(f"Total pages: {len(docs)}")

with open("/home/iordanissapidis/KG-RAG/examples/calculate_term_overlap/out.txt", 'w', encoding='utf-8') as outfile:
    outfile.write(f"Total unique words in KG: {len(unique_words)}\n")
    outfile.write(f"Total unique words in docs: {len(unique_words_doc)}\n")
    outfile.write(f"Total common words: {len(unique_words.intersection(unique_words_doc))}\n")
    outfile.write(f"{time.time()-start}\n")
    outfile.write(f"Items:\n")
    for item in unique_words.intersection(unique_words_doc):
        outfile.write(item + '\n')