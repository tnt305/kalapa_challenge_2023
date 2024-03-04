import glob
import os
from typing import List
from tqdm import tqdm
import pandas as pd
from langchain.schema.document import Document
from langchain.vectorstores.chroma import Chroma
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from embedding import config 
from embedding.embedding import mE5Embedding

db_raw = None
# db_raw = Chroma(collection_name="kalapa_medical_corpus_raw",
#             embedding_function=mE5Embedding(),
#             persist_directory=config.VECTORSTORES_LOCAL,
#             collection_metadata={"hnsw:space": "cosine"})

db_clean = Chroma(collection_name="kalapa_medical_corpus_clean",
                  embedding_function=mE5Embedding(),
                  persist_directory= config.VECTORSTORES_LOCAL,
                  collection_metadata={"hnsw:space": "cosine"})


# def run_embedding_raw_corpus(list_corpus_path: List = [], chunk_size=200, chunk_overlap=50):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
#                                                    chunk_overlap=chunk_overlap,
#                                                    separators=["\n", "."])
#     for document_path in list_corpus_path:
#         raw_documents = TextLoader(document_path).load()
#         documents = text_splitter.split_documents(raw_documents)
#         for text in documents:
#             db_raw.add_texts(texts=[f"passage: {text.page_content}"], metadatas=[text.metadata])


def run_embedding_clean_corpus(corpus_path, chunk_size=200, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                   chunk_overlap=chunk_overlap,
                                                   separators=["\n", "."])
    df = pd.read_csv(corpus_path)
    for index, row in tqdm(df.iterrows()):
        raw_documents = [Document(page_content=f"{row['question']}\n{row['answer']}", metadata={"source": row["name"]})]
        print(row['question'])
        # documents = text_splitter.split_documents(raw_documents)
        for text in raw_documents:
            db_clean.add_texts(texts=[f"passage: {text.page_content}"], metadatas=[text.metadata])




if __name__ == "__main__":
    list_corpus_path = glob.glob(os.path.join("/kaggle/working/MEDICAL/corpus/corpus", "*"))

    # run_embedding_raw_corpus(list_corpus_path, chunk_size=800, chunk_overlap=200)
    # run_embedding_clean_corpus(corpus_path="./clean_v2.csv", chunk_size=800, chunk_overlap=200)
    run_embedding_clean_corpus(corpus_path="/kaggle/input/kalapa/klp_clean_v3.csv", chunk_size=800, chunk_overlap=200)

