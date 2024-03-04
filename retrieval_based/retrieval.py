from retrieval_based.utils import *
from retrieval_based.retrieval_choice import *
from tqdm import tqdm
from bs4 import BeautifulSoup
import os
import numpy as np
from tqdm import tqdm
import time
import re
import matplotlib.pyplot as plt
import numpy as np
import faiss
from faiss import write_index, read_index
import pandas as pd
from rank_bm25 import BM25Okapi, BM25Plus
import pickle


df = pd.read_csv('/content/MEDICAL/public_test.csv')
l_questions = df["question"].tolist()
l_answer_per_questions = []
l_columns = ['option_1','option_2','option_3','option_4','option_5','option_6']
for index, row in df.iterrows():
    l_answers = []
    for column in l_columns:
        if not isinstance(row[column], str):
            continue
        else:
            l_answers.append(clean_answer(row[column]))
    l_answer_per_questions.append(l_answers)





file_names = []
corpus = []
import os
for file_name in os.listdir('./MEDICAL/corpus'):
    with open(f'./MEDICAL/corpus/corpus/{file_name}', 'r') as f:
        doc = f.readlines()

    file_names.append(" ".join(file_name.split("-")))
    corpus.append(" ".join(doc))

titles = get_titles(file_names, corpus)

documents = []
for title, document in zip(titles, corpus):
    doc_dict = process_document(title, document)
    documents.append(doc_dict)


l_sub_corpus = []
l_idx_corpus = []
for idx, corpus in tqdm(enumerate(documents)):
    sub_corpus = prepare_sentence_for_encode(corpus)
    l_sub_corpus += sub_corpus
    l_idx_corpus += [idx for item in sub_corpus]

tokenized_corpus = [tien_xu_li(doc).split(" ") for doc in l_sub_corpus_new]
bm25 = BM25Okapi(tokenized_corpus, k1= 1.5, b= 0.7, epsilon= 0.7)

def main():
    l_predict_df = []
    for idx_qa, question in enumerate(l_questions):

        question = tien_xu_li(question)

        tokenized_query = question.split(" ")
        l_idxs = [i for i in range(len(l_sub_corpus))]

        search_idx_selected = bm25.get_top_n(tokenized_query, l_idxs , n= 5)                         
        answer_per_question = l_answer_per_questions[idx_qa]
        l_subcorpus_matched = [l_sub_corpus[idx] for idx in search_idx_selected] 

        embedding_selected_idx = get_top_1_embedding(question, l_subcorpus_matched)
        idx_corpus_matched = l_idx_corpus[search_idx_selected[embedding_selected_idx]] 
        selected_sentences = find_relavant_in_current_document(idx_corpus_matched, documents,
                                                            question, answer_per_question)

        prediction_str = get_predict(answer_per_question, question, selected_sentences)
        l_predict_df.append(prediction_str)
    
    return l_predict_df

df_public = pd.read_csv('./MEDICAL/public_test.csv')
l_id = df_public["id"].tolist()

df_submit = pd.DataFrame()
df_submit["id"] = l_id
df_submit["answer"] = main()
df_submit.to_csv("bm25+.csv", index = False)