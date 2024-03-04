from langchain.vectorstores.chroma import Chroma
import random
from embedding import config
from embedding.embedding import mE5Embedding

medical_corpus_db = Chroma(collection_name="kalapa_medical_corpus_clean",
                           embedding_function=mE5Embedding(),
                           persist_directory=config.VECTORSTORES_LOCAL,
                           collection_metadata={"hnsw:space": "cosine"})

embedding_model = mE5Embedding()

def litm_reordering(documents):
    """Los in the middle reorder: the most relevant will be at the
    middle of the list and more relevant elements at beginning / end.
    See: https://arxiv.org/abs//2307.03172"""

    tmp_documents = list(reversed(documents))
    reordered_result = []
    for i, value in enumerate(tmp_documents):
        if i % 2 == 1:
            reordered_result.append(value)
        else:
            reordered_result.insert(0, value)
    return reordered_result

def write_log(question, choices, context, prompt, output, output_json):
    with open("./logs.txt", "a+") as f:
        f.write(f"Context:\n{context}\n")
        f.write(f"Question: {question}\n")
        f.write(f"Choices:\n{choices}\n")
        f.write(f"Prompt:\n{prompt}\n")
        f.write(f"Output: {output}\n")
        f.write(f"Output JSON: {output_json}\n")
        f.write("------------------------------------------------------------------------------\n")


def process_single_row(row):
    question = row["question"].strip()
    list_answer = [str(row["option_1"]), str(row["option_2"]), str(row["option_3"]), str(row["option_4"]),
                   str(row["option_5"]), str(row["option_6"])]
    tmp_ans = []
    for c, a in zip(["A", "B", "C", "D", "E", "F"], list_answer):
        if a in ["nan", "", "NaN"]:
            continue
        if a.startswith(c):
            tmp_ans.append(a)
            continue
        tmp_ans.append(f"{c} {a}")
    answer_choices = "\n".join(tmp_ans)
    return question, answer_choices, len(tmp_ans)

def preprocess_question(question):
    question = question.replace("?.", "?")
    if question[-1] != "?":
        question += "?"
    type_predict = 0
    if "là gì" in question.lower() and not any(word in question.lower() for word in ["các", "những"]):
        type_predict = 1
    if "cách gì" in question.lower() and not any(word in question.lower() for word in ["các", "những"]):
        type_predict = 1
    if 'bệnh gì' in question.lower() and not any(word in question.lower() for word in ["các", "những"]):
        type_predict = 1
    if 'bệnh nào' in question.lower() and not any(word in question.lower() for word in ["các", "những"]):
        type_predict = 1
    if 'bệnh lý nào' in question.lower():
        type_predict = 1
    if 'bao nhiêu' in question.lower():
        type_predict = 1
    if 'nhất' in question.lower():
        type_predict = 1
    if 'định nghĩa' in question.lower():
        type_predict = 1
    if "đúng hay sai" in question.lower():
        type_predict = 2
    if "có phải" in question.lower():
        type_predict = 2
    if "hay không" in question.lower():
        type_predict = 2
    if type_predict == 1 or type_predict == 2:
        question += " (仅选择 1 个正确答案。)"
    else:
        question += " (您必须选择 2 个或更多答案。)"
    return question
  
def get_context(question, answer, top_k, use_litm=True, use_mmr=True):
    
    if 'dưới đây' not in question:
        embedding_vector = embedding_model.embed_documents([f"query: {question}"])
        if use_mmr:
            docs = medical_corpus_db.max_marginal_relevance_search_by_vector(embedding_vector[0], k=top_k, lambda_mult = 0.7)
        else:
            docs = medical_corpus_db.similarity_search_by_vector(embedding_vector[0], k=top_k)
        sources = {}
        context = []
        for doc in docs:
            if doc.metadata["source"] not in sources:
                sources[doc.metadata["source"]] = 1
            else:
                sources[doc.metadata["source"]] += 1

            if sources[doc.metadata["source"]] >= 3:
                continue
            context.append(doc.page_content)
        context = context[:6]
    else:
        searches = answer.split('\n')
        context = []
        for s in searches:
            embedding_vector = embedding_model.embed_documents([f"query: {question}. {s}"])
            docs = medical_corpus_db.similarity_search_by_vector(embedding_vector[0], k= 1)
            context += [doc.page_content for doc in docs]


    context = list(set(context))
    if use_litm:
        context = litm_reordering(context)
    return "\n".join([c.replace("passage: ", "") for c in context])

def extract_letters(line):
    matches = re.findall(r'\b([A-F])\b', line)
    return matches

def process_output(output, num_ans):
    res = ""
    MAP_ANS = {0:"a", 1:"b", 2:"c", 3:"d", 4:"e", 5:"f"}
    # output = re.split(r"[.,、]", output)
    # output = [c.strip().lower() for c in output if len(c)]
    output = [c.lower() for c in extract_letters(output)]
    print("OUTPUT: ", output)
    for i in range(num_ans):
        if MAP_ANS[i] in output:
            res += "1"
        else:
            res += "0"
    if all(c == "0" for c in res):
        # Randomly select an index to change to 1
        index_to_change = random.randint(0, num_ans - 1)
        res = res[:index_to_change] + "1" + res[index_to_change + 1:]
    print(res)
    return res