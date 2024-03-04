import pandas as pd
from tqdm import tqdm


df = pd.read_csv('../kalapa-ner/public_test_with_ner.csv')

def gen_answer_type(df):
    q_type = []
    for question in tqdm(df['question']):
        question = question.replace("?.", "?")
        if question[-1] != "?":
            question += "?"
        type_predict = ''
        if "là gì" in question.lower() and not any(word in question.lower() for word in ["các", "những"]):
            type_predict = "one answer"
        elif 'nhất' in question.lower():
            type_predict = "one answer"
        elif "cách gì" in question.lower() and not any(word in question.lower() for word in ["các", "những"]):
            type_predict = "one answer"
        elif 'bệnh gì' in question.lower() and not any(word in question.lower() for word in ["các", "những"]):
            type_predict = "one answer"
        elif 'bệnh nào' in question.lower() and not any(word in question.lower() for word in ["các", "những"]):
            type_predict = "one answer"
        elif 'bao nhiêu' in question.lower():
            type_predict = "one answer"
        elif 'thứ mấy' in question.lower():
            type_predict = "one answer"
        elif 'bệnh lý nào' in question.lower():
            type_predict = "one answer"
        elif 'định nghĩa' in question.lower():
            type_predict = "one answer"
        elif "đúng hay sai" in question.lower():
            type_predict = "one answer"
        elif "có phải" in question.lower():
            type_predict = "one answer"
        elif "hay không" in question.lower():
            type_predict = "one answer"
        else:
            type_predict = "unknown"
        q_type.append(type_predict)
    return q_type 

def gen_length(df):
    len_answer = []
    for index, row in tqdm(df.iterrows()):
        list_answer = [str(row["option_1"]), str(row["option_2"]), str(row["option_3"]), str(row["option_4"]),
                    str(row["option_5"]), str(row["option_6"])]
        #print(len([item for item in list_answer if item != 'nan']))
        len_answer.append(len([item for item in list_answer if item != 'nan']))
    
    return len_answer


def main(df):
    q_type = gen_answer_type(df)
    len_answer  = gen_length(df)
    df['question_type'] = q_type
    df['len_answer'] = len_answer
    df_label = df[['id','question_type', 'len_answer']]
    df_label.to_csv("df_label1.csv", index = False)

df = pd.read_csv('../kalapa-ner/public_test_with_ner.csv')
main(df)