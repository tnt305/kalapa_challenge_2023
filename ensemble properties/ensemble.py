import pandas as pd
import numpy as np
from tqdm import tqdm



df2 = pd.read_csv('./df_label.csv')
df_bm25 = pd.read_csv('./bm25+.csv')
df_gwen  = pd.read_csv('./qwen 7b 6434.csv')
df1 = df_bm25[['answer']].rename(columns ={'answer': 'bm25'})
df12 = df_gwen[['answer']].rename(columns ={'answer': 'qwen7b'})
df13 = df_gwen[['answer']].rename(columns ={'answer': 'mediatek'})
data  = pd.concat([df2, df1, df12, df13], axis =1)

# emsemble method follows the rules provided in the requirements
# this is exmaple ensemble, similar applied to second stage ensemble
import numpy as np

def get_answer(data):
    answer = []
    for index, row in data.iterrows():
        if (row['bm25'] == row['qwen7b']) or (row['bm25'] == row['qwen7b']) or (row['qwen7b'] == row['mediatek']):
            # Append both values if they are equal
            answer.append(row['bm25'])
        else: 
            items = ""
            result = ""
            if row['question_type'] == 'one answer':
                # choose the items in the results has the highest value
                for a, b, c in zip(str(row['bm25']), str(row['qwen7b']), str(row['mediatek'])):
                    sum_abc =  int(a) + int(b) + int(c)
                    items += str(sum_abc)

                max_item_indices = np.argsort([int(i) for i in items])[-2:]

                for i in range(len(items)):   
                    if len(max_item_indices) < 2:
                        if i ==  max_item_indices:
                            result += "1"
                        else:
                            result += "0"
                    else:
                        result = row['bm25']
            elif row['question_type'] == 'unknown':
                for a, b, c in zip(str(row['bm25']), str(row['qwen7b']), str(row['mediatek'])):
                    sum_abc =  int(a) + int(b) + int(c)
                    items += str(sum_abc)
                max_value =  max(items) #np.argsort([int(i) for i in items])

                for i in range(len(items)):
                    if max_value > 3:
                        if items[i] >= max_value - 2:
                            result += "1"
                        else:
                            result += "0"
                    else: 
                        if items[i] >= max_value - 1:
                            result += "1"
                        else:
                            result += "0"
            answer.append(result)
    return answer

data['answer'] = get_answer(data)