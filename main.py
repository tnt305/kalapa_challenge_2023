import json
import re
import argparse
import pandas as pd
import random
import config
from embedding.embedding import mE5Embedding
from llm.qwen import QwenInfer
from llm.gemma import GemmaInfer
from llm.mixtral import MixtralInfer
from llm.baichuan import BaichuanInfer
from utils import *
from tqdm import tqdm



#### List models you can play around
#vilm/vinallama-7b-chat
# model = GemmaInfer('01-ai/Yi-6B-Chat-4bits')MediaTek-Research/Breeze-7B-Instruct-v0_1
#Ashen2020/vivianne-medical-ai
# shibing624/ziya-llama-13b-medical-merged
# BioMistral/BioMistral-7B-SLERP
# ShengHongHaung/medical-everywhere-v0.1
# Qwen/Qwen1.5-14B-Chat
#model = MixtralInfer('/kaggle/input/mixtral/pytorch/8x7b-instruct-v0.1-hf/1')
#model = QwenInfer('MediaTek-Research/Breeze-7B-Instruct-v0_1')


def get_args_parser():
    parser = argparse.ArgumentParser( add_help=False)
    parser.add_argument('--model', type=str, choices=['QwenInfer', 'GemmaInfer', 'MixtralInfer', 'BaichuanInfer'], help='Specify the model name')
    parser.add_argument('--model_path', action = 'store_true', help = 'pretrain_model from huggingface')
    return parser


def main(args):
    # Determine which model to use based on the argument
    if args.model == 'QwenInfer':
        model = QwenInfer(args.model_path)  # Instantiate QwenInfer
    elif args.model == 'GemmaInfer':
        model = GemmaInfer(args.model_path)  # Instantiate GemmaInfer
    else:
        raise ValueError("Invalid model name specified.")

    df = pd.read_csv("/kaggle/input/kalapa-ner/public_test_with_ner.csv")
    result = {"id": [], "answer": []}
    for index, row in tqdm(df.iterrows()):
        result["id"].append(row["id"].strip())
        question, choices, num_choices = process_single_row(row)
        context = get_context(question, choices, top_k=10)
        question = preprocess_question(question)
        output, prompt = model.generate(question, choices, context)  # Use the selected model for generation
        output_json = process_output(output, num_choices)
        result["answer"].append(output_json)
        write_log(question, choices, context, prompt, output, output_json)

    newdf = pd.DataFrame(result, dtype=str)
    newdf.to_csv("/kaggle/working/submit1.csv", index=False)


if __name__ == "__main__":
    args  = get_args_parser()
    main(args)
