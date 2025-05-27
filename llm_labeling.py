import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import transformers
import torch, json, math
import random
import numpy as np
import os, pickle
from tqdm import tqdm
import argparse
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from intask_utils import predict_result, create_new_dataset, pack_data


prompt_temps_dic = {
    '001': 'Premise: {} \nHypothesis: {} \nLabel:',
    '002': 'Question 1: {} \nQuestion 2: {} \nLabel:',
    # '003': 'Context: {} \nQuestion: {} \nAnswer:',
    '003': 'Context: {}\nAnswer: {}',
    '004': 'Sentence: {} \nLabel: {}',
    '005': 'Question: {} \nAnswer: {}',
    '006': 'Sentence 1: {} \nSentence 2: {} \nLabel:',
    '007': 'Question: {} \nSentence: {} \nLabel:'
}
query_temps_dic = {
    '001': 'Premise: {} \nHypothesis: {} \nLabel:',
    '002': 'Question 1: {} \nQuestion 2: {} \nLabel:',
    # '003': 'Context: {} \nQuestion: {} \nAnswer:',
    '003': 'Context: {} \nAnswer:',
    '004': 'Sentence: {} \nLabel:',
    '005': 'Question: {} \nAnswer:',
    '006': 'Sentence 1: {} \nSentence 2: {} \nLabel:',
    '007': 'Question: {} \nSentence: {} \nLabel:'
}


def LLM_pseudo_labeling(dataset_name, prompt_id, qa_data, labeled_idxs, model, model_name, device='cuda:0', k=1):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from utils.data import get_instruction, load_dataset, Prompt
    from llama import make_prompt,call_model
    if 'gpt' in model_name:
        from openai import OpenAI
        llm = OpenAI(
            api_key='YOUR_API_KEY')
    else:
        llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map=device,  # device
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        generate = lambda prompt, max_new_tokens: call_model(prompt, model=llm, tokenizer=tokenizer, device=device)
        model_name = model_name.replace("/", "_")

    labeled_data = [qa_data[i] for i in labeled_idxs]

    prompter = Prompt()

    prompt = query_temps_dic[prompt_id]

    query_embeddings = [model.encode(prompt.format(data['question'])) for data in labeled_data]
    example_query_embeddings = torch.tensor(np.array(query_embeddings), dtype=torch.float)
    example_prompts = [(prompt.format(d['question']) + " " + str(d['answer'])) for d
                       in labeled_data]

    demo = True
    task_demo = ''
    task_inst = get_instruction(dataset_name)
    for i, data in enumerate(qa_data):
        if i in labeled_idxs:
            data['pseudo_label'] = data['answer']
        if i not in labeled_idxs:
            t_data = data.copy()
            t_e = model.encode(prompt.format(t_data['question']))
            t_e = torch.unsqueeze(torch.tensor(t_e), dim=0)
            t_p = prompt.format(t_data['question'])
            sim = F.cosine_similarity(t_e, example_query_embeddings)
            top_k = torch.argsort(sim, descending=True)[:k].numpy()
            entries = [example_prompts[k] for k in top_k]

            few_shot_prompt = make_prompt(
                source_prompts=entries,
                source_instruction=task_inst,
                target_prompts=t_p,
                target_instruction=task_inst,
                k=k,
                task_demo=task_demo,
                demo=demo)

            #######
            if 'gpt' in model_name:

                messages = [{"role": "user", "content": few_shot_prompt}]
                completion = llm.chat.completions.create(model=model_name, messages=messages, n=1, max_tokens=5,
                                                         stop=['--', '\n\n', ';', '#'], temperature=0)
                prediction = [choice.message.content for choice in completion.choices]
                data['pseudo_label'] = prediction[0]
            else:
                prediction, response = generate(few_shot_prompt, max_new_tokens=15)
                data['pseudo_label'] = prediction


    return qa_data

parser = argparse.ArgumentParser()
parser.add_argument('-k', default=1, type=int)
parser.add_argument('-m', default="meta-llama/Llama-2-7b-hf", type=str)
parser.add_argument('-d', default="0", type=str)
args = parser.parse_args()


dataset_names=['ARC-Challenge', 'medmcqa','financial_phrasebank','sciq','social_i_qa']

result={}
for dataset_name in dataset_names:
    qa_data, prompt_id = pack_data(dataset_name,seed=42)
    prompt = prompt_temps_dic[prompt_id]

    # Step 1: Compute Embeddings for (Query, Answer) Pairs
    query_choice_embeddings = []
    labels = []
    query_indices = []  # Track which query each answer belongs to
    labeled_mask = []

    # Load embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    np.random.seed(42)
    labeled_idxs = np.random.choice(len(qa_data), 100, replace=False)

    model_name = args.m
    qa_data = LLM_pseudo_labeling(dataset_name, prompt_id, qa_data, labeled_idxs, model, model_name, device=f'cuda:{args.d}', k=4)
    acc=predict_result(dataset_name, qa_data, model, model_name, device=f'cuda:{args.d}', k=arg.k)
    result[dataset_name] = acc
print(result)


