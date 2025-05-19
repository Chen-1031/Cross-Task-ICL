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
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from intask_utils import create_new_dataset, pack_data

prompt_temps_dic = {
    '001': 'Premise: {} \nHypothesis: {} \nLabel:',
    '002': 'Question 1: {} \nQuestion 2: {} \nLabel:',
    # '003': 'Context: {} \nQuestion: {} \nAnswer:',
    '003': 'Context: {}\nAnswer: {}',
    '004': 'Sentence: {} \nLabel:',
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



def predict_result(dataset_name, qa_data, model, model_name, device='cuda:0', k=1):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from utils.data import get_instruction, load_dataset, Prompt
    from llama import make_prompt,call_model
    if 'gpt' in model_name:
        from openai import OpenAI
        llm = OpenAI(
            api_key='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')

        target_data = load_dataset('target', dataset_name, 'test')
        target_id = target_data['prompt_temp_id']
        target_dataset = target_data['data']
        task_inst = get_instruction(dataset_name)
        prompter = Prompt()

        prompt = query_temps_dic[target_id]

        query_embeddings = [model.encode(prompt.format(data['question'])) for data in qa_data]
        example_query_embeddings = torch.tensor(np.array(query_embeddings), dtype=torch.float)

        example_prompts = [(prompt.format(d['question']) + " " + str(d['answer'])) for d
                           in qa_data]

        demo = True
        task_demo = ''

        ids = []
        prompts = []
        preds = []
        gold_labels = []
        # for data in target_dataset:
        for data in tqdm(target_dataset[:250],
                         desc=f"Evaluating {dataset_name} "):
            t_data = data.copy()
            t_e = t_data['emb']
            t_e = torch.unsqueeze(t_e, dim=0)
            t_p = prompter.get_input_data_without_def(t_data, target_id)
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

            messages = [{"role": "user", "content": few_shot_prompt}]
            completion = llm.chat.completions.create(model="gpt-3.5-turbo", messages=messages, n=1, max_tokens=5,
                                                     stop=['--', '\n\n', ';', '#'], temperature=0)
            prediction = [choice.message.content for choice in completion.choices]
            ids.append(data['id'])
            if type(data['label']) is list:
                gold_labels.append(data['label'][0])
            else:
                gold_labels.append(data['label'])
            prompts.append(few_shot_prompt)
            preds.append(prediction)
            # break

        eval_dic = dict()
        eval_dic['id'] = ids
        # eval_dic['prompt'] = prompts
        eval_dic['pred'] = preds
        eval_dic['true_label'] = gold_labels

        df = pd.DataFrame(eval_dic)
        df = df.fillna('')
        result_path = f"/home/zihan/CD_ICL/results/{dataset_name}_intask_gt_K{k}.csv"
        df.to_csv(result_path, index=False)

        acc = 0


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

        target_data = load_dataset('target', dataset_name, 'test')
        target_id = target_data['prompt_temp_id']
        target_dataset = target_data['data']
        task_inst = get_instruction(dataset_name)

        ###########


        prompter = Prompt()

        prompt = query_temps_dic[target_id]

        query_embeddings = [model.encode(prompt.format(data['question'])) for data in qa_data]
        example_query_embeddings = torch.tensor(np.array(query_embeddings), dtype=torch.float)
        example_prompts = [(prompt.format(d['question']) + " " + str(d['answer'])) for d
                           in qa_data]

        demo = True
        task_demo = ''

        ids = []
        prompts = []
        preds = []
        gold_labels = []
        # for data in target_dataset:
        for data in tqdm(target_dataset,
                         desc=f"Evaluating {dataset_name} "):
            t_data = data.copy()
            t_e = t_data['emb']
            t_e = torch.unsqueeze(t_e, dim=0)
            t_p = prompter.get_input_data_without_def(t_data, target_id)
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

            prediction, response = generate(few_shot_prompt, max_new_tokens=15)

            ids.append(data['id'])
            if type(data['label']) is list:
                gold_labels.append(data['label'][0])
            else:
                gold_labels.append(data['label'])
            prompts.append(few_shot_prompt)
            preds.append(prediction)
            # break

        eval_dic = dict()
        eval_dic['id'] = ids
        # eval_dic['prompt'] = prompts
        eval_dic['pred'] = preds
        eval_dic['true_label'] = gold_labels

        df = pd.DataFrame(eval_dic)
        df = df.fillna('')
        output = list(df['pred'])
        gold = np.array(df['true_label'])
        gold = [str(o).lower() for o in gold]

        output = [str(o).replace('Label:', '') for o in output]
        output = [o.strip(' .[]":\'').lower() for o in output]
        output = [o.split('.')[0] for o in output]
        output = [o.split(',')[0] for o in output]
        output = [o.split(':')[0] for o in output]
        output = [o.split('-')[0] for o in output]
        output = np.array(output)
        acc = np.mean(output == gold) * 100
        print(f"Acc for {dataset_name} is ", acc)
    return acc



dataset_names=['ARC-Challenge', 'medmcqa','financial_phrasebank','sciq','social_i_qa']
# dataset_name = 'financial_phrasebank'
# dataset_name='social_i_qa'
# dataset_name = 'ARC-Challenge'
# dataset_name='medmcqa'
# dataset_name='sciq'
result={}
for dataset_name in dataset_names:
    qa_data, prompt_id = pack_data(dataset_name,seed=42)
    prompt = prompt_temps_dic[prompt_id]
    # TODO: before construct graph maybe random shuffle the queries first.

    # Step 1: Compute Embeddings for (Query, Answer) Pairs
    query_choice_embeddings = []
    labels = []
    query_indices = []  # Track which query each answer belongs to
    labeled_mask = []

    # Load embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    np.random.seed(42)

    model_name="meta-llama/Llama-2-13b-hf"
    model_name="gpt-3.5-turbo"
    acc=predict_result(dataset_name, qa_data, model, model_name, device='cuda:0', k=1)
    result[dataset_name] = acc
print(result)
# Print resultss

