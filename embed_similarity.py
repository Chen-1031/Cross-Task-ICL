import pandas as pd
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import openai
from utils.data import load_prompts,load_dataset, get_instruction, load_target_train
import argparse
from tqdm import tqdm
from utils.data import Prompt
from utils.data import Tasks
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import networkx as nx
import torch.nn as nn
import torch.nn.functional as F

def call_model(prompt, model, tokenizer, device, max_new_tokens=10, model_max_length=2048):
    max_inpt_tokens = tokenizer.model_max_length if model_max_length is None else model_max_length

    inpts = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_inpt_tokens - max_new_tokens).to(
        device)
    # print(len(inpts.input_ids[0]))
    gen = model.generate(input_ids=inpts.input_ids[:, -(max_inpt_tokens - max_new_tokens):],
                         attention_mask=inpts.attention_mask[:, -(max_inpt_tokens - max_new_tokens):],
                         pad_token_id=tokenizer.eos_token_id, max_new_tokens=max_new_tokens, num_beams=1,
                         do_sample=False)
    # gen = model.generate(input_ids=inpts.input_ids, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id, num_beams=1, do_sample=False)
    text = tokenizer.decode(gen[0])
    actual_prompt = tokenizer.decode(inpts.input_ids[0, -(max_inpt_tokens - max_new_tokens):])
    # actual_prompt = tokenizer.decode(inpts.input_ids[0])
    pred = text[len(actual_prompt):]
    if pred.startswith("\n\n"):
        pred = pred[2:]
    pred = pred.split("\n")[0]
    return pred, text


def make_prompt(source_prompts, source_instruction, target_prompts, target_instruction, k, task_demo='', demo=False):
    demonstration_part = ''
    if demo:
        demonstration_part = 'Definition: ' + source_instruction + '\n' + "\n".join(source_prompts[:k]) + '\n'
    input_part = 'Definition: ' + target_instruction + '\n' + task_demo + target_prompts

    return demonstration_part + input_part


def graph_based_selection(source_dataset_name, target_dataset_name, model_name, device='cuda:2', weighted=False, k=1,
                      main_dir='results'):
    ###### Load model #####

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map=device,  # device
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    generate = lambda prompt, max_new_tokens: call_model(prompt, model=model, tokenizer=tokenizer, device=device)
    model_name = model_name.replace("/", "_")
    ###########

    result_dir = os.path.join(main_dir, f"dataset_name={target_dataset_name}")

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    ############ load dataset ############
    prompter = Prompt()

    source_data = load_dataset('source', source_dataset_name, 'train')
    source_id = source_data['prompt_temp_id']
    source_dataset = source_data['data']
    source_inst = get_instruction(source_dataset_name)

    source_dataset_prompts = [(prompter.get_input_data_without_def(d, source_id) + " " + str(d['label'])) for d
                              in source_dataset]
    source_dataset_embs = torch.stack([d['emb'] for d in source_dataset], dim=0)

    target_inst = get_instruction(target_dataset_name)

    target_data = load_dataset('target', target_dataset_name, 'test')
    target_id = target_data['prompt_temp_id']
    target_dataset = target_data['data']


    demo = True
    task_demo = ''
    a = Tasks()


    ids = []
    prompts = []
    preds = []
    gold_labels = []
    #for data in target_dataset:
    for data in tqdm(target_dataset[:250],
                         desc=f"Evaluating {target_dataset_name} with Source {source_dataset_name}"):
        t_data = data.copy()
        t_e = t_data['emb']
        t_e = torch.unsqueeze(t_e, dim=0)
        t_p = prompter.get_input_data_without_def(t_data, target_id)
        sim = F.cosine_similarity(t_e, source_dataset_embs)
        top_k = torch.argsort(sim, descending=True)[:k].numpy()
        entries = [source_dataset_prompts[k] for k in top_k]

        few_shot_prompt = make_prompt(
            source_prompts=entries,
            source_instruction=source_inst,
            target_prompts=t_p,
            target_instruction=target_inst,
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
    #eval_dic['prompt'] = prompts
    eval_dic['pred'] = preds
    eval_dic['true_label'] = gold_labels
    if weighted:
        result_path = os.path.join(result_dir,
                               f"source_dataset={source_dataset_name}-model={model_name}-method=graphbased-shots={k}_bsl.csv")
    else:
        result_path = os.path.join(result_dir,
                               f"source_dataset={source_dataset_name}-model={model_name}-method=graphbased-shots={k}_bsl.csv")
    result_df = pd.DataFrame(eval_dic)
    result_df.to_csv(result_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dataset_name', required=True, type=str)
    parser.add_argument('--target_dataset_name', required=True, type=str)
    parser.add_argument('--method', default='totally-cross-sim', type=str)
    parser.add_argument('--model_name', default="meta-llama/Llama-2-7b-hf", type=str)
    parser.add_argument('--device', default="cuda:0", required=True, type=str)
    parser.add_argument('--suffix', default=" ", type=str)
    parser.add_argument('--weighted', default=False, type=bool)
    parser.add_argument('--k', default=1, type=int)
    args = parser.parse_args()


    graph_based_selection(args.source_dataset_name, args.target_dataset_name, args.model_name, args.device, args.weighted, args.k,'results')