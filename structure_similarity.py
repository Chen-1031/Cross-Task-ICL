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
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

class GCN(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, output_dim=64, n_layer=2):
        super(GCN, self).__init__()
        self.n_layer = n_layer
        self.convs = nn.ModuleList([GCNConv(input_dim, hidden_dim)])
        for _ in range(n_layer - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, output_dim))

    def forward(self, x, edge_index):
        for i in range(self.n_layer):
            x = self.convs[i](x, edge_index)
            if i < self.n_layer - 1:
                x = F.relu(x)
        return x


def construct_task_graph(embeddings, weighted=False, top_k=20):
    n = len(embeddings)
    graph = nx.Graph()
    bar = tqdm(range(n), desc=f'construct graph')
    for i in range(n):
        if weighted:
            graph.add_edge(i, i, weight=1)
        else:
            graph.add_edge(i, i)
        cur_emb = embeddings[i].reshape(1, -1)
        cur_scores = F.cosine_similarity(cur_emb, embeddings)
        sorted_indices = torch.argsort(cur_scores, descending=True)[:top_k].numpy()
        for idx in sorted_indices:
            if idx != i:
                if weighted:
                    graph.add_edge(i, idx, weight=cur_scores[idx])
                else:
                    graph.add_edge(i, idx)
        bar.update(1)
    return graph





def add_nodes(embeddings, graph, emd, weighted=False, top_k=20):
    new_idx = embeddings.shape[0]
    cur_emb = emd.reshape(1, -1)
    cur_scores = F.cosine_similarity(cur_emb, embeddings)
    sorted_indices = torch.argsort(cur_scores, descending=True)[:top_k].numpy()
    graph.add_edge(new_idx, new_idx)
    for idx in sorted_indices:
        if idx != new_idx:
            if weighted:
                graph.add_edge(new_idx, idx, weight=cur_scores[idx])
            else:
                graph.add_edge(new_idx, idx)
    return graph
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
                      main_dir='results',suffix=''):
    ###### Load model #####

    if 'gpt' in model_name:
        from openai import OpenAI
        llm = OpenAI(
            api_key='YOUR_API_KEY')
    else:

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
    source_graph = construct_task_graph(source_dataset_embs, weighted)

    target_traindata = load_target_train(target_dataset_name)
    target_trainid = target_traindata['prompt_temp_id']
    target_traindataset = target_traindata['data']
    target_inst = get_instruction(target_dataset_name)
    target_traindataset_embs = torch.stack([d['emb'] for d in target_traindataset], dim=0)
    target_graph = construct_task_graph(target_traindataset_embs, weighted)

    target_data = load_dataset('target', target_dataset_name, 'test')
    target_id = target_data['prompt_temp_id']
    target_dataset = target_data['data']

    demo = True
    task_demo = ''
    a = Tasks()

    ############ compute similarity ############
    GNNmodels=[]
    source_feat_out=[source_dataset_embs]
    ##fixed aggregation
    A = nx.adjacency_matrix(source_graph).todense()
    A = torch.tensor(np.array(A), dtype=torch.float)
    A_2hop = torch.mm(A, A)
    source_feat_out.append(torch.mm(A, source_dataset_embs))
    source_feat_out.append(torch.mm(A_2hop, source_dataset_embs))
    ##fixed aggregation

    source_graph = from_networkx(source_graph)
    if suffix == " ":
        nlayers=[1,1,2,2]
    else:
        if '_' in suffix:
            numbers_str, _ = suffix.split('_')
            nlayers = [int(char) for char in numbers_str]
        else:
            nlayers = [int(char) for char in suffix]
    print(nlayers)
    for random_seed in range(len(nlayers)):
        torch.manual_seed(random_seed)
        if '_' in suffix:
            _, outdim = suffix.split('_')
            outdim = int(outdim)
        else:
            outdim = target_traindataset_embs.shape[-1]
        GNNmodel = GCN(input_dim=target_traindataset_embs.shape[-1], hidden_dim=target_traindataset_embs.shape[-1],
            output_dim=outdim, n_layer=nlayers[random_seed])
        GNNmodels.append(GNNmodel)
        output = GNNmodel(source_dataset_embs, source_graph.edge_index)
        source_feat_out.append(output)

    ids = []
    prompts = []
    preds = []
    gold_labels = []
    #for data in target_dataset:
    for data in tqdm(target_dataset,
                         desc=f"Evaluating {target_dataset_name} with Source {source_dataset_name}"):
        t_data = data.copy()
        t_e = t_data['emb']
        target_feat=[t_data['emb']]
        t_p = prompter.get_input_data_without_def(t_data, target_id)
        new_target_graph = add_nodes(target_traindataset_embs, target_graph, t_data['emb'], weighted)
        add_embeds = torch.cat((target_traindataset_embs, t_e.unsqueeze(0)), dim=0)
        ##fixed aggregation
        graphdata = nx.adjacency_matrix(new_target_graph).todense()
        graphdata = torch.tensor(np.array(graphdata), dtype=torch.float)
        graphdata_2hop = torch.mm(graphdata, graphdata)
        target_feat.append(torch.mm(graphdata, add_embeds)[-1])
        target_feat.append(torch.mm(graphdata_2hop, add_embeds)[-1])
        ##fixed aggregation

        graphdata = from_networkx(new_target_graph)

        for GNNmodel in GNNmodels:
            output = GNNmodel(add_embeds, graphdata.edge_index)[-1]
            target_feat.append(output)


        cosine_similarities = []

        for i in range(len(target_feat)):
            cur_emb = target_feat[i].reshape(1, -1)
            cosine_sim = F.cosine_similarity(cur_emb, source_feat_out[i])
            cosine_similarities.append(cosine_sim)
        average_similarity = torch.stack(cosine_similarities).mean(dim=0)

        #max_idx = torch.argmax(average_similarity)
        top_k = torch.argsort(average_similarity, descending=True)[:5].numpy()
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
        if 'gpt' in model_name:

            messages = [{"role": "user", "content": few_shot_prompt}]
            completion = llm.chat.completions.create(model=model_name, messages=messages, n=1, max_tokens=5,
                                                     stop=['--', '\n\n', ';', '#'], temperature=0)
            prediction = [choice.message.content for choice in completion.choices]
            ids.append(data['id'])
            if type(data['label']) is list:
                gold_labels.append(data['label'][0])
            else:
                gold_labels.append(data['label'])
            prompts.append(few_shot_prompt)
            preds.append(prediction)
        else:
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

    result_path = os.path.join(result_dir,
                               f"source_dataset={source_dataset_name}-model={model_name}-method=graphsim-shots={k}_{suffix}.csv")

    result_df = pd.DataFrame(eval_dic)
    result_df.to_csv(result_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dataset_name', required=True, type=str)
    parser.add_argument('--target_dataset_name', required=True, type=str)
    parser.add_argument('--model_name', default="meta-llama/Llama-2-7b-hf", type=str)
    parser.add_argument('--device',default="cuda:0", required=True, type=str)
    parser.add_argument('--suffix', default=" ", type=str)
    parser.add_argument('--k', default=1, type=int)
    args = parser.parse_args()

    graph_based_selection(args.source_dataset_name, args.target_dataset_name, args.model_name, args.device, args.k,'results',args.suffix)
