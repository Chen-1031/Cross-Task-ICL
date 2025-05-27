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
    '004': 'Sentence: {} \nLabel:',
    '005': 'Question: {} \nAnswer: {}',
    '006': 'Sentence 1: {} \nSentence 2: {} \nLabel:',
    '007': 'Question: {} \nSentence: {} \nLabel:'
}


dataset_names=['ARC-Challenge', 'medmcqa','financial_phrasebank','sciq','social_i_qa']


parser = argparse.ArgumentParser()
parser.add_argument('-k', default=1, type=int)
parser.add_argument('-m', default="meta-llama/Llama-2-7b-hf", type=str)
parser.add_argument('-d', default="0", type=str)
args = parser.parse_args()

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

    for i, data in enumerate(qa_data):

        for choice in data["choices"]:

            qa_pari_embed = model.encode(prompt.format(data['query'], data["choices"][choice]))  ##TODO: IF for different task

            query_choice_embeddings.append(qa_pari_embed)
            query_indices.append(i)  # Store query ID

            if i in labeled_idxs:
                if choice == data["answer"]:
                    labels.append(1)  # Correct answer → 1
                    labeled_mask.append(True)  # Labeled
                else:
                    labels.append(0)  # Incorrect answer → 0
                    labeled_mask.append(True)
            else:
                labels.append(-1)  # Unlabeled node
                labeled_mask.append(False)

    query_choice_embeddings = torch.tensor(np.array(query_choice_embeddings), dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)
    labeled_mask = torch.tensor(labeled_mask, dtype=torch.bool)

    # Step 2: Build Graph Edges (Positive & Negative)
    n_nodes = len(query_choice_embeddings)
    similarity_matrix = torch.tensor(cosine_similarity(query_choice_embeddings))
    threshold = 0.7  # Similarity threshold for positive edges

    positive_edges = []
    negative_edges = []

    print("Constructing Graph")
    # Create positive edges (Similarity-based edges)
    for i in range(n_nodes):
        sorted_indices = torch.argsort(similarity_matrix[i], descending=True)[:15].numpy()
        for j in sorted_indices:
            if j != i:
                positive_edges.append([i, j])

    # Add negative edges (Mutual exclusion for answer choices of the same query)
    for i in range(len(qa_data)):
        answer_nodes = [idx for idx, q_idx in enumerate(query_indices) if q_idx == i]
        for j in range(len(answer_nodes)):
            for k in range(j + 1, len(answer_nodes)):
                negative_edges.append([answer_nodes[j], answer_nodes[k]])  # Negative edges

    remove=True
    if remove:
        positive_edges_set = set(tuple(edge) for edge in positive_edges)
        negative_edges_set = set(tuple(edge) for edge in negative_edges)
        positive_edges = list(positive_edges_set - negative_edges_set)

    positive_edges = torch.tensor(positive_edges, dtype=torch.long).t().contiguous()
    negative_edges = torch.tensor(negative_edges, dtype=torch.long).t().contiguous()

    # Step 3: Define GNN Model with Positive and Negative Edge Influence
    class AnswerSelectionGNN(torch.nn.Module):
        def __init__(self, input_dim, hidden_dim, num_classes):
            super(AnswerSelectionGNN, self).__init__()
            self.gat1 = GATConv(input_dim, hidden_dim, heads=2)
            self.gat2 = GATConv(hidden_dim * 2, num_classes, heads=1)

        def forward(self, x, pos_edge_index, neg_edge_index):
            # Standard GAT propagation for positive edges
            x = self.gat1(x, pos_edge_index)
            x = F.relu(x)
            x = self.gat2(x, pos_edge_index)

            # Negative edge penalty
            # if neg_edge_index.numel() > 0:  # Ensure negative edges exist
            #     neg_sim = (x[neg_edge_index[0]] * x[neg_edge_index[1]]).sum(dim=1)
            #     neg_penalty = torch.tanh(neg_sim)  # Suppress similarity
            #     x[neg_edge_index[0]] -= neg_penalty.unsqueeze(1)

            return x


    # Step 4: Train the GNN Model for Label Propagation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    GNNmodel = AnswerSelectionGNN(input_dim=query_choice_embeddings.shape[1], hidden_dim=64, num_classes=2).to(device)



    query_choice_embeddings = query_choice_embeddings.to(device)
    positive_edges = positive_edges.to(device)
    negative_edges = negative_edges.to(device)
    labels = labels.to(device)
    labeled_mask = labeled_mask.to(device)

    optimizer = torch.optim.Adam(GNNmodel.parameters(), lr=0.005, weight_decay=5e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    lambda_neg = 0.4  # Strength of negative edge loss
    print("Begin Training")
    for epoch in range(25):
        GNNmodel.train()
        optimizer.zero_grad()

        out = GNNmodel(query_choice_embeddings, positive_edges, negative_edges)
        # print(out.shape)
        # # Apply softmax across the four choices per query
        # logits = out.view(len(qa_data), 4, -1)
        # print(logits.shape)
        # probs = F.softmax(logits, dim=1)
        # print(probs.shape)
        # print(probs.view(-1, 2).shape)
        #
        # # Compute loss only for labeled nodes
        # loss = loss_fn(probs.view(-1, 2)[labeled_mask], labels[labeled_mask])


        probs = F.softmax(out)

        # Compute loss only for labeled nodes
        loss = loss_fn(probs[labeled_mask], labels[labeled_mask])

        # Compute negative edge loss
        neg_edge_nodes = negative_edges.t()
        neg_similarity = (out[neg_edge_nodes[0]] * out[neg_edge_nodes[1]]).sum(dim=1)
        neg_loss = torch.mean(neg_similarity)

        # Final loss
        final_loss = loss + lambda_neg * neg_loss
        final_loss.backward()
        optimizer.step()

        if epoch % 1 == 0:
            print(f"Epoch {epoch}: Loss = {final_loss.item()}")

    # Step 5: Predict Labels for Unlabeled Queries
    GNNmodel.eval()
    output = GNNmodel(query_choice_embeddings, positive_edges, negative_edges)
    print(output.shape,query_choice_embeddings.shape)
    predicted_probs = torch.softmax(output.cpu(), dim=1).detach().numpy()
    predicted_scores = predicted_probs[:, 1]
    if dataset_name in ['social_i_qa','financial_phrasebank']:
        predicted_scores = predicted_scores.reshape(len(qa_data), 3)
    else:
        predicted_scores = predicted_scores.reshape(len(qa_data), 4)
    best_choice_indices = np.argmax(predicted_scores, axis=1)
    if dataset_name=='social_i_qa':
        choice_mapping = np.array(["A", "B", "C"])
    elif dataset_name=='financial_phrasebank':
        choice_mapping = np.array(["neutral", 'positive', 'negative'])
    else:
        choice_mapping = np.array(["A", "B", "C", "D"])
    query_predictions = choice_mapping[best_choice_indices]

    for i, data in enumerate(qa_data):

        if i in labeled_idxs:
            data['pseudo_label']=data['answer']
        else:
            data['pseudo_label'] = query_predictions[i]


    model_name = args.m
    acc=predict_result(dataset_name, qa_data, model, model_name, device=f'cuda:{args.d}', k=arg.k)
    result[dataset_name] = acc

print(result)
# Print results

