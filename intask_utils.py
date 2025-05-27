import torch
import torch.nn.functional as F
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


def predict_result(dataset_name, qa_data, model, model_name, device='cuda:0', k=1, method='llm'):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from utils.data import get_instruction, load_dataset, Prompt
    from llama import make_prompt, call_model
    if 'gpt' in model_name:
        from openai import OpenAI
        llm = OpenAI(
            api_key='YOUR_API_KEY')

        target_data = load_dataset('target', dataset_name, 'test')
        target_id = target_data['prompt_temp_id']
        target_dataset = target_data['data']
        task_inst = get_instruction(dataset_name)
        prompter = Prompt()

        prompt = query_temps_dic[target_id]

        query_embeddings = [model.encode(prompt.format(data['question'])) for data in qa_data]
        example_query_embeddings = torch.tensor(np.array(query_embeddings), dtype=torch.float)

        example_prompts = [(prompt.format(d['question']) + " " + str(d['pseudo_label'])) for d
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
            # break

        eval_dic = dict()
        eval_dic['id'] = ids
        # eval_dic['prompt'] = prompts
        eval_dic['pred'] = preds
        eval_dic['true_label'] = gold_labels

        df = pd.DataFrame(eval_dic)
        df = df.fillna('')
        result_path = f"/results/{dataset_name}_intask_{method}_k{k}.csv"
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

        example_prompts = [(prompt.format(d['question']) + " " + str(d['pseudo_label'])) for d
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

# Sample dataset with labeled and unlabeled queries
def create_new_dataset(data_list, N=500, seed=42):
    selected_samples = []
    np.random.seed(seed)

    if type(data_list[0]['answer']) == list:
        labels = list(set([example['answer'][0] for example in data_list]))
        if len(data_list)<N:
            return data_list, labels
        count = {}
        for label in labels:
            count[label] = 0


        chosen_indices = np.random.randint(low=0, high=len(data_list), size=len(data_list))

        for chosen in chosen_indices:
            if (chosen not in selected_samples) and (count[data_list[chosen]['answer'][0]] < (N / (len(count)))):
                selected_samples.append(chosen)
                count[data_list[chosen]['answer'][0]] += 1

        selected_samples.sort()

        selected_data = list(np.array(data_list)[selected_samples])[:N]
        return selected_data, labels

    else:
        labels = list(set([example['answer'] for example in data_list]))
        if len(data_list)<N:
            return data_list, labels
        count = {}
        for label in labels:
            count[label] = 0

        chosen_indices = np.random.randint(low=0, high=len(data_list), size=len(data_list))

        for chosen in chosen_indices:

            if (chosen not in selected_samples) and (count[data_list[chosen]['answer']] < (N / (len(count)))):
                selected_samples.append(chosen)
                count[data_list[chosen]['answer']] += 1

        selected_samples.sort()
        selected_data = list(np.array(data_list)[selected_samples])[:N]
        return selected_data, labels

def pack_data(dataset_name,seed=42):


    if dataset_name == 'medmcqa':
        dataset = load_dataset(dataset_name)['train']

        train_idxs = []
        train_queries = []
        train_choices = []
        train_questions = []
        train_labels = []

        text2label = {
            0: 'A',
            1: 'B',
            2: 'C',
            3: 'D'
        }

        for d in dataset:
            train_idxs.append(d['id'])
            train_labels.append(text2label[d['cop']])
            train_choices.append({"A":d['opa'], "B":d['opb'], "C":d['opc'], "D":d['opd']})
            train_questions.append(d['question']+' \nA. '+d['opa']+' \nB. '+d['opb']+' \nC. '+d['opc']+' \nD. '+d['opd'])
            train_queries.append(d['question'])


        df = {
            'id': train_idxs,
            "query":train_queries,
            "question": train_questions,
            "choices": train_choices,
            'answer': train_labels
        }
        df = pd.DataFrame(df)
        test_s_df = df.to_dict('records')
        selected_data, labels = create_new_dataset(test_s_df, N=500, seed=42)
        prompt_temp_id = '005'

    elif dataset_name == 'sciq':
        dataset = load_dataset(dataset_name)['train']

        train_idxs = []
        train_queries = []
        train_choices = []
        train_questions = []
        train_labels = []

        id = 1
        import random
        text2label = {
            'A': 'distractor3',
            'B': 'distractor1',
            'C': 'distractor2',
            'D': 'correct_answer'
        }

        for d in dataset:

            k = ['A', 'B', 'C', 'D']
            k_ = ['A', 'B', 'C', 'D']
            random.shuffle(k)
            q = d['question']
            choice_dict={}
            for i, l in zip(k, k_):
                op = text2label[i]
                if op == 'correct_answer':
                    train_labels.append(l)
                choice_dict[l]=d[op]
                q += f'\n{l}. ' + d[op]


            train_idxs.append(id)
            train_choices.append(choice_dict)
            train_questions.append(q)
            train_queries.append(d['question'])

        df = {
            'id': train_idxs,
            "query":train_queries,
            "question": train_questions,
            "choices": train_choices,
            'answer': train_labels
        }

        df = pd.DataFrame(df)
        test_s_df = df.to_dict('records')
        selected_data, labels = create_new_dataset(test_s_df, N=500, seed=42)
        prompt_temp_id = '005'

    elif dataset_name == 'ARC-Challenge':

        dataset = load_dataset('ai2_arc', dataset_name)['train']
        train_idxs = []
        train_queries = []
        train_choices = []
        train_questions = []
        train_labels = []

        id = 1
        import random

        label2text = {
            'A': 'A',
            'B': 'B',
            "C": "C",
            "D": "D",
            '2': 'A',
            '1': 'B',
            "3": "C",
            "4": "D",
        }

        for d in dataset:
            if len(d['choices']['text']) != 4:
                continue
            train_idxs.append(d['id'])
            id += 1
            train_questions.append(
                d['question'] + '\nA. ' + d['choices']['text'][0] + '\nB. ' + d['choices']['text'][1] + '\nC. ' +
                d['choices']['text'][2] + '\nD. ' + d['choices']['text'][3])
            train_labels.append(label2text[d['answerKey']])
            train_queries.append(d['question'])
            train_choices.append({"A":d['choices']['text'][0], "B":d['choices']['text'][1], "C":d['choices']['text'][2], "D":d['choices']['text'][3]})

        df = {
            'id': train_idxs,
            "query":train_queries,
            "question": train_questions,
            "choices": train_choices,
            'answer': train_labels
        }
        df = pd.DataFrame(df)
        test_s_df = df.to_dict('records')
        selected_data, labels = create_new_dataset(test_s_df, N=500, seed=42)
        prompt_temp_id = '005'

    elif dataset_name == 'social_i_qa':
        dataset = load_dataset(dataset_name)['train']
        train_idxs = []
        train_queries = []
        train_choices = []
        train_questions = []
        train_labels = []
        id = 1

        text2label = {
            '1': 'A',
            '2': 'B',
            '3': 'C'
        }
        label2text = {
            1: 'answerA',
            2: 'answerB',
            3: 'answerC'
        }

        for d in dataset:
            train_idxs.append(id)
            id += 1

            train_labels.append(text2label[d['label']])
            q=d['context']+ '\nQuestion:' +d['question']

            question = q + ' \nA. ' + d['answerA'] + ' \nB. ' + d['answerB'] + ' \nC. ' + d['answerC']
            train_questions.append(question)
            train_queries.append(q)
            train_choices.append({"A":d['answerA'], "B":d['answerB'], "C":d['answerC']})

        df = {
            'id': train_idxs,
            "query":train_queries,
            "question": train_questions,
            "choices": train_choices,
            'answer': train_labels
        }
        test_df = pd.DataFrame(df)
        test_s_df = test_df.to_dict('records')
        print(len(test_s_df))
        selected_data, labels = create_new_dataset(test_s_df, N=500, seed=42)
        print(len(selected_data))
        prompt_temp_id = '003'

    elif dataset_name == 'financial_phrasebank':
        dataset = load_dataset(dataset_name, 'sentences_allagree', trust_remote_code=True)['train']
        dataset = dataset.train_test_split(test_size=0.2, stratify_by_column="label")
        train_idxs = []
        train_queries = []
        train_choices = []
        train_questions = []
        train_labels = []
        id = 1

        text2label = {
            1: 'neutral',
            2: 'positive',
            0: 'negative'
        }

        for d in dataset['test']:
            train_idxs.append(id)
            id += 1
            train_questions.append(d['sentence'])
            train_queries.append(d['sentence'])
            train_labels.append(text2label[d['label']])
            train_choices.append({"neutral": 'neutral', 'positive': 'positive', 'negative': 'negative'})

        df = {
            'id': train_idxs,
            "query": train_queries,
            "question": train_questions,
            "choices": train_choices,
            'answer': train_labels
        }
        test_df = pd.DataFrame(df)
        test_s_df = test_df.to_dict('records')
        print(len(test_s_df))
        selected_data, labels = create_new_dataset(test_s_df, N=500, seed=42)
        print(len(selected_data))
        prompt_temp_id = '004'

    return selected_data, prompt_temp_id