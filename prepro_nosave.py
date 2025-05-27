import pandas as pd
import transformers
import torch, json, math
import random
import numpy as np
import os, pickle
from tqdm import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer



class Prompt:
    def __init__(self):

        self.prompt_temps_dic = {
            '001': 'Definition: {} \nPremise: {} \nHypothesis: {} \nLabel:',
            '002': 'Definition: {} \nQuestion 1: {} \nQuestion 2: {} \nLabel:',
            '003': 'Definition: {} \nContext: {} \nQuestion: {} \nAnswer:',
            '004': 'Definition: {} \nSentence: {} \nLabel:',
            '005': 'Definition: {} \nQuestion: {} \nAnswer:',
            '006': 'Definition: {} \nSentence 1: {} \nSentence 2: {} \nLabel:',
            '007': 'Definition: {} \nQuestion: {} \nSentence: {} \nLabel:',
        }

        self.prompt_temps_dic_without_def = {
            '001': 'Premise: {} \nHypothesis: {} \nLabel:',
            '002': 'Question 1: {} \nQuestion 2: {} \nLabel:',
            '003': 'Context: {} \nQuestion: {} \nAnswer:',
            '004': 'Sentence: {} \nLabel:',
            '005': 'Question: {} \nAnswer:',
            '006': 'Sentence 1: {} \nSentence 2: {} \nLabel:',
            '007': 'Question: {} \nSentence: {} \nLabel:'
        }

    def get_input_data_with_def(self, instructions, data, prompt_id):

        promt = self.prompt_temps_dic[prompt_id]
        if prompt_id in ['004', '005']:
            return promt.format(instructions, data['sentence'])
        elif prompt_id == '003':
            return promt.format(instructions, data['context'], data['sentence'])
        elif prompt_id in ['001', '002', '006', '007']:
            return promt.format(instructions, data['sentence1'], data['sentence2'])
        raise ValueError('Prompt ID not coded')

    def get_data_label(self, data, prompt_id):

        if prompt_id in ['001', '002', '003', '004', '005', '006', '007']:
            return data['label']

        raise ValueError('Prompt ID not coded')

    def get_input_data_without_def(self, data, prompt_id):

        promt = self.prompt_temps_dic_without_def[prompt_id]
        if prompt_id in ['004', '005']:
            return promt.format(data['sentence'])
        elif prompt_id == '003':
            return promt.format(data['context'], data['sentence'])
        elif prompt_id in ['001', '002', '006', '007']:
            return promt.format(data['sentence1'], data['sentence2'])
        raise ValueError('Prompt ID not coded')
def create_embedding(data, model):
    o = model.encode(data, convert_to_tensor=True).detach().cpu()
    return o


def create_embeddings(dataset, prompter, id, model, print_example=False, batch_size=1024):
    num_batch = math.ceil(len(dataset) / batch_size)

    for batch_i in tqdm(range(num_batch), desc='Getting emb of batches'):

        batch_dataset = dataset[batch_i * batch_size:(batch_i + 1) * batch_size]
        batch_inputs = [prompter.get_input_data_without_def(d, id) for d in batch_dataset]
        emb = create_embedding(batch_inputs, model)
        for i, d in enumerate(batch_dataset):
            d['emb'] = emb[i]
            if print_example:
                print(d)
                print_example = False


def create_new_dataset(data_list, k=500):
    selected_samples = []

    if type(data_list[0]['label']) == list:
        labels = list(set([example['label'][0] for example in data_list]))

        print(f"Label Space: {labels}")
        count = {}
        for label in labels:
            count[label] = 0

        np.random.seed(0)
        chosen_indices = np.random.randint(low=0, high=len(data_list), size=len(data_list))

        for chosen in chosen_indices:
            if (chosen not in selected_samples) and (count[data_list[chosen]['label'][0]] < (k / (len(count)))):
                selected_samples.append(chosen)
                count[data_list[chosen]['label'][0]] += 1

        selected_samples.sort()

        print(f"Count of selected labels: {count}")

        selected_data = list(np.array(data_list)[selected_samples])
        return selected_data, labels

    else:
        labels = list(set([example['label'] for example in data_list]))

        print(f"Label Space: {labels}")
        count = {}
        for label in labels:
            count[label] = 0

        np.random.seed(0)
        chosen_indices = np.random.randint(low=0, high=len(data_list), size=len(data_list))

        for chosen in chosen_indices:

            if (chosen not in selected_samples) and (count[data_list[chosen]['label']] < (k / (len(count)))):
                selected_samples.append(chosen)
                count[data_list[chosen]['label']] += 1

        selected_samples.sort()
        print(f"Count of selected labels: {count}")
        selected_data = list(np.array(data_list)[selected_samples])
        return selected_data, labels

path='data/source'
if not os.path.exists(path):
    os.makedirs(path)
path='data/target'
if not os.path.exists(path):
    os.makedirs(path)
path='data/target-unlabeled'
if not os.path.exists(path):
    os.makedirs(path)
prompter=Prompt()
instruct={'mnli':'Given Sentence 1 which is a premise and Sentence 2 which is a hypothesis do natural language inference on the pair. In natural language inference we mark whether the premise and hypothesis are "neutral", "contradiction" or "entailment". The pair are said to be "entailed" if the premise justifies/supports the hypothesis, if the pair contradict each other we label them as "contradiction" and label them "neutral" in all other cases.',
        'qqp': 'Given two question pairs do text classification based on whether they are duplicates or not. The questions are mined from the popular online discussion forum Quora. As duplicate quetion might be present on Quora, the task is to label two identical questions as "duplicate" if they ask the same query else label the pair as "not duplicate".',
        'boolq': 'Given a context and a question do binary true and false type text classification. You are given a passage as context and a question related to the passage that can be answered as "True" or "False". Based on the context, question and your reasoning ability answer in a "True" and "False".',
        'conll2003_ner': 'Given a sentence do token classification on it seek to locate and classify named entities mentioned in the sentence provided. The pre-defined named entity categories along with there labeles are Person (PER), Location (LOC), Organization (ORG) and Miscellaneous (MIS). If the token is not an entity mark it as None. As the entity is more than two tokens long use the prefix B with the named entity token to represent the beginning and  use the prefix I till the entity ends.',
        'conll2003_pos': 'Given a sentence do token classification by doing Part-of-speech (POS) tagging, which is a process in natural language processing (NLP) where each word in a text is labeled with its corresponding part of speech. This can include nouns, verbs, adjectives, and other grammatical categories.',
        'commonsense_qa': 'The following task relates to commonsense reasoning. It consists of a question that can be easily solved using logical abilities and reasoning, a set of five options  "A.", "B.", "C.", "D." and "E." are also provided along with the question, one of these options answers the question logically. Use your reasoning ability to select the most appropriate answer from the provided choices "A.", "B.", "C.", "D." and "E." and assign these choices (i.e  "A.", "B.", "C.", "D." and "E.") as the label',
        'ARC-Easy': 'Given a question answering task from the 3rd to 9th-grade science exam. The question contains four options "A.", "B.", "C." and "D." Select the most appropriate choice that answers the question',
        'race': 'Given a reading comprehension type question-answering from an english exam for school students. You are given a context and multiple choice question containing four options "A.", "B.", "C." and "D.". The question is answerable from the comprehension. Based on the question, the option and the context select the most appropriate answer from the provided choices "A.", "B.", "C." and "D.".',
        'ag_news': 'Given a sentence do text classification, the sentence is a clipping from a news article that may be either related to sports, business, technology, or world news. You are to recognize the category of the sentence and label them as "sports", "business", "technology" or "world" news',
        'sst2': 'Given a movie review do text classification, based on the sentiment conveyed by the review label it as "positive" or "negative"',
        'medmcqa': 'Given a multiple choice question containing four options "A.", "B.", "C." and "D." from a medical entrance exam. The question is related to a sub-field of medical science like Microbiology, Radiology, Ophthalmology, Surgery, Human anatomy, etc. Based on the question, the option and your knowledge of the medical field select the most appropriate answer from the provided choices "A.", "B.", "C." and "D.".',
        'sciq': 'Given a question from a scientific exam about Physics, Chemistry, and Biology, among others. The question is in multiple choice format with four answer options "A.", "B.", "C." and "D.". Using your knowledge about the scientific fields answer the question and provide the label "A", "B", "C" and "D" as answer',
        'ARC-Challenge': 'Given a question answering task from the 3rd to 9th-grade science exam. The question contains four options "A.", "B.", "C." and "D." Select the most appropriate choice that answers the question',
        'social_i_qa': 'Given an action as the context and a related question, you are to answer the question based on the context using your social intelligence. The question is of multiple choice form with three options "A", "B" and "C". Select the most appropriate answer from the provided choices "A", "B" and "C".',
        'financial_phrasebank': 'Given a sentence mined from a financial news article, you are to determine the sentiment polarity of the sentence. The task deals with financial sentiment analysis. Based on the sentiment conveyed by the sentence, label the sentence as "negative", "positive" or "neutral"',
          }

def get_dataset(dataset_name,split='data/source',k=0,emb_model="all-MiniLM-L6-v2"):
    model = SentenceTransformer(emb_model)
    if k==0:
        if "unlabeled" in split:
            k=8
        elif "target" in split:
            k=500
        else:
            k = 10000
    if dataset_name == 'mnli':
        dataset = load_dataset('glue', dataset_name)['train']
        test_id = []
        test_s1 = []
        test_s2 = []
        test_l = []
        import random
        label2text = {
            2: 'contradiction',
            1: 'neutral',
            0: 'entailment'
        }

        for d in dataset:
            test_id.append(d['idx'])
            test_s1.append(d['premise'])
            test_s2.append(d['hypothesis'])
            test_l.append(label2text[d['label']])

        df = {
            'id': test_id,
            'label': test_l,
            'sentence1': test_s1,
            'sentence2': test_s2
        }
        test_df = pd.DataFrame(df)
        test_df = test_df.sample(k, random_state=0)
        test_s_df = test_df.to_dict('records')
        prompt_temp_id = '001'
        create_embeddings(test_s_df, prompter, prompt_temp_id, model)
        set_type = 'source'

    elif dataset_name == 'qqp':
        dataset = load_dataset('glue', dataset_name)['train']
        test_id = []
        test_s1 = []
        test_s2 = []
        test_l = []
        label2text = {
            1: 'duplicate',
            0: 'not duplicate'
        }

        for d in dataset:
            test_id.append(d['idx'])
            test_s1.append(d['question1'])
            test_s2.append(d['question2'])
            test_l.append(label2text[d['label']])

        df = {
            'id': test_id,
            'label': test_l,
            'sentence1': test_s1,
            'sentence2': test_s2
        }
        test_df = pd.DataFrame(df)
        test_df = test_df.sample(k, random_state=0)
        test_s_df = test_df.to_dict('records')
        prompt_temp_id = '002'
        create_embeddings(test_s_df, prompter, prompt_temp_id, model)
        set_type = 'source'

    elif dataset_name == 'boolq':
        dataset = load_dataset(dataset_name)['train']
        test_id = []
        test_s = []
        test_c = []
        test_l = []
        id = 1
        for d in dataset:
            test_id.append(id)
            id += 1
            test_s.append(d['question'])
            test_c.append(d['passage'])
            test_l.append(str(d['answer']))

        df = {
            'id': test_id,
            'label': test_l,
            'sentence': test_s,
            'context': test_c
        }
        test_df = pd.DataFrame(df)
        test_s_df = test_df.to_dict('records')
        prompt_temp_id = '003'
        create_embeddings(test_s_df, prompter, prompt_temp_id, model)
        set_type = 'source'

    elif dataset_name=='conll2003_ner':
        dname = 'conll2003'
        dataset= load_dataset(dname)['train']
        test_id=[]
        test_s=[]
        test_l=[]
        id=1

        label2text={0:'O',
        1:'B-PER',
        2:'I-PER',
        3:'B-ORG',
        4:'I-ORG',
        5:'B-LOC',
        6:'I-LOC',
        7:'B-MISC',
        8:'I-MISC'
        }

        for d in dataset:
            l=[]
            test_id.append(d['id'])
            for i in d['ner_tags']:
                l.append(label2text[i])
            test_s.append(' '.join(d['tokens']))
            test_l.append(' '.join(l))

        df={
            'id':test_id,
            'label':test_l,
            'sentence':test_s
        }
        test_df=pd.DataFrame(df)
        test_df=test_df.sample(k,random_state=0)
        test_s_df=test_df.to_dict('records')
        prompt_temp_id = '004'
        create_embeddings(test_s_df, prompter, prompt_temp_id, model)
        set_type = 'source'

    elif dataset_name=='conll2003_pos':
        dname = 'conll2003'
        dataset= load_dataset(dname)['train']
        test_id=[]
        test_s=[]
        test_l=[]
        id=1

        label = {'"': 0, "''": 1, '#': 2, '$': 3, '(': 4, ')': 5, ',': 6, '.': 7, ':': 8, '``': 9, 'CC': 10, 'CD': 11,
                 'DT': 12,
                 'EX': 13, 'FW': 14, 'IN': 15, 'JJ': 16, 'JJR': 17, 'JJS': 18, 'LS': 19, 'MD': 20, 'NN': 21, 'NNP': 22,
                 'NNPS': 23,
                 'NNS': 24, 'NN|SYM': 25, 'PDT': 26, 'POS': 27, 'PRP': 28, 'PRP$': 29, 'RB': 30, 'RBR': 31, 'RBS': 32,
                 'RP': 33,
                 'SYM': 34, 'TO': 35, 'UH': 36, 'VB': 37, 'VBD': 38, 'VBG': 39, 'VBN': 40, 'VBP': 41, 'VBZ': 42,
                 'WDT': 43,
                 'WP': 44, 'WP$': 45, 'WRB': 46}

        label2text = dict([(v, k) for k, v in label.items()])

        for d in dataset:

            l = []
            test_id.append(d['id'])
            for i in d['pos_tags']:
                l.append(label2text[i])
            test_s.append(' '.join(d['tokens']))
            test_l.append(' '.join(l))

        df = {
            'id': test_id,
            'label': test_l,
            'sentence': test_s
        }
        test_df = pd.DataFrame(df)
        test_df = test_df.sample(k, random_state=0)
        test_s_df = test_df.to_dict('records')
        prompt_temp_id = '004'
        create_embeddings(test_s_df, prompter, prompt_temp_id, model)
        set_type = 'source'

    elif dataset_name=='commonsense_qa':
        dataset= load_dataset('commonsense_qa')['validation']
        test_id=[]
        test_s=[]
        test_l=[]

        text2label={
            'A':0,
            'B':1,
            'C':2,
            'D':3,
            'E':4
        }

        label2text={
            0:'\nA. ',
            1:'\nB. ',
            2:'\nC. ',
            3:'\nD. ',
            4:'\nE. '
        }

        for d in dataset:
            test_id.append(d['id'])
            test_l.append(d['answerKey'])

            q=d['question']
            for i,a in enumerate(d['choices']['text']):
                q+=' '+label2text[i]+a

            test_s.append(q)

        df={
            'id':test_id,
            'label':test_l,
            'sentence':test_s
        }

        test_df=pd.DataFrame(df)
        test_s_df=test_df.to_dict('records')
        prompt_temp_id = '005'
        create_embeddings(test_s_df, prompter, prompt_temp_id, model)
        set_type = 'source'

    elif dataset_name=='ARC-Easy':
        dataset= load_dataset('ai2_arc',dataset_name)['train']
        test_id=[]
        test_s=[]
        test_l=[]
        id=1

        label2text={
            'A':'A',
            'B':'B',
            "C":"C",
            "D":"D",
            '1':'A',
            '2':'B',
            "3":"C",
            "4":"D",
        }

        for d in dataset:
            if len(d['choices']['text'])!=4:
                continue
            test_id.append(d['id'])
            id+=1
            test_s.append(d['question']+'\nA. '+d['choices']['text'][0]+'\nB. '+d['choices']['text'][1]+'\nC. '+d['choices']['text'][2]+'\nD. '+d['choices']['text'][3])
            test_l.append(label2text[d['answerKey']])

        df={
            'id':test_id,
            'label':test_l,
            'sentence':test_s
        }
        test_df=pd.DataFrame(df)
        test_s_df=test_df.to_dict('records')
        prompt_temp_id = '005'
        create_embeddings(test_s_df, prompter, prompt_temp_id, model)
        set_type = 'source'

    elif dataset_name=='race':
        dataset= load_dataset('race','all')['train']
        test_id=[]
        test_s=[]
        test_c=[]
        test_l=[]
        for d in dataset:

            q=d['question']+' \nA. '+d['options'][0]+' \nB. '+d['options'][1]+' \nC. '+d['options'][2]+' \nD. '+d['options'][3]

            test_id.append(d['example_id'])
            test_s.append(q)
            test_c.append(d['article'])
            test_l.append(d['answer'])

        df={
            'id':test_id,
            'label':test_l,
            'sentence':test_s,
            'context':test_c
        }
        test_df=pd.DataFrame(df)
        test_df=test_df.sample(k,random_state=0)
        test_s_df=test_df.to_dict('records')
        prompt_temp_id = '003'
        create_embeddings(test_s_df, prompter, prompt_temp_id, model)
        set_type = 'source'

    elif dataset_name == 'ag_news':

        dataset = load_dataset(dataset_name)['train']
        test_id = []
        test_s = []
        test_l = []
        id = 1

        label2text = {
            0: 'world',
            1: 'sports',
            2: 'business',
            3: 'technology'
        }

        for d in dataset:
            test_id.append(id)
            id += 1
            test_s.append(d['text'])
            test_l.append(label2text[d['label']])

        df = {
            'id': test_id,
            'label': test_l,
            'sentence': test_s
        }
        test_df = pd.DataFrame(df)
        test_df = test_df.sample(k, random_state=0)
        test_s_df = test_df.to_dict('records')
        prompt_temp_id = '004'
        create_embeddings(test_s_df, prompter, prompt_temp_id, model)
        set_type = 'source'

    elif dataset_name == 'sst2':

        dataset = load_dataset('glue', 'sst2')['train']
        test_id = []
        test_s = []
        test_l = []
        label2text = {
            1: 'positive',
            0: 'negative'
        }

        for d in dataset:
            test_id.append(d['idx'])
            test_s.append(d['sentence'])
            test_l.append(label2text[d['label']])

        df = {
            'id': test_id,
            'label': test_l,
            'sentence': test_s
        }
        test_df = pd.DataFrame(df)
        test_df = test_df.sample(k, random_state=0)
        test_s_df = test_df.to_dict('records')
        set_type = 'source'
        prompt_temp_id = '004'
        create_embeddings(test_s_df, prompter, prompt_temp_id, model)

    elif dataset_name == 'medmcqa':

        if "unlabeled" in split:
            set_type = 'unlabeled'
            dataset = load_dataset(dataset_name)['train']
        else:
            set_type = 'target'
            dataset = load_dataset(dataset_name)['validation']


        test_id = []
        test_s = []
        test_l = []

        text2label = {
            0: 'A',
            1: 'B',
            2: 'C',
            3: 'D'
        }

        for d in dataset:
            test_id.append(d['id'])
            test_l.append(text2label[d['cop']])

            q = d['question'] + ' \nA. ' + d['opa'] + ' \nB. ' + d['opb'] + ' \nC. ' + d['opc'] + ' \nD. ' + d['opd']

            test_s.append(q)

        df = {
            'id': test_id,
            'label': test_l,
            'sentence': test_s
        }
        test_df = pd.DataFrame(df)
        test_s_df = test_df.to_dict('records')
        selected_data, labels = create_new_dataset(test_s_df, k=k)
        prompt_temp_id = '005'
        create_embeddings(selected_data, prompter, prompt_temp_id, model)

    elif dataset_name == 'sciq':
        if "unlabeled" in split:
            set_type = 'unlabeled'
            dataset = load_dataset(dataset_name)['train']
        else:
            set_type = 'target'
            dataset = load_dataset(dataset_name)['test']


        test_id = []
        test_s = []
        test_l = []
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
            for i, l in zip(k, k_):
                op = text2label[i]
                if op == 'correct_answer':
                    test_l.append(l)

                q += f'\n{l}. ' + d[op]

            test_id.append(id)
            id += 1
            test_s.append(q)

        df = {
            'id': test_id,
            'label': test_l,
            'sentence': test_s
        }
        test_df = pd.DataFrame(df)
        test_s_df = test_df.to_dict('records')
        selected_data, labels = create_new_dataset(test_s_df)
        prompt_temp_id = '005'
        create_embeddings(selected_data, prompter, prompt_temp_id, model)

    elif dataset_name == 'ARC-Challenge':
        if "unlabeled" in split:
            set_type = 'unlabeled'
            dataset = load_dataset('ai2_arc',dataset_name)['train']
        else:
            set_type = 'target'
            dataset = load_dataset('ai2_arc', dataset_name)['test']


        test_id = []
        test_s = []
        test_l = []
        id = 1
        import random

        label2text = {
            'A': 'A',
            'B': 'B',
            "C": "C",
            "D": "D",
            '1': 'A',
            '2': 'B',
            "3": "C",
            "4": "D",
        }

        for d in dataset:
            if len(d['choices']['text']) != 4:
                continue
            test_id.append(d['id'])
            id += 1
            test_s.append(
                d['question'] + '\nA. ' + d['choices']['text'][0] + '\nB. ' + d['choices']['text'][1] + '\nC. ' +
                d['choices']['text'][2] + '\nD. ' + d['choices']['text'][3])
            test_l.append(label2text[d['answerKey']])

        df = {
            'id': test_id,
            'label': test_l,
            'sentence': test_s
        }
        test_df = pd.DataFrame(df)
        test_s_df = test_df.to_dict('records')
        selected_data, labels = create_new_dataset(test_s_df)
        prompt_temp_id = '005'
        create_embeddings(test_s_df, prompter, prompt_temp_id, model)

    elif dataset_name == 'social_i_qa':
        if "unlabeled" in split:
            set_type = 'unlabeled'
            dataset = load_dataset(dataset_name, trust_remote_code=True)['train']
        else:
            set_type = 'target'
            dataset = load_dataset(dataset_name, trust_remote_code=True)['validation']


        test_id = []
        test_s = []
        test_l = []
        test_c = []
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
            test_id.append(id)
            id += 1

            test_l.append(text2label[d['label']])
            test_c.append(d['context'])

            q = d['question'] + ' \nA. ' + d['answerA'] + ' \nB. ' + d['answerB'] + ' \nC. ' + d['answerC']
            test_s.append(q)

        df = {
            'id': test_id,
            'label': test_l,
            'sentence': test_s,
            'context': test_c
        }
        test_df = pd.DataFrame(df)
        test_s_df = test_df.to_dict('records')
        selected_data, labels = create_new_dataset(test_s_df)

        prompt_temp_id = '003'
        create_embeddings(test_s_df, prompter, prompt_temp_id, model)

    elif dataset_name == 'financial_phrasebank':
        dataset = load_dataset(dataset_name, 'sentences_allagree', trust_remote_code=True)['train']
        #dataset = dataset.train_test_split(test_size=0.2, stratify_by_column="label")
        if "unlabeled" in split:
            set_type = 'unlabeled'
            #dataset=dataset['test']
        else:
            set_type = 'target'
            #dataset = dataset['train']

        test_id = []
        test_s = []
        test_l = []
        id = 1

        text2label = {
            1: 'neutral',
            2: 'positive',
            0: 'negative'
        }

        for d in dataset:
            test_id.append(id)
            id += 1
            test_s.append(d['sentence'])
            test_l.append(text2label[d['label']])

        df = {
            'id': test_id,
            'label': test_l,
            'sentence': test_s
        }
        test_df = pd.DataFrame(df)
        test_s_df = test_df.to_dict('records')
        selected_data, labels = create_new_dataset(test_s_df)

        prompt_temp_id = '004'
        create_embeddings(test_s_df, prompter, prompt_temp_id, model)

    if set_type == 'source':
        data_to_save = {
            'prompt_temp_id': prompt_temp_id,
            'data': test_s_df,
            'set': set_type
        }

    elif set_type == 'target':
        data_to_save = {
            'prompt_temp_id': prompt_temp_id,
            'data': selected_data,
            'labels': labels,
            'set': set_type
        }

    elif set_type == 'unlabeled':
        data_to_save = {
            'prompt_temp_id': prompt_temp_id,
            'data': selected_data,
            'labels': labels,
            'set': set_type
        }
    return data_to_save


def get_target_trainset(dataset_name,split='data/target',k=10000,emb_model='all-MiniLM-L6-v2'):
    model = SentenceTransformer(emb_model)

    if dataset_name == 'medmcqa':
        set_type = 'source'
        dataset = load_dataset(dataset_name)['train']

        test_id = []
        test_s = []
        test_l = []

        text2label = {
            0: 'A',
            1: 'B',
            2: 'C',
            3: 'D'
        }

        for d in dataset:
            test_id.append(d['id'])
            test_l.append(text2label[d['cop']])

            q = d['question'] + ' \nA. ' + d['opa'] + ' \nB. ' + d['opb'] + ' \nC. ' + d['opc'] + ' \nD. ' + d['opd']

            test_s.append(q)

        df = {
            'id': test_id,
            'label': test_l,
            'sentence': test_s
        }
        test_df = pd.DataFrame(df)
        if len(test_df)>k:
            test_df = test_df.sample(k, random_state=0)
        test_s_df = test_df.to_dict('records')
        prompt_temp_id = '005'
        create_embeddings(test_s_df, prompter, prompt_temp_id, model)

    elif dataset_name == 'sciq':
        set_type = 'source'
        dataset = load_dataset(dataset_name)['train']
        num_k=k
        test_id = []
        test_s = []
        test_l = []
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
            for i, l in zip(k, k_):
                op = text2label[i]
                if op == 'correct_answer':
                    test_l.append(l)

                q += f'\n{l}. ' + d[op]

            test_id.append(id)
            id += 1
            test_s.append(q)

        df = {
            'id': test_id,
            'label': test_l,
            'sentence': test_s
        }
        test_df = pd.DataFrame(df)
        if len(test_df)>num_k:
            test_df = test_df.sample(num_k, random_state=0)
        test_s_df = test_df.to_dict('records')
        prompt_temp_id = '005'
        create_embeddings(test_s_df, prompter, prompt_temp_id, model)

    elif dataset_name == 'ARC-Challenge':
        set_type = 'source'
        dataset = load_dataset('ai2_arc',dataset_name)['train']

        test_id = []
        test_s = []
        test_l = []
        id = 1
        import random

        label2text = {
            'A': 'A',
            'B': 'B',
            "C": "C",
            "D": "D",
            '1': 'A',
            '2': 'B',
            "3": "C",
            "4": "D",
        }

        for d in dataset:
            if len(d['choices']['text']) != 4:
                continue
            test_id.append(d['id'])
            id += 1
            test_s.append(
                d['question'] + '\nA. ' + d['choices']['text'][0] + '\nB. ' + d['choices']['text'][1] + '\nC. ' +
                d['choices']['text'][2] + '\nD. ' + d['choices']['text'][3])
            test_l.append(label2text[d['answerKey']])

        df = {
            'id': test_id,
            'label': test_l,
            'sentence': test_s
        }
        test_df = pd.DataFrame(df)
        if len(test_df)>k:
            test_df = test_df.sample(k, random_state=0)
        test_s_df = test_df.to_dict('records')
        prompt_temp_id = '005'
        create_embeddings(test_s_df, prompter, prompt_temp_id, model)

    elif dataset_name == 'social_i_qa':
        set_type = 'source'
        dataset = load_dataset(dataset_name, trust_remote_code=True)['train']


        test_id = []
        test_s = []
        test_l = []
        test_c = []
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
            test_id.append(id)
            id += 1

            test_l.append(text2label[d['label']])
            test_c.append(d['context'])

            q = d['question'] + ' \nA. ' + d['answerA'] + ' \nB. ' + d['answerB'] + ' \nC. ' + d['answerC']
            test_s.append(q)

        df = {
            'id': test_id,
            'label': test_l,
            'sentence': test_s,
            'context': test_c
        }
        test_df = pd.DataFrame(df)
        if len(test_df)>k:
            test_df = test_df.sample(k, random_state=0)
        test_s_df = test_df.to_dict('records')
        prompt_temp_id = '003'
        create_embeddings(test_s_df, prompter, prompt_temp_id, model)

    elif dataset_name == 'financial_phrasebank':
        dataset = load_dataset(dataset_name, 'sentences_allagree', trust_remote_code=True)['train']
        #dataset = dataset.train_test_split(test_size=0.2, stratify_by_column="label")
        set_type = 'source'
        #dataset=dataset['test']

        test_id = []
        test_s = []
        test_l = []
        id = 1

        text2label = {
            1: 'neutral',
            2: 'positive',
            0: 'negative'
        }

        for d in dataset:
            test_id.append(id)
            id += 1
            test_s.append(d['sentence'])
            test_l.append(text2label[d['label']])

        df = {
            'id': test_id,
            'label': test_l,
            'sentence': test_s
        }
        test_df = pd.DataFrame(df)
        if len(test_df)>k:
            test_df = test_df.sample(k, random_state=0)
        test_s_df = test_df.to_dict('records')
        prompt_temp_id = '004'
        create_embeddings(test_s_df, prompter, prompt_temp_id, model)

    if set_type == 'source':
        data_to_save = {
            'prompt_temp_id': prompt_temp_id,
            'data': test_s_df,
            'set': set_type
        }
        return data_to_save

