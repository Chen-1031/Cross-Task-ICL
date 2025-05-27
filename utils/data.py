import json,os
import pickle


def get_instruction(dataset_name):
    instruct = {
        'mnli':'Given Sentence 1 which is a premise and Sentence 2 which is a hypothesis do natural language inference on the pair. In natural language inference we mark whether the premise and hypothesis are "neutral", "contradiction" or "entailment". The pair are said to be "entailed" if the premise justifies/supports the hypothesis, if the pair contradict each other we label them as "contradiction" and label them "neutral" in all other cases.',
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
    return instruct[dataset_name]


def load_dataset(benchmark, dataset_name, set):
    from prepro_nosave import get_dataset
    folder_path=f'data/{benchmark}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    test_set = get_dataset(dataset_name, split=folder_path)
    return test_set

def load_target_train(dataset_name):
    from prepro_nosave import get_target_trainset
    test_set = get_target_trainset(dataset_name)
    return test_set


def load_prompts(dataset_name, path_d='target-prompts', k=4):
    path = f'data/{path_d}/{dataset_name}_prompts_k={k}.pkl'
    with open(path, 'rb') as f:
        test_set = pickle.load(f)
    return test_set


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


class Tasks():
    source_tasks = ['ag_news', 'ARC-Easy', 'boolq', 'commonsense_qa', 'conll2003_pos', 'conll2003_ner', 'mnli', 'qqp',
                    'race', 'sst2']
    target_tasks = ['ARC-Challenge', 'comve_t1', 'comve_t2', 'financial_phrasebank', 'medmcqa', 'sciq', 'social_i_qa',
                    'medical-abstracts-tc', 'scicite']

    target_demos = {
        'ARC-Challenge': 'Question: Which of these do scientists offer as the most recent explanation as to why many plants and animals died out at the end of the Mesozoic era?\nA. "worldwide disease \nB. global mountain building \nC. rise of mammals that preyed upon plants and animals \nD. impact of an asteroid created dust that blocked the sunlight \n Answer: D \n',
        'boolq': 'Context: Newcastle upon Tyne (locally /njuːˈkæsəl/ ( listen)), commonly known as Newcastle, is a city in Tyne and Wear, North East England, 103 miles (166 km) south of Edinburgh and 277 miles (446 km) north of London on the northern bank of the River Tyne, 8.5 mi (13.7 km) from the North Sea. Newcastle is the most populous city in the North East, and forms the core of the Tyneside conurbation, the eighth most populous urban area in the United Kingdom. Newcastle is a member of the English Core Cities Group and is a member of the Eurocities network of European cities. Question: is newcastle upon tyne the same as newcastle"? \n Answer: true \n',
        'medmcqa': 'Question: Growth hormone has its effect on growth through?\nA. Directly \nB. IG1-1 \nC. Tyroxine \nD. Intranuclear receptors \n Answer: B \n',
        'sciq': 'Question: What type of organism is commonly used in preparation of foods such as cheese and yogurt?\nA. Viruses \nB. Protozoa \nC. Gymnosperms \nD. Mesophilic organisms \n Answer: D \n',
        'social_i_qa': 'Context: Cameron decided to have a barbecue and gathered her friends together.\n Question: How would Others feel as a result?\nA. Like attending \nB. Like staying home \nC. A good friend to have \n Answer: A \n',
        'financial_phrasebank': 'Sentence: For the last quarter of 2010 , Componenta \'s net sales doubled to EUR131m from EUR76m for the same period a year earlier , while it moved to a zero pre-tax profit from a pre-tax loss of EUR7m .\n Label: positive \n',
    }

    aligner = "Aligner: The previous task relates to {} and had to be labeled {}. Use your reasoning ability to learn from the knowledge learned from the previous task to solve the current one which is {} and has labels {}. The new task is:-\n"
    source_info = {
        'ag_news': ['text classification', 'sports, business, technology, or world news'],
        'ARC-Easy': ['multiple choice question answering', 'one of the provided options'],
        'race': ['read comprehension type multiple choice question answering', 'one of the provided options'],
        'commonsense_qa': ['multiple choice question answering in common-sense reasoning',
                           'one of the provided options'],
        'boolq': ['question answering', 'true or false'],
        'conll2003_pos': ['sequence classification', 'into part-of-speech tags'],
        'conll2003_ner': ['sequence classification', 'into name-entity tags'],
        'mnli': ['text classification of two sentences', 'neutral, contradiction or entailment'],
        'qqp': ['text classification of two sentences', 'duplicate or not duplicate'],
        'sst2': ['text classification of reviews', 'negative or positive']
    }
    target_info = {
        'ARC-Challenge': ['multiple choice question answering', 'one of the provided options'],
        'comve_t1': ['common-sense reasoning', 'one or two'],
        'comve_t1': ['common-sense reasoning', 'one of the provided options'],
        'boolq': ['multiple choice question answering', 'true or false'],
        'medmcqa': ['multiple choice question answering of medical questions', 'one of the provided options'],
        'sciq': ['multiple choice question answering of science questions', 'one of the provided options'],
        'social_i_qa:': ['multiple choice question answering in social common-sense reasoning',
                         'one of the provided options'],
        'financial_phrasebank': ['text classification of reviews', 'duplicate or not duplicate'],
        'sst2': ['text classification', 'negative or positive']
    }

    def get_aligner(self, source, target):
        return self.aligner.format(self.source_info[source][0], self.source_info[source][1],
                                   self.target_info[target][0], self.target_info[target][1])

    def get_one_example(self, target_name):
        return self.target_demos[target_name]
