import os
import time
import argparse
print("Script started")

parser = argparse.ArgumentParser()
parser.add_argument('-s', default=" ", type=str)
parser.add_argument('-k', default=1, type=int)
parser.add_argument('-m', default="meta-llama/Llama-2-7b-hf", type=str)
parser.add_argument('-d', default="0", type=str)
parser.add_argument('-method', default="sim", type=str)
#simple
parser.add_argument("-w", action="store_true")


args = parser.parse_args()
if ("7b" in args.m) or ("8B" in args.m):
    model_name = args.m
    task_pairs = {'ARC-Challenge': 'ARC-Easy', 'financial_phrasebank': 'sst2', 'medmcqa': 'commonsense_qa',
                  'sciq': 'commonsense_qa', 'social_i_qa': 'race'}
elif "13b" in args.m:
    model_name = "meta-llama/Llama-2-13b-hf"
    task_pairs = {'ARC-Challenge': 'ARC-Easy', 'financial_phrasebank': 'qqp', 'medmcqa': 'race',
                  'sciq': 'commonsense_qa', 'social_i_qa': 'race'}

elif "gpt" in args.m:
    model_name = "gpt-3.5-turbo"
    task_pairs = {'ARC-Challenge': 'race', 'financial_phrasebank': 'ag_news', 'medmcqa': 'boolq', 'sciq': 'race', 'social_i_qa': 'race'}

for target,source in task_pairs.items():
    result_dir = os.path.join('results', f"dataset_name={target}")
    if os.path.exists(result_dir):
        if args.method == "sim":
            if args.w:
                cmd_line = f"python simple_aggregate.py --source_dataset_name {source} --target_dataset_name {target} --model_name {model_name} --device cuda:{args.d} --suffix {args.s} --weighted {args.w} --k {args.k}"
            else:
                cmd_line = f"python simple_aggregate.py --source_dataset_name {source} --target_dataset_name {target} --model_name {model_name} --device cuda:{args.d} --suffix {args.s} --k {args.k}"
        elif args.method == "graph":
            if args.w:
                cmd_line = f"python structure_similarity.py --source_dataset_name {source} --target_dataset_name {target} --model_name {model_name} --device cuda:{args.d} --suffix {args.s} --weighted {args.w} --k {args.k}"
            else:
                cmd_line = f"python structure_similarity.py --source_dataset_name {source} --target_dataset_name {target} --model_name {model_name} --device cuda:{args.d} --suffix {args.s} --k {args.k}"
        elif args.method == "bsl":
            cmd_line = f"python embed_similarity.py --source_dataset_name {source} --target_dataset_name {target} --model_name {model_name} --device cuda:{args.d} --suffix {args.s} --k {args.k}"
        print(cmd_line)
        ret_status = os.system(cmd_line)
        if ret_status != 0:
            print('DRIVER (non-zero exit status from execution)>>{ret_status}<<')
            exit()
-mthod

# python structure_similarity.py --source_dataset_name 'ARC-Easy' --target_dataset_name 'ARC-Challenge' --model_name "meta-llama/Llama-2-13b-hf" --device cuda:0 --suffix 112 --k 4