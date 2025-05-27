import os
import time
import argparse
print("Script started")

parser = argparse.ArgumentParser()
parser.add_argument('-s', default=" ", type=str)
parser.add_argument('-k', default=1, type=int)
parser.add_argument('-m', default="meta-llama/Llama-2-7b-hf", type=str)
parser.add_argument('-d', default="0", type=str)
parser.add_argument('-method', default="embsim", type=str)
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
    model_name = args.m
    task_pairs = {'ARC-Challenge': 'race', 'financial_phrasebank': 'ag_news', 'medmcqa': 'boolq', 'sciq': 'race', 'social_i_qa': 'race'}

for target,source in task_pairs.items():
    result_dir = os.path.join('results', f"dataset_name={target}")
    if os.path.exists(result_dir):
        if args.method == "embsim":
            cmd_line = f"python embed_similarity.py --source_dataset_name {source} --target_dataset_name {target} --model_name {model_name} --device cuda:{args.d} --k {args.k}"
        elif args.method == "graphsim":
            cmd_line = f"python structure_similarity.py --source_dataset_name {source} --target_dataset_name {target} --model_name {model_name} --device cuda:{args.d} --suffix {args.s} --k {args.k}"

        print(cmd_line)
        ret_status = os.system(cmd_line)
        if ret_status != 0:
            print('DRIVER (non-zero exit status from execution)>>{ret_status}<<')
            exit()

