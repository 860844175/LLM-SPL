import pickle
import pandas as pd
from tqdm import tqdm
import networkx as nx
from itertools import combinations
import json

from glob import glob
import openai
import os
import tiktoken
import re
import multiprocessing as mp
from multiprocessing import Pool
import openai
import time


import argparse

def fix_spaces(string):
    
    fixed_string = '\n'.join([i.strip() for i in string.split('\n')])

    fixed_string = (('\n'.join([i for i in fixed_string.split('\n') if not (i in ['+', '-', '', '*'] or i.startswith('@@')or i.startswith('index') or i.startswith('+++') or i.startswith('---') or 'Copyright' in i)])))

    fixed_string = '\n'.join([re.sub(r'\s{2,}', ' ', i) if i.startswith('-') or i.startswith('+') else re.sub(r'\s{2,}', '\n', i) for i in fixed_string.split('\n')])
    return fixed_string

def truncate_text(text, max_tokens):
    tokens = encoding.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        return encoding.decode(tokens)
    else:
        return text

def read_code_diff(reponame, commit_id):
    code_diff_path = '../data/code_diff/{}/{}.txt'.format(reponame, id2commit[commit_id])
    if os.path.exists(code_diff_path):
        with open(code_diff_path, 'r', encoding = 'latin-1') as f:
            code_diff = f.read()
    else:
        code_diff = pickle.load(open('../data/code_diff/{}/{}.pkl'.format(reponame, id2commit[commit_id]), 'rb'))

    code_diff = fix_spaces(code_diff)
    return code_diff


def get_response(input):

    prompt = """ 
            Role: You are a software security analyst with expertise in analyzing, fixing, and patching vulnerabilities.

            Consider the following provided two commits, each containing the commit ID, commit description, and code diff. Your task is to analyze the commit information to determine whether these two commits jointly address a same problem, bug, issue, or vulnerability. Provide a conclusion and reasoning to support your answer.

            Additionally, at the end of your response, please give the answer again to the question whether these two commits jointly address a same problem, bug, issue, or vulnerability using a single line. The answer can only be either YES, NO or UNKNOWN.

            """ 


    reponame, commit1, commit2, output_folder = input

    out_path = '{}/[{},{}].json'.format(output_folder, commit1, commit2)
    out_path1 = '{}/[{},{}].json'.format(output_folder, commit2, commit1)
    if os.path.exists(out_path) or os.path.exists(out_path1):
        return

    element = {'reponame' : reponame,
                'commit1' : commit1,
                'commit2' : commit2}
    message1, message2 = commit_info_dict[reponame][id2commit[commit1]][0], commit_info_dict[reponame][id2commit[commit2]][0]

    code_diff1, code_diff2 = fix_spaces(read_code_diff(reponame, commit1)), fix_spaces(read_code_diff(reponame, commit2))

    commit_out1 = '\n\nCOMMIT 1:\n\nCOMMIT ID: {}\n\nMESSAGE: {}\n\nCODE DIFF: {}\n\n'.format(id2commit[commit1], message1, code_diff1)

    commit_out2 = '\n\nCOMMIT 2:\n\nCOMMIT ID: {}\n\nMESSAGE: {}\n\nCODE DIFF: {}\n\n'.format(id2commit[commit2], message2, code_diff2)


    commit_out1, commit_out2 = truncate_text(commit_out1, 1500), truncate_text(commit_out2, 1500)

    commit_out = commit_out1 + commit_out2

    prompt = prompt4 + commit_out

    element['prompt'] = prompt4

    element['commit_pair_info'] = commit_out

    response = openai.ChatCompletion.create(

        model = 'gpt-3.5-turbo-0613',
        messages = [
            {'role':'user', 'content': prompt}
        ],
    )
    output = response['choices'][0]['message']['content']

    element['response'] = output
    element['usage'] = response['usage']
    with open(out_path, 'w') as f:
        json.dump(element, f)


commit_info_dict = pickle.load(open('', 'rb'))

total_dataset = pickle.load(open('../dataset/total_dataset.pkl', 'rb'))

multi_cve_list = []
for cve, item in total_dataset.groupby('cve'):
    positive = item[item.label == 1]
    if len(positive) > 1:
        multi_cve_list.append(cve)
commit2ids = {commit:idx for idx, commit in enumerate(total_dataset.commit.unique())}
total_dataset['commit_id'] = total_dataset['commit'].apply(lambda x: commit2ids[x])
id2commit = {idx:commit for commit, idx in commit2ids.items()}

print('load encoding model')
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")


openai.api_key = ""

def main():


    repo_total_commits = {}
    repo_total_positive_commits = {}
    for idx, (reponame, item) in tqdm(enumerate(total_dataset[total_dataset.cve.isin(multi_cve_list)].groupby('repo')), total = len(total_dataset.repo.unique())):
        positive = item[item.label == 1]
        repo_total_positive_commits[reponame] = positive.commit_id.unique().tolist()
        repo_total_commits[reponame] = item.commit_id.unique().tolist()

    output_folder = './feedback/'

    edge_list = pickle.load(open('../data/top8_base_spl_relevance_feedback_edge_list.pkl', 'rb'))
    def multi_get_response(inputs, number_of_workers = 16):
        length = len(inputs)

        with Pool(number_of_workers) as p:
            list(
                tqdm(p.imap(get_response, [(reponame, commit1, commit2, output_folder) for (reponame, commit1, commit2) in (inputs)]), 
                total=length))
            p.close()
            p.join()


    multi_get_response(list(edge_list), number_of_workers = 10)



if __name__ == "__main__":
    main()
        




