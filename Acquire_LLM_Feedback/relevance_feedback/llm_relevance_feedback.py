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
    
    # # fixed_string = string
    fixed_string = '\n'.join([i.strip() for i in string.split('\n')])

    fixed_string = (('\n'.join([i for i in fixed_string.split('\n') if not (i in ['+', '-', '', '*'] or i.startswith('@@')or i.startswith('index') or i.startswith('+++') or i.startswith('---') or 'Copyright' in i)])))

    fixed_string = '\n'.join([re.sub(r'\s{2,}', ' ', i) if i.startswith('-') or i.startswith('+') else re.sub(r'\s{2,}', '\n', i) for i in fixed_string.split('\n')])

    return fixed_string

def truncate_text(text, max_tokens):
    try:
        tokens = encoding.encode(text)
    except:
        tokens = encoding.encode(text, disallowed_special=())
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
        code_diff = pickle.load(open('../data//code_diff/{}/{}.pkl'.format(reponame, id2commit[commit_id]), 'rb'))

    code_diff = fix_spaces(code_diff)
    return code_diff


def get_response(input):

    prompt = """Role: You are a software security analyst with expertise in analyzing, fixing, and patching vulnerabilities.

                Considering the following provided specific vulnerability information from CVE (including CVE ID and CVE description) and the corresponding commit data from its related repository (including commit ID, commit description, and code diff). Your task is to analyze the vulnerability and commit information to determine whether this commit is a patch for the vulnerability. Provide a conclusion and reasoning to support your answer.

                Additionally, at the end of your response, please give the answer again to the question whether this commit is a patch for the vulnerability using a single line. The answer can only be either YES, NO, or UNKNOWN. 
                """
    
    reponame, cve, cmt, output_folder = input

    out_path = '{}/[{},{}].json'.format(output_folder, cve, cmt)

    if os.path.exists(out_path):
        return

    element = {'reponame' : reponame,
               'cve' : id2cve[cve],
               'commit' : id2commit[cmt],
               'prompt': prompt}
    
    desc = cve_desc_dict[cve]

    message = commit_info_dict[reponame][id2commit[cmt]][0]

    code_diff = fix_spaces(read_code_diff(reponame, cmt))

    cve_out = "CVE: {}\n\n CVE description : {}\n\n".format(id2cve[cve], desc)

    cmt_out = "COMMIT: {}\n\nMESSAGE: {}\n\nCODE DIFF: {}\n\n".format(id2commit[cmt], message, code_diff)

    cmt_out = truncate_text(cmt_out, 2000)

    info_out = cve_out + cmt_out

    prompt = prompt + info_out

    element['cve_commit_info'] = info_out

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

total_dataset = pickle.load(open('../dataset/total_dataset', 'rb'))

multi_cve_list = []
for cve, item in total_dataset.groupby('cve'):
    positive = item[item.label == 1]
    if len(positive) > 1:
        multi_cve_list.append(cve)
commit2ids = {commit:idx for idx, commit in enumerate(total_dataset.commit.unique())}
total_dataset['commit_id'] = total_dataset['commit'].apply(lambda x: commit2ids[x])
id2commit = {idx:commit for commit, idx in commit2ids.items()}

vuln_data = pd.read_csv('../data/vuln_data.csv')
cve2ids = {cve:idx for idx, cve in enumerate(vuln_data.cve.unique())}
vuln_data['cve_id'] = vuln_data['cve'].apply(lambda x: cve2ids[x])
id2cve = {idx:cve for cve, idx in cve2ids.items()}

total_dataset['cve_id'] = total_dataset['cve'].apply(lambda x: cve2ids[x])

cve_desc_dict = {cve:desc for cve, desc in zip(vuln_data.cve_id, vuln_data.desc)}

output_folder = './feedback/'

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

    edge_list = pickle.load(open('./top100_base_spl_edge_list.pkl', 'rb'))


    def multi_get_response(inputs, number_of_workers = 16):
        length = len(inputs)

        with Pool(number_of_workers) as p:
            list(
                tqdm(p.imap(get_response, [(reponame, cve, commit, output_folder) for (reponame, cve, commit) in (inputs)]), 
                total=length))
            p.close()
            p.join()


    multi_get_response(list(edge_list), number_of_workers = 10)


if __name__ == "__main__":
    main()
        




