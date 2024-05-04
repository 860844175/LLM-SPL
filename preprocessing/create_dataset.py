import git
import numpy as np
import pandas as pd
import time
import os
import random
import pickle
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing as mp

import random


'''
creating pair dataset, get sampled commit from each repo as negative samples to form pair data.

Output:
1. data/repo_commit.txt: a dict, key is repo name, value is a list of sampled commit from this repo.

2. data/Dataset.csv: columns = ['cve', 'repo', 'true_commit', 'commit', 'label']
 
'''

num_cores = mp.cpu_count()
print('num_cores: ', num_cores)

repo_path ='../gitrepo'

print('folder_path: ', folder_path)

df = pd.read_csv('../data/data_df.csv')

repos = df.repo.unique()
repo_commit = dict()



def get_qualified_sample_commit(reponame):
    t1 = time.time()
    repo = git.Repo(os.path.join(repo_path, reponame))
    total_commits = [str(item) for item in repo.iter_commits()]
    commits = []
    for commit in (total_commits):
        try:
            repo.git.diff(commit+ '~1', commit)
            if repo.commit(commit).message != '':
                commits.append(commit)
        except:
            pass
        if len(commits) >= 1000:
            break
    return commits

def multi_process_get_qualified_sample_commit(repos, pool_nums = 16):
    with Pool(pool_nums) as p:
        res = list(tqdm(p.imap(get_qualified_sample_commit, repos), total = len(repos), desc = 'get_qualified_sample_commit'))
    p.close()
    p.join()
    result = dict()
    for idx, item in enumerate(res):
        result[repos[idx]] = item
    return result

repo_commit = multi_process_get_qualified_sample_commit(repos, pool_nums=num_cores)

pickle.dump(repo_commit, open('../data/repo_commit.pkl', 'wb'))


sample_size = 1500
random.seed(2021)

total_list = []
for idx, (cve, item) in enumerate(df.groupby('cve')):
    reponame = list(item.repo.unique())[0]
    pos_commit = list(item.commit.unique())[0]

    repo_total_commits = set(df[df.repo == reponame].commit.unique().tolist()).union(set(repo_commit[reponame]))

    neg_commits = list(repo_total_commits - set([pos_commit]))
    if len(neg_commits) >= sample_size:
        neg_commits = random.sample(neg_commits, sample_size)

    total_list.append([cve, reponame, pos_commit, 1])
    for commit in neg_commits:
        total_list.append([cve, reponame, commit, 0])

total_df = pd.DataFrame(total_list, columns=['cve', 'repo', 'commit', 'label'])

print('total_df.shape: ', total_df.shape)

repo_commit1 = {}
for reponame in repos:
    df_tmp = total_df[total_df.repo == reponame]
    commit_tmp = df_tmp.commit.unique()
    repo_commit1[reponame] = commit_tmp

pickle.dump(total_df, open('{}/Dataset.pkl'.format(folder_path), 'wb'))

pickle.dump(repo_commit1, open('{}/repo_commits.pkl'.format(folder_path), 'wb'))




