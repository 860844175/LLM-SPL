import pandas as pd
import pickle
import numpy as np
import git
import multiprocessing as mp
from tqdm import tqdm
from multiprocessing import Pool
from itertools import chain

gitpath = '../data/gitrepo/'

code_diff_folder = '../data/code_diff/'


def get_code_diff(inputs):
    reponame, commit = inputs

    code_diff_path1 = os.path.join(code_diff_folder, reponame, commit + '.txt')
    code_diff_path2 = os.path.join(code_diff_folder, reponame, commit + '.pkl')
    if os.path.exists(code_diff_path1) or os.path.exists(code_diff_path2):
        return
    repo = git.Repo(gitpath + reponame)
    code_diff = repo.git.diff(commit + '~1',
                    commit,
                    ignore_blank_lines=True,
                    ignore_space_at_eol=True)
    try:
        with open(code_diff_path1, 'w') as f:
            f.write(code_diff)
    except:
        pickle.dump(code_diff, open(code_diff_path2, 'wb'))


def multi_get_code_diff(inputs, number_of_works = 16):
    
    reponame, commits = inputs
    length = len(commits)
    with Pool(number_of_works) as p:
        list(p.imap(get_code_diff, [(reponame, commit) for commit in commits]))

        p.close()
        p.join()

repo_list = list(commit_info_dict.keys())
for reponame in tqdm(repo_list, total = len(repo_list)):
    code_diff_dict[reponame]  = {}
    commits = list(commit_info_dict[reponame].keys())
    multi_get_code_diff((reponame, commits), 16)
    


