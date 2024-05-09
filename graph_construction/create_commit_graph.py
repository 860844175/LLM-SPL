import pandas as pd
import numpy as np
import pickle
import os
import networkx as nx
import json
from multiprocessing import Pool
import multiprocessing as mp
from itertools import combinations, chain
from tqdm import tqdm
import datetime


def match_committer_time(commit_info_1, commit_info_2, time_threshold):
    commit_date_1 = commit_info_1[2]
    commit_date_2 = commit_info_2[2]
    return calculate_commit_commit_difference_days(commit_date_1, commit_date_2) <= time_threshold

def match_author_time(commit_info_1, commit_info_2, time_threshold):
    authored_date1 = commit_info_1[-1]
    authored_date2 = commit_info_2[-1]
    return calculate_commit_commit_difference_days(authored_date1, authored_date2) <= time_threshold

def calculate_commit_commit_difference_seconds(commit_date1, commit_date2):
    d1 = datetime.datetime.strptime(commit_date1, "%Y%m%d %H%M%S")
    d2 = datetime.datetime.strptime(commit_date2, "%Y%m%d %H%M%S")
    if d2 >= d1:
        return abs((d2-d1).total_seconds())
    else:
        return abs((d2-d1).total_seconds())

def calculate_commit_commit_difference_days(commit_date1, commit_date2):
    d1 = datetime.datetime.strptime(commit_date1, "%Y%m%d %H%M%S")
    d2 = datetime.datetime.strptime(commit_date2, "%Y%m%d %H%M%S")
    if d2 >= d1:
        return abs((d2-d1).days)
    else:
        return abs((d1-d2).days)


def match_committer(commit_info_1, commit_info_2):
    committer_1 = commit_info_1[1]
    committer_2 = commit_info_2[1]
    
    return committer_1 == committer_2 and committer_1

def match_author(commit_info_1, commit_info_2):
    author_1 = commit_info_1[-2]
    author_2 = commit_info_2[-2]
    return author_1 == author_2 and author_1

def match_parent(commit_id_1, commit_id_2, commit_info_1, commit_info_2):
    cmt_id_1, cmit_id_2 = commit2ids[commit_id_1], commit2ids[commit_id_2]
    parent_1 = [i for i in commit_info_1[4]]
    parent_2 = commit_info_2[4]
    if commit_id_1 in parent_2 or commit_id_2 in parent_1:
        return True
    else:
        return False


def get_commit_connection(inputs):

    reponame, commit1, commit2, time_interval = inputs

    commit_info1, commit_info2 = commit_info_dict[reponame][commit1], commit_info_dict[reponame][commit2]

    if match_committer_time(commit_info1, commit_info2, time_interval) and match_committer(commit_info1, commit_info2):
        return [commit1, commit2, {"type": 'committed_and_time'}]
    
    if match_author_time(commit_info1, commit_info2, time_interval) and match_author(commit_info1, commit_info2):
        return [commit1, commit2, {"type": 'authored_and_time'}]

    if match_parent(commit1, commit2, commit_info1, commit_info2):
        return [commit1, commit2, {"type": 'commits_parent'}]


def multi_get_commit_graph(inputs, number_of_workers = 16):
    with Pool(number_of_workers) as p:
        ret = list(
            tqdm(p.imap(get_commit_connection, inputs),
                 total = len(inputs))
                 )
        p.close()
        p.join()
    ret = list(chain(*ret))
    return ret


def main():  

    parser = argparse.ArgumentParser(description='raw commit graph construction')
    
    parser.add_argument('n', type = int)

    args = parser.parse_args()

    n = args.n

    dataset_df = pd.read_csv('../dataset/total_dataset.csv')

    cve2ids = {i:idx  for idx, i in enumerate(dataset_df.cve.unique())}
    id2cve = {idx:i  for idx, i in enumerate(dataset_df.cve.unique())}
    commit2ids = {commit:idx for idx, commit in enumerate(dataset_df.commit.unique())}
    id2commits = {idx:commit for commit, idx in commit2ids.items()}
    dataset_df['cve_id'] = dataset_df.cve.apply(lambda x: cve2ids[x])
    dataset_df['commit_id'] = dataset_df.commit.apply(lambda x: commit2ids[x])

    commit_info_dict = pickle.load(open('', 'rb'))

    number_of_workers = mp.cpu_count()

    inputs = list(commit_info_dict.keys())
    raw_commit_graph = multi_get_commit_graph(inputs, number_of_workers)

    pickle.dump(raw_commit_graph, open('../graph/raw_commit_graph.pkl', 'wb'))