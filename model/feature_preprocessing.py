import json

import argparse
import pickle
import pandas as pd 
import torch.nn as nn
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import torch
from transformers import AutoModel
import numpy as np



def main():

    parser = argparse.ArgumentParser(description='data processing')

    parser.add_argument('-d', '--dataset_path', default = 'total_dataset.pkl', type=str)

    parser.add_argument('-o', '--output_path', default = 'processed_total_dataset.pkl', type=str)

    args = parser.parse_args()

    dataset_path = args.dataset_path
    full_dataset_path = '../dataset/{}'.format(dataset_path)
    print("== Loaded Dataset Path: {}".format(full_dataset_path))

    output_path = args.output_path
    full_output_path = '../dataset/{}'.format(output_path)

    print("== Processed Dataset Path: {}".format(full_output_path))

    vuln_cols = ['vuln_emb' + str(i) for i in range(128)]
    cmt_cols = ['cmt_emb' + str(i) for i in range(128)]

    dataset = pickle.load(open(full_dataset_path, 'rb'))

    dataset.drop(columns = ['message'], inplace = True)

    print('The shape of raw dataset: ', dataset.shape)

    print("The number of CVE in the dataset: ", len(dataset.cve.unique()))

    print('number of gpt feature: {}'.format(dataset.gpt_feature.sum()))



    dense_features = ['addcnt', 'delcnt', 'total_cnt', 'issue_cnt', 'web_cnt', 'bug_cnt', 'cve_cnt',
                    'time_dis', 'inter_token_cwe_cnt', 'inter_token_cwe_ratio', 'vuln_commit_tfidf',
                    'func_same_cnt', 'func_same_ratio', 'func_unrelated_cnt',
                    'filepath_same_cnt', 'filepath_same_ratio', 'filepath_unrelated_cnt',
                    'file_same_cnt', 'file_same_ratio', 'file_unrelated_cnt', 'vuln_type_1',
                    'vuln_type_2', 'vuln_type_3', 'mess_shared_num', 'mess_shared_ratio',
                    'mess_max', 'mess_sum', 'mess_mean', 'mess_var', 'code_shared_num',
                    'code_shared_ratio', 'code_max', 'code_sum', 'code_mean', 'code_var']

    dense_features += ['llm_relevance_score']


    sparse_features = ['cve_match_raw', 'bug_match_raw', 'cwe_match', 'func_same_ratio_range', 'vuln_commit_tfidf_ratio_range', 
                        'file_same_ratio_range', 'filepath_same_ratio_range', 'total_cnt_range', 'addcnt_range', 'delcnt_range', 
                        'filepath_same_cnt_range', 'func_same_cnt_range', 'file_same_cnt_range', 'time_range']

    num_bins = 10
    # =================================================================================================
    dataset['time_range'] = pd.cut(dataset['time_dis'], bins=num_bins, labels=[str(i) for i in range(num_bins)])
    dataset['mess_shared_ratio_range'] = pd.cut(dataset['mess_shared_ratio'], bins=num_bins, labels=[str(i) for i in range(num_bins)])
    dataset['file_same_ratio_range'] = pd.cut(dataset['file_same_ratio'], bins=num_bins, labels=[str(i) for i in range(num_bins)])
    dataset['func_same_ratio_range'] = pd.cut(dataset['func_same_ratio'], bins=num_bins, labels=[str(i) for i in range(num_bins)])
    dataset['vuln_commit_tfidf_ratio_range'] = pd.cut(dataset['vuln_commit_tfidf'], bins=num_bins, labels=[str(i) for i in range(num_bins)])
    dataset['filepath_same_ratio_range'] = pd.cut(dataset['filepath_same_ratio'], bins=num_bins, labels=[str(i) for i in range(num_bins)])
    dataset['code_shared_ratio_range'] = pd.cut(dataset['code_shared_ratio'], bins=num_bins, labels=[str(i) for i in range(num_bins)])

    dataset['cve_match_raw'] = dataset['cve_match'].astype('category')
    dataset['bug_match_raw'] = dataset['bug_match'].astype('category')
    dataset['cwe_match'] = pd.cut(dataset['inter_token_cwe_ratio'], bins=num_bins, labels=[str(i) for i in range(num_bins)])

    dataset['total_cnt_range'] = pd.cut(dataset['total_cnt'], bins=num_bins, labels=[str(i) for i in range(num_bins)])
    dataset['addcnt_range'] = pd.cut(dataset['addcnt'], bins=num_bins, labels=[str(i) for i in range(num_bins)])
    dataset['delcnt_range'] = pd.cut(dataset['delcnt'], bins=num_bins, labels=[str(i) for i in range(num_bins)])

    dataset['filepath_same_cnt_range'] = pd.cut(dataset['filepath_same_cnt'], bins=num_bins, labels=[str(i) for i in range(num_bins)])
    dataset['func_same_cnt_range'] = pd.cut(dataset['func_same_cnt'], bins=num_bins, labels=[str(i) for i in range(num_bins)])
    dataset['file_same_cnt_range'] = pd.cut(dataset['file_same_cnt'], bins=num_bins, labels=[str(i) for i in range(num_bins)])
    # dataset['time_range'] = dataset['time_range'].astype('category')

    dataset['func_unrelated_cnt_range'] = pd.cut(dataset['func_unrelated_cnt'], bins=num_bins, labels=[str(i) for i in range(num_bins)])

    dataset['code_mean_range'] = pd.cut(dataset['code_mean'], bins=num_bins, labels=[str(i) for i in range(num_bins)])
    dataset['code_sum_range'] = pd.cut(dataset['code_sum'], bins=num_bins, labels=[str(i) for i in range(num_bins)])
    dataset['code_var_range'] = pd.cut(dataset['code_var'], bins=num_bins, labels=[str(i) for i in range(num_bins)])
    dataset['mess_sum_range'] = pd.cut(dataset['mess_sum'], bins=num_bins, labels=[str(i) for i in range(num_bins)])
    dataset['code_shared_num_range'] = pd.cut(dataset['code_shared_num'], bins=num_bins, labels=[str(i) for i in range(num_bins)])
    dataset['mess_shared_num_range'] = pd.cut(dataset['mess_shared_num'], bins=num_bins, labels=[str(i) for i in range(num_bins)])
    dataset['inter_token_cwe_cnt_range'] = pd.cut(dataset['inter_token_cwe_cnt'], bins=num_bins, labels=[str(i) for i in range(num_bins)])

    dataset['code_max_range'] = pd.cut(dataset['code_max'], bins=num_bins, labels=[str(i) for i in range(num_bins)])
    dataset['web_cnt_range'] = pd.cut(dataset['web_cnt'], bins=num_bins, labels=[str(i) for i in range(num_bins)])
    dataset['file_unrelated_cnt_range'] = pd.cut(dataset['file_unrelated_cnt'], bins=num_bins, labels=[str(i) for i in range(num_bins)])
    dataset['filepath_unrelated_cnt_range'] = pd.cut(dataset['filepath_unrelated_cnt'], bins=num_bins, labels=[str(i) for i in range(num_bins)])
    dataset['cve_cnt_range'] = pd.cut(dataset['cve_cnt'], bins=num_bins, labels=[str(i) for i in range(num_bins)])

    dataset['mess_mean_range'] = pd.cut(dataset['mess_mean'], bins=num_bins, labels=[str(i) for i in range(num_bins)])
    dataset['mess_var_range'] = pd.cut(dataset['mess_var'], bins=num_bins, labels=[str(i) for i in range(num_bins)])
    dataset['bug_cnt_range'] = pd.cut(dataset['bug_cnt'], bins=num_bins, labels=[str(i) for i in range(num_bins)])
    dataset['mess_max_range'] = pd.cut(dataset['mess_max'], bins=num_bins, labels=[str(i) for i in range(num_bins)])
    dataset['issue_cnt_range'] = pd.cut(dataset['issue_cnt'], bins=num_bins, labels=[str(i) for i in range(num_bins)])
    dataset['vuln_type_1_range'] = pd.cut(dataset['vuln_type_1'], bins=num_bins, labels=[str(i) for i in range(num_bins)])
    dataset['vuln_type_2_range'] = pd.cut(dataset['vuln_type_2'], bins=num_bins, labels=[str(i) for i in range(num_bins)])
    dataset['vuln_type_3_range'] = pd.cut(dataset['vuln_type_3'], bins=num_bins, labels=[str(i) for i in range(num_bins)])


    sparse_features = dataset.select_dtypes('category').columns.tolist()

    for feat in tqdm(sparse_features):
        lbe = LabelEncoder()
        dataset[feat] = lbe.fit_transform(dataset[feat])
    # import ipdb;ipdb.set_trace()
        
    for feat in tqdm(dense_features):
        mean = dataset[feat].mean()
        std = dataset[feat].std()
        dataset[feat] = (dataset[feat] - mean) / (std + 1e-12) 

    dense_features += vuln_cols + cmt_cols

    print('\nnumber of dense features: ', len(dense_features))
    # import ipdb; ipdb.set_trace()

    print('\nnumber of sparse features: ', len(sparse_features))



    vuln_data = pd.read_csv('../../data/vuln_data_20230117.csv')
    vuln_data = vuln_data[~vuln_data.cve.isin(multi_repo_cve_list)]
    vuln_data = vuln_data[vuln_data.cve.isin(cwe_data.cve.unique())]


    print(f'======================= vuln_data.shape: {vuln_data.shape} =======================')

    repo2ids = {reponame:idx for idx, reponame in enumerate(dataset.repo.unique())}
    # Key:cve Value: cve_id
    cve2ids = {i:idx  for idx, i in enumerate(vuln_data.cve.unique())}
    # Key:commit Value: commit_id
    commit2ids = {commit:idx for idx, commit in enumerate(dataset.commit.unique())}

    print('min commit_id : {}, max commit_id : {}'.format(min(commit2ids.values()), max(commit2ids.values())))

    dataset['cve_id'] = [cve2ids[i] for i in dataset.cve]
    dataset['commit_id'] = [commit2ids[i] for i in dataset.commit]

    print('\nvuln_embed_list.shape: ', vuln_embed_list.shape)

    vuln_embed_table = pd.DataFrame(vuln_embed_list, columns = ['vuln_emb' + str(i) for i in range(len(vuln_cols))])

    vuln_embed_table['cve_id'] = vuln_embed_table.index

    dataset = dataset.merge(vuln_embed_table, on='cve_id', how='left')

    mess_embed_list = torch.tensor(pickle.load(open('../data/commit_message_embedding.pkl', 'rb')).numpy())
    mess_embed_table = pd.DataFrame(mess_embed_list, columns = ['cmt_emb' + str(i) for i in range(len(cmt_cols))])
    mess_embed_table['commit_id'] = mess_embed_table.index
    dataset = dataset.merge(mess_embed_table, on='commit_id', how='left')

    graph_feature = torch.tensor(mess_embed_list, dtype=torch.float32)

    pickle.dump(dataset, open(full_output_path, 'wb'))

    pickle.dump(graph_feature, open('../dataset/graph_feature.pkl', 'wb'))

    


if __name__ == '__main__':
    main()