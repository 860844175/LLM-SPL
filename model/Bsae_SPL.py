from util import *
from encoding_module import *
import pandas as pd
import numpy as np
import random
import pickle
import gc
import math
import time
import logging
from tqdm import tqdm
import xgboost as xgb
import os
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.metrics import mean_squared_log_error 

import argparse

import warnings
warnings.filterwarnings('ignore')


# ============ Linear Regression ============
def linear_regression(X_train, y_train, X_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    predict = model.predict(X_test)
    return predict


# ============ Logistic Regression ============
def logistic_regression(X_train, y_train, X_test):
    model = LogisticRegression(
        class_weight='balanced', solver='saga', multi_class='ovr', n_jobs=5, max_iter=200)
    model.fit(X_train, y_train)
    predict = model.predict_proba(X_test)[:, 0]
    return predict
    # return model


# ============ XGBoost ============
def xgboost(X_train, y_train, X_test):
    param = {
        'max_depth': 5,
        'eta': 0.05,
        'verbosity': 1,
        'random_state': 2021,
        'objective': 'binary:logistic',
        'tree_method': 'gpu_hist'
    }

    def myFeval(preds, dtrain):
        labels = dtrain.get_label()
        return 'error', math.sqrt(mean_squared_log_error(preds, labels))
    print("XGBoost 训练 & 预测")
    xgb_train = xgb.DMatrix(X_train, y_train)
    model = xgb.train(param, xgb_train, num_boost_round=500, feval=myFeval)
    predict = model.predict(xgb.DMatrix(X_test))
    return predict


# ============ LightGBM ============
def lightgbm(X_train, y_train, X_test):
    param = {'device': 'cpu',
             'learning_rate': 0.04,
             'max_depth': 5,
             'verbose': -1,
             'metric' : 'binary_logloss',
             }
    
    print("LGBM 训练 & 预测")
    model = lgb.train(param, lgb.Dataset(
        data=X_train, label=y_train), num_boost_round=500)
    predict = model.predict(X_test)
    return predict


# ============ CNN ============
class CNNDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(np.array(X), dtype=torch.float)
        self.y = torch.tensor(np.array(y), dtype=torch.long)
        self.len = self.X.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data = self.X[idx]
        label = self.y[idx]
        return data, label


class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)
        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]
        probs = (P*class_mask).sum(1).view(-1, 1)
        log_p = probs.log()
        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class Net(nn.Module):
    def __init__(self, num_feature):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(num_feature, 32),
            nn.Linear(32, 8),
            # nn.Linear(8, 1)
            nn.Linear(8, 2)
        )
        self.soft = nn.Softmax()

    def forward(self, input_):
        s1 = self.model(input_)
        out = self.soft(s1)
        return out


def cnn(X_train, y_train, X_test):
    lr = 0.001
    num_workers = 10
    alpha = 10
    batch_size = 1024
    num_epochs = 20
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = FocalLoss(class_num=2, alpha=torch.tensor([1, 1000]))

    train_dataset = CNNDataset(X_train, y_train)
    test_dataset = CNNDataset(X_test, pd.Series([1]*X_test.shape[0]))
    num_feature = X_train.shape[1]
    model = Net(num_feature).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  pin_memory=False)
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 pin_memory=False)

    print("CNN 训练 & 预测")
    for epoch in range(num_epochs):
        model.train()
        predict = []
        t1 = time.time()
        train_prediction = []
        train_label = []
        for i, (data, label) in enumerate(train_dataloader):
            data = data.to(device)
            label = label.to(device)
            label_size = data.size()[0]
            pred = model(data)
            prediction = torch.argmax(pred, dim=1)
            train_prediction.extend(prediction.cpu().detach().numpy().tolist())
            train_label.extend(label.cpu().detach().numpy().tolist())
            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        t2 = time.time()
        logging.info('Epoch [{}/{}], Time {}s, Loss: {:.4f}, Lr:{:.4f}'.format(
            epoch + 1, num_epochs, int(t2 - t1), loss.item(), lr))
        torch.save(model.state_dict(),
                   '../data/cnn_20_{:02}.ckpt'.format(epoch))
        train_recall = recall_score(train_label, train_prediction)
        train_precision = precision_score(train_label, train_prediction)
        train_f1 = f1_score(train_label, train_prediction) 
        train_accuracy = accuracy_score(train_label, train_prediction) 
        print(f'epoch : {epoch}, train_recall : {train_recall}, train_precision : {train_precision}, train_f1 : {train_f1}, train_accuracy : {train_accuracy}') 

    model.eval()
    test_prediction = []
    test_label = []
    with torch.no_grad():
        predict = []
        for i, (data, label) in enumerate(test_dataloader):
            data = data.to(device)
            pred = model(data)
            prediction = torch.argmax(pred, dim=1)
            test_prediction.extend(prediction.cpu().detach().numpy().tolist())
            test_label.extend(label.cpu().detach().numpy().tolist())
            predict.extend(pred.cpu().detach().numpy().tolist())
        test_recall = recall_score(test_label, test_prediction)
        test_precision = precision_score(test_label, test_prediction)
        test_f1 = f1_score(test_label, test_prediction)
        test_accuracy = accuracy_score(test_label, test_prediction)
        predict = np.array(predict)
        return predict



# ======================== metric ========================

# sort data based on 'sortby' list, and then get the rank of each data
def get_rank(df, sortby, ascending=False):
    gb = df.groupby('cve')
    l = []
    for item1, item2 in gb:
        item2 = item2.reset_index()
        item2 = item2.sort_values(sortby + ['commit'], ascending=ascending)
        item2 = item2.reset_index(drop=True).reset_index()
        l.append(item2[['index', 'level_0']])

    df = pd.concat(l)
    df['rank'] = df['level_0']+1
    df = df.sort_values(['index'], ascending=True).reset_index(drop=True)
    return df['rank']

def multi_get_score(result, rank_col_name, n = 10):
    overall_recall = 0
    for idx, (cve, item) in enumerate(result.groupby('cve')):
        positive = item[item.label == 1]
        item_sorted = item.sort_values(by= rank_col_name, ascending=True).head(n)
        rank_n = set(positive.commit.tolist()) & (set(item_sorted.commit.tolist()))
        recall_score = len(rank_n) / len(positive)
        overall_recall += recall_score
    return overall_recall/result.cve.nunique()
    
def multi_get_full_score(df, suffix, result2, start = 1, end = 10):
    metric1_list = []
    metric2_list = []
    for i in range(start, end+1):
        metric1 = multi_get_score(df, 'rank_'+suffix, i)
        metric1_list.append(metric1)
    result2['metric1_'+suffix] = metric1_list
    return result2

# =============== model fusion ===============

def fusion_max(result, cols):
    def get_max(row, columns):
        return max([row[column] for column in columns])

    result['fusion_max'] = result.apply(lambda row: get_max(row, cols), axis=1)
    result['rank_fusion_max'] = get_rank(result, ['fusion_max'], False)
    result.drop(['fusion_max'], axis=1)
    return result


def fusion_min(result, cols):
    def get_min(row, columns):
        return min([row[column] for column in columns])

    result['fusion_min'] = result.apply(lambda row: get_min(row, cols), axis=1)
    result['rank_fusion_min'] = get_rank(result, ['fusion_min'], False)
    result.drop(['fusion_min'], axis=1)
    return result


def fusion_sum(result, cols):
    def get_sum(row, columns):
        return sum([row[column] for column in columns])

    result['fusion_sum'] = result.apply(lambda row: get_sum(row, cols), axis=1)
    result['rank_fusion_sum'] = get_rank(result, ['fusion_sum'], False)
    result.drop(['fusion_sum'], axis=1)
    return result


def fusion_borda(row, cols):
    def get_sum(row, columns):
        return sum([row[column] for column in columns])

    result['fusion_borda'] = result.apply(
        lambda row: get_sum(row, cols), axis=1)
    result['rank_fusion_borda'] = get_rank(result, ['fusion_borda'], True)
    result.drop(['fusion_borda'], axis=1)
    return result


def fusion_voting(result, cols, suffix=''):
    def get_closest(row, columns):
        l = [row[column] for column in columns]
        l.sort()
        if l[1] - l[0] >= l[2] - l[1]:
            return l[1]+l[2]
        else:
            return l[1]+l[0]

    result['closest'] = result.apply(
        lambda row: get_closest(row, cols), axis=1)
    result['sum'] = 0
    for column in cols:
        result['sum'] = result['sum'] + result[column]
    result['last'] = result['sum'] - result['closest']
    result['rank_fusion_voting' +
        suffix] = get_rank(result, ['closest', 'last'], True)
    result.drop(['sum', 'closest', 'last'], axis=1)
    return result

def main():

    parser = argparse.ArgumentParser(description='base SPL training')

    parser.add_argument('dataset_path', default = '../dataset/total_dataset.csv', type = str)

    args = parser.parse_args()

    dataset_path = args.dataset_path

    vuln_data_path = '../data/vuln_data.csv'

    df = pickle.load(open(dataset_path, 'rb'))

    commit2ids = {commit:idx for idx, commit in enumerate(df.commit.unique())}
    commit_id_list = [commit2ids[commit] for commit in df.commit]
    df['commit_id'] = commit_id_list

    feature_cols = ['addcnt', 'delcnt', 'total_cnt', 'issue_cnt', 'web_cnt', 'bug_cnt', 'cve_cnt',
                    'time_dis', 'inter_token_cwe_cnt', 'inter_token_cwe_ratio', 'vuln_commit_tfidf',
                    'cve_match', 'bug_match', 'func_same_cnt', 'func_same_ratio', 'func_unrelated_cnt',
                    'filepath_same_cnt', 'filepath_same_ratio', 'filepath_unrelated_cnt',
                    'file_same_cnt', 'file_same_ratio', 'file_unrelated_cnt', 'vuln_type_1',
                    'vuln_type_2', 'vuln_type_3', 'mess_shared_num', 'mess_shared_ratio',
                    'mess_max', 'mess_sum', 'mess_mean', 'mess_var', 'code_shared_num',
                    'code_shared_ratio', 'code_max', 'code_sum', 'code_mean', 'code_var']

    if 'llm_relevance_score' in df.columns:
        output_folder_name = 'iter0'
    else:
        output_folder_name = 'iter1'
        feature_cols += ['llm_relevance_score']
    

    multi_cve_list = []
    for cve, item in df.groupby('cve'):
        positive = item[item.label == 1]
        if len(positive) > 1:
            multi_cve_list.append(cve)

    print('\nLoad the message embedding')
    mess_embed_table = torch.tensor(pickle.load(open('../data/commit_message_embedding.pkl', 'rb')).numpy())
    mess_embed_table = pd.DataFrame(mess_embed_table, columns = ['cmt_emb' + str(i) for i in range(mess_embed_table.shape[1])])
    mess_embed_table['commit_id'] = mess_embed_table.index
    df = df.merge(mess_embed_table, on='commit_id', how='left')

    print('\nLoad the vuln embedding')
    vuln_data = pd.read_csv(vuln_data_path)

    print(f'shape of vuln_data : {vuln_data.shape}')
    cve2ids = {i:idx  for idx, i in enumerate(vuln_data.cve.unique())}
    cve_id_list = [cve2ids[cve] for cve in df.cve]
    df['cve_id'] = cve_id_list

    vuln_embed_table = pickle.load(open('../data/cve_desc_embedding.pkl', 'rb'))
    vuln_embed_table = pd.DataFrame(vuln_embed_table, columns = ['vuln_emb' + str(i) for i in range(vuln_embed_table.shape[1])])
    vuln_embed_table['cve_id'] = vuln_embed_table.index
    df = df.merge(vuln_embed_table, on='cve_id', how='left')
    cvelist = df.cve.unique()
    kf = KFold(n_splits=5, shuffle=True, random_state = 42)
    repolist = df.repo.unique()



    vuln_cols = ['vuln_emb' + str(i) for i in range(128)]
    cmt_cols = ['cmt_emb' + str(i) for i in range(128)]

    result = df[['cve', 'commit', 'label']]
    result.loc[:, 'prob_linear'] = 0
    result.loc[:, 'prob_logistic'] = 0
    result.loc[:, 'prob_xgb'] = 0
    result.loc[:, 'prob_lgb'] = 0
    result.loc[:, 'prob_cnn'] = 0

    total_ps_predict = []
    for idx, (train_index, test_index) in enumerate(kf.split(cvelist)):

        # split by cve
        cve_train = cvelist[train_index]
        train = df[df.cve.isin(cve_train)]
        print('shape of train : {}'.format(train.shape))

        test = df[df.cve.isin(cve_train) == False]
        print('shape of test : {}'.format(test.shape))

        print(f'The number of repo in training data : {train.repo.nunique()}')
        print(f'The number of repos in testing data : {test.repo.nunique()}')
        tmp_train = train[['cve', 'repo', 'commit', 'label']].copy()
        tmp_test = test[['cve', 'repo', 'commit', 'label']].copy()

        print('X_train and X_test')
        X_train = train[feature_cols + vuln_cols + cmt_cols]

        y_train = train['label']
        X_test = test[feature_cols + vuln_cols + cmt_cols]
        y_test = test['label']
        ps_X_train = train[ps_cols]
        ps_X_test = test[ps_cols]

        # xgboost
        xgb_predict = xgboost(X_train, y_train, X_test)
        result.loc[X_test.index, 'prob_xgb'] = xgb_predict
        
        # lightgbm
        lgb_predict = lightgbm(X_train, y_train, X_test)
        result.loc[X_test.index, 'prob_lgb'] = lgb_predict

        # cnn
        cnn_predict = cnn(X_train, y_train, X_test)
        result.loc[X_test.index, 'prob_cnn'] = cnn_predict[:,1]

    result['rank_xgb'] = get_rank(result, ['prob_xgb'])
    result['rank_lgb'] = get_rank(result, ['prob_lgb'])
    result['rank_cnn'] = get_rank(result, ['prob_cnn'])
    pickle.dump(result, open('../{}/rank_result.pkl'.format(output_folder_name), 'wb'))

    result2 = multi_get_full_score(result, 'xgb', result2)
    result2 = multi_get_full_score(result, 'lgb', result2)
    result2 = multi_get_full_score(result, 'cnn', result2)
    pickle.dump(result2, open('../{}/metric_result.pkl'.format(output_folder_name), 'wb'))

    result = pickle.load(open('../{}/rank_result.pkl'.format(output_folder_name), 'rb'))
    tmp_col1 = ['prob_xgb', 'prob_lgb', 'prob_cnn']
    tmp_col2 = ['rank_xgb', 'rank_lgb', 'rank_cnn']
    result = fusion_max(result, tmp_col1)
    result = fusion_min(result, tmp_col1)
    result = fusion_sum(result, tmp_col1)
    result = fusion_borda(result, tmp_col2)
    result = fusion_voting(result, tmp_col2)
    pickle.dump(result, open('../{}/rank_fusion_result.pkl'.format(output_folder_name), 'wb'))


    # save metric result
    result2 = pd.DataFrame()
    result2 = multi_get_full_score(result, 'fusion_max', result2)
    result2 = multi_get_full_score(result, 'fusion_min', result2)
    result2 = multi_get_full_score(result, 'fusion_sum', result2)
    result2 = multi_get_full_score(result, 'fusion_borda', result2)
    result2 = multi_get_full_score(result, 'fusion_voting', result2)

    pickle.dump(result2, open('../{}/metric_fusion_result.pkl'.format(output_folder_name), 'wb'))