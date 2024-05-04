import argparse
import pandas as pd
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as Data
import torch.nn.functional as F
from sklearn.model_selection import KFold
import torch.optim as optim
import torch.nn as nn
import warnings
import dgl
from model import *
from utils import *
import networkx as nx
warnings.filterwarnings("ignore")
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_auc_score
from loss import *

def main():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('graph_path', type = str)

    parser.add_argument('output_path', type = str)

    parser.add_argument("--dataset_path", "-d", type = str, default = '../dataset/processed_total_df.pkl')

    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float)

    parser.add_argument('--num_layers', '--num_layers_wd', default=3, type=int)

    parser.add_argument('--dropout', '--dropout', default=0.4, type=float)

    parser.add_argument('--batch_size', '--batch_size', default=40960, type=int)

    parser.add_argument('--num_epochs', '--num_epochs', default=200, type=int)

    parser.add_argument('--gnn_hidden_size', '--gnn_hidden_size', default=256, type=int)

    parser.add_argument('--graph_feature_path', '--graph_feature_path', default = '../dataset/graph_feature.pkl', type = str)

    parser.add_argument('--num_heads', '--num_heads', default=4, type=int)

    parser.add_argument('--weight_decay', '--weight_decay', default=1e-5, type=float)

    parser.add_argument('--alpha', '--alpha', default=1300, type=float)

    parser.add_argument('--writer_folder_name', '--writer_folder_name', default = '', type = str)


    args = parser.parse_args()

    graph_path = args.graph_path

    full_graph_path = '../dataset/{}'.format(graph_path)
    print('=== graph path: {} ===\n'.format(full_graph_path))


    output_path = args.output_path
    full_output_path = '../result/{}'.format(output_path)
    print('=== output path: {} ===\n'.format(full_output_path))

    dataset_path = args.dataset_path
    full_dataset_path = '../dataset/{}'.format(dataset_path)

    gnn_hidden_size = args.gnn_hidden_size
    print('=== gnn_hidden_size: {} ===\n'.format(gnn_hidden_size))

    num_heads = args.num_heads
    print('==num_heads: {}==\n'.format(num_heads))

    weight_decay = args.weight_decay
    print('==weight_decay: {}==\n'.format(weight_decay))

    alpha = args.alpha
    alpha = torch.tensor([1, alpha])
    print('==alpha: {}==\n'.format(alpha))

    lr = args.lr
    print('==lr: {}==\n'.format(lr))

    num_layers = args.num_layers
    print('==num_layers: {}==\n'.format(num_layers))

    dropout = args.dropout
    print('==dropout: {}==\n'.format(dropout))

    batch_size = args.batch_size
    print('==batch_size: {}==\n'.format(batch_size))
    
    num_epochs = args.num_epochs
    print('==num_epochs: {}==\n'.format(num_epochs))

    graph_feature_path = args.graph_feature_path
    print('==graph_feature_path: {}==\n'.format(graph_feature_path))

    # only for comparison

    dataset = pickle.load(open(full_dataset_path, 'rb'))

    multi_cve_list = []
    
    for cve, item in dataset.groupby('cve'):
        positive = item[item.label == 1]
        if len(positive) > 1:
            multi_cve_list.append(cve)

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

    graph_feature = pickle.load(open(graph_feature_path, 'rb'))

    print(f'\n======================= graph_feature.shape: {graph_feature.shape} =======================')

    total_graph = pickle.load(open(full_graph_path, 'rb'))

    print(f'======================= total_graph.shape: {len(total_graph)} =======================')

    G = nx.Graph()
    for reponame, u, v in total_graph:
        G.add_node(u)
        G.add_node(v)
        weight = 1.0
        if G.has_edge(u, v):
            G[u][v]['weight'] += weight
        else:
            G.add_edge(u, v, weight=weight)
            
    print("Number of nodes of G : {} Number of edges fo G : {}\n ".format(G.number_of_nodes(),G.number_of_edges()))


    src = [i[0] for i in G.to_directed().edges()]
    dst = [i[1] for i in G.to_directed().edges()]
    weights = [i[2]['weight'] for i in G.to_directed().edges(data=True)]
    commit2ids = {commit:idx for idx, commit in enumerate(dataset.commit.unique())}

    
    dgl_Graph = dgl.graph((src, dst), num_nodes=len(commit2ids))
    dgl_Graph.edata['weight'] = torch.tensor(weights)
    dgl_Graph = dgl.add_self_loop(dgl_Graph)


    embed_size = 32

    dropout = dropout

    batch_size = batch_size

    num_epochs = num_epochs

    learning_rate = lr


    cvelist = dataset.cve.unique()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    result = dataset[['cve', 'repo', 'commit', 'label']]
    result.loc[:, 'prob'] = 0

    for fold_idx, (train_index, test_index) in enumerate(kf.split(cvelist)):

        print('\nFold: ', fold_idx+1)

        cve_train = cvelist[train_index]

        train = dataset[dataset.cve.isin(cve_train)]
        test = dataset[dataset.cve.isin(cve_train) == False]
        test_result = test[['cve', 'commit', 'label']]
        test_result.loc[:, 'prob'] = 0

        train_dataset = Data.TensorDataset(
                                torch.LongTensor(train[['cve_id', 'commit_id']].values),
                                torch.LongTensor(train[sparse_features].values),
                                torch.FloatTensor(train[dense_features].values),
                                torch.FloatTensor(train['label'].values))
        
        test_dataset = Data.TensorDataset(
                                torch.LongTensor(test[['cve_id', 'commit_id']].values),
                                torch.LongTensor(test[sparse_features].values),
                                torch.FloatTensor(test[dense_features].values),
                                torch.FloatTensor(test['label'].values))


        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        print(f'positive samples in train dataset: {train.label.sum()}')

        print(f'positive samples in test dataset: {test.label.sum()}')
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        num_batches_per_epoch = int((len(train_dataset) - 1) / batch_size) + 1
        
        print(f'\nThe number of batch for each eopch : {num_batches_per_epoch}' )

        sparse_features_unique_nums = [dataset[i].nunique() for i in sparse_features]

        model = WideDeep(cate_fea_uniques = sparse_features_unique_nums, 
                    num_fea_size = len(dense_features), 
                    emb_size = embed_size,
                    num_classes = 2,
                    num_layers = num_layers,
                    dropout = dropout,
                    graph = dgl_Graph,
                    gnn_hidden_size = gnn_hidden_size,
                    num_heads = num_heads,
                    graph_feature = graph_feature,
                    device = device)

        if torch.cuda.device_count() > 1:
            print(f'\n{torch.cuda.device_count()} GPUs are available')
            
            model = nn.DataParallel(model)
        
        model.to(device)

        loss_function = FocalLoss(class_num = 2, alpha = alpha, gamma = 2)

        optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 25, gamma = 0.1)

        training_start_time = est_time(time.time())
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            print(f'\nEpoch {epoch}/{num_epochs}')
            model.train()
            train_prediction = []
            train_label = []
            train_prob = []
            train_loss = 0
            for idx, (ids, sparse, dense, label) in enumerate(train_dataloader):
                ids, sparse, dense, label = ids.to(device), sparse.to(device), dense.to(device), label.to(device)
                prob = model(ids, sparse, dense)
                prob = torch.clamp(prob, min=1e-32, max=1)
                loss = loss_function(prob, label.long())
                import math
                if math.isnan(loss.item()):
                    import ipdb; ipdb.set_trace()
                train_loss += (loss.item())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                prediction = torch.argmax(prob, dim=1)

                train_prediction.extend(prediction.cpu().detach().numpy().tolist())
                train_label.extend(label.cpu().detach().numpy().tolist())
                train_prob.extend(prob[:,1].cpu().detach().numpy().tolist())
            print('{:.4f}, {:.4f}, {:.4f}'.format(weight1, weight2, weight3))

            train_f1 = f1_score(train_label, train_prediction) 
            writer.add_scalar('Train/F1', train_f1, epoch)

            train_auc = roc_auc_score(train_label, train_prob)
            writer.add_scalar('Train/Auc', train_auc, epoch)

            train_avg_loss = train_loss / (idx + 1)
            writer.add_scalar('Train/AvgLoss', train_avg_loss, epoch)

            print('epoch : {}, train_auc : {:.6f}, train_f1 : {:.6f}, '.format(epoch, train_auc, train_f1))    
            scheduler.step()
            if epoch >= 2 and epoch % 1 == 0 and epoch != 1:
                predict_prob = []
                model.eval()
                with torch.no_grad():
                    test_prediction = []
                    test_label = []
                    test_prob = []
                    test_loss = 0
                    for idx, (ids, sparse, dense, label) in enumerate(test_dataloader):
                        ids, sparse, dense, label = ids.to(device), sparse.to(device), dense.to(device), label.to(device)
                        prob = model(ids, sparse, dense)
                        prob = torch.clamp(prob, min=1e-32, max=1)
                        loss = loss_function(prob, label.long())
                        import math
                        if math.isinf(loss.item()):
                            import ipdb; ipdb.set_trace()
                        test_loss += loss.item()
                        predict_prob.extend(prob[:,1].tolist())
                        prediction = torch.argmax(prob, dim=1)
                        test_prediction.extend(prediction.cpu().detach().numpy().tolist())
                        test_label.extend(label.cpu().detach().numpy().tolist()) 
                    test_auc = roc_auc_score(test_label, predict_prob)
                    writer.add_scalar('Test/Auc', test_auc, epoch)
                
                    test_f1 = f1_score(test_label, test_prediction) 
                    writer.add_scalar('Test/F1', test_f1, epoch)

                    test_result['prob'] = predict_prob
                    test_result.reset_index(drop=True, inplace=True)
                    test_result['rank_prob'] = get_rank(test_result, ['prob'], False)

                    multi_score_1 = int(get_cve_patch_cnt_by_rank(test_result[test_result.cve.isin(multi_cve_list)], 'rank_prob')[0])

                    multi_score_2 = get_ndcg_score(test_result[test_result.cve.isin(multi_cve_list)])
                    
                    aver_test_loss = test_loss / (idx + 1)
                    writer.add_scalar('Test/AvgLoss', aver_test_loss, epoch)
                    print('test loss: {:6f}'.format(aver_test_loss)) 

                    print('multi_score_1 : {:.6f}, multi_score_2 : {:.6f}, test auc: {:.4f}, test f1 : {:.6f}'.format(multi_score_1, multi_score_2, test_auc, test_f1)
                    
        result.loc[test.index, 'prob'] = predict_prob 

    pickle.dump(result, open(full_output_path, 'wb'))


if __name__ == '__main__':
    main()