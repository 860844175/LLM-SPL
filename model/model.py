import torch
from torch import nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl import DGLGraph
from dgl.nn import GraphConv, SAGEConv, GATConv
import numpy as np
import torch.nn.init as init


gcn_msg = fn.u_mul_e('h', 'weight', 'm')
    
gcn_reduce = fn.sum(msg='m', out='h')


class GAT(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads):
        super(GAT, self).__init__()
        self.layer1 = GATConv(in_dim, hidden_dim, num_heads)
        self.layer2 = GATConv(hidden_dim * num_heads, out_dim, 1)

    def forward(self, g, h):
        h = self.layer1(g, h)
        h = h.view(-1, h.size(1) * h.size(2)) # (in_feat, num_heads, out_dim) -> (in_feat, num_heads * out_dim)
        h = F.elu(h)
        h = self.layer2(g, h)
        h = h.squeeze() # (in_feat, 1, out_dim) -> (in_feat, out_dim)
        return h


class Gate(nn.Module):
    def __init__(self, input_dim, num_classes = 2, method = 'addition'):
        super(Gate, self).__init__()
        self.method = method
        self.linear_1 = nn.Linear(input_dim, 1)
        self.linear_2 = nn.Linear(input_dim, 1)
        self.linear_3 = nn.Linear(input_dim, 1)
        torch.nn.init.xavier_normal_(self.linear_1.weight)
        torch.nn.init.xavier_normal_(self.linear_2.weight)
        torch.nn.init.xavier_normal_(self.linear_3.weight)
        if method == 'addition':
            self.output_linear = nn.Linear(input_dim, num_classes)
        elif method == 'concat':
            self.output_linear = nn.Linear(input_dim * 3, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, A, B, C):
        a_weight = self.sigmoid(self.linear_1(A))   
        b_weight = self.sigmoid(self.linear_2(B))   
        c_weight = self.sigmoid(self.linear_3(C))   
        
        total_weight = a_weight + b_weight + c_weight
        
        a_weight_normalized = torch.div(a_weight, total_weight)
        b_weight_normalized = torch.div(b_weight, total_weight)
        c_weight_normalized = torch.div(c_weight, total_weight)
        
        if self.method == 'addition':
            combined = a_weight_normalized * A + b_weight_normalized * B + c_weight_normalized * C
        elif self.method == 'concat':
            combined = torch.cat((a_weight_normalized * A, b_weight_normalized * B, c_weight_normalized * C), dim = 1)

        combined = self.output_linear(combined)
        return combined
        
class WideDeep(nn.Module):
    def __init__(self, cate_fea_uniques,
                 num_fea_size=0,
                 emb_size=8,
                 hidden_dims=256,
                 num_classes=1,
                 num_layers = 2,
                 dropout= 0.5,
                 graph = None,
                 graph_feature = None, 
                 gnn_hidden_size = 256,
                 num_heads = 4, 
                 device = None):
        '''
        :param cate_fea_uniques:
        :param num_fea_size: 
        :param emb_size:
        :param hidden_dims:
        :param num_classes:
        :param dropout:
        '''
        super(WideDeep, self).__init__()
        self.cate_fea_size = len(cate_fea_uniques)
        self.num_fea_size = num_fea_size
        self.num_classes = num_classes
        self.emb_size = emb_size
        hidden_dims = [hidden_dims] * num_layers
        dropout = [dropout] * num_layers
        self.device = device
        self.graph = graph.to(self.device)
        self.graph_feature = graph_feature.to(self.device)
        self.GAT = GAT(self.graph_feature.shape[1], gnn_hidden_size, self.emb_size, num_heads)
        self.GAT.to(self.device)
        
        init_method = nn.init.xavier_normal_

        self.sparse_emb = nn.ModuleList([])
        for voc_size in cate_fea_uniques:
            emb = nn.Embedding(voc_size, emb_size)
            init_method(emb.weight)
            self.sparse_emb.append(emb)
            
        self.wide_linear = nn.Linear(self.num_fea_size, self.emb_size)
        init_method(self.wide_linear.weight)
        
        
        self.all_dims = [self.cate_fea_size * emb_size + self.num_fea_size] + hidden_dims

        for i in range(1, len(self.all_dims)):
            linear = nn.Linear(self.all_dims[i-1], self.all_dims[i])
            init_method(linear.weight)
            setattr(self, 'linear_' + str(i), linear)
            setattr(self, 'batchNorm_' + str(i), nn.BatchNorm1d(self.all_dims[i]))
            setattr(self, 'activation_' + str(i), nn.ReLU())
            setattr(self, 'dropout_' + str(i), nn.Dropout(dropout[i-1]))
        
        self.gate = Gate(self.emb_size, self.num_classes, 'concat')

        
        self.dnn_linear = nn.Linear(self.all_dims[-1], self.emb_size)
        init_method(self.dnn_linear.weight)
        
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
    

    def forward(self, X_id, X_sparse, X_dense=None):
        # import ipdb; ipdb.set_trace()
        """
        X_sparse: sparse_feature [batch_size, sparse_feature_num]
        X_dense: dense_feature  [batch_size, dense_feature_num]
        """
        commit_ids = X_id[:,1]
        
        sparse_embed = [emb(X_sparse[:, i]) for i, emb in enumerate(self.sparse_emb)]
        sparse_embed = torch.cat(sparse_embed, dim=-1)   # batch_size, sparse_feature_num * emb_dim

        
        x = torch.cat([sparse_embed, X_dense], dim=-1)

    
        wide_out = self.wide_linear(X_dense)
        
        dnn_out = x
        for i in range(1, len(self.all_dims)):
            dnn_out = getattr(self, 'linear_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'batchNorm_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'activation_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'dropout_' + str(i))(dnn_out)
        
        deep_out = self.dnn_linear(dnn_out)
    

        gcn = self.GAT(self.graph, self.graph_feature)

        gcn_out = gcn[commit_ids]
        gcn_out = torch.reshape(gcn_out, (gcn_out.shape[0], -1))

        out = self.gate(wide_out, deep_out, gcn_out)
        
        out = self.softmax(out)

        return out