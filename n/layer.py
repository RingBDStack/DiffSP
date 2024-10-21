import torch
import torch.nn as nn
import dgl
from torch_sparse import SparseTensor
from torch_geometric.nn.conv import TransformerConv
from torch_geometric.nn.conv import GATConv
import math
import torch.nn as nn

class GNNLayer(nn.Module):
    def __init__(self,
                 hidden_X,
                 hidden_t,
                 dropout):
        super().__init__()

        self.update_X = nn.Sequential(
            nn.Linear(hidden_X + hidden_t, hidden_X),
            nn.ReLU(),
            nn.LayerNorm(hidden_X),
            nn.Dropout(dropout)
        )

    def forward(self, A, h_X, h_t):
        if A.nnz == 0: # if there is no edges, there will be error while A @ h_X
            h_aggr_X = h_X
        else:
            h_aggr_X = A @ h_X

        num_nodes = h_X.size(0)
        h_t_expand = h_t.expand(num_nodes, -1)
        h_aggr_X = torch.cat([h_aggr_X, h_t_expand], dim=1)

        h_X = self.update_X(h_aggr_X)

        return h_X
    

class ATTLayer(nn.Module):
    def __init__(self,
                 hidden_X,
                 hidden_t,
                 dropout):
        super().__init__()

        self.update_X = nn.Sequential(
            nn.Linear(hidden_X + hidden_t, hidden_X),
            nn.ReLU(),
            nn.LayerNorm(hidden_X),
            nn.Dropout(dropout)
        )

        self.att = TransformerConv(hidden_X, hidden_X, heads=4, concat=False)

    def forward(self, edge_index, h_X, h_t):
        num_nodes = h_X.size(0)
        # row, col = torch.triu_indices(num_nodes, num_nodes, 1)
        # full_edge_index = torch.stack([row, col]).to(h_X.device)
        
        h_att_X = self.att(h_X, edge_index)

        num_nodes = h_X.size(0)
        h_t_expand = h_t.expand(num_nodes, -1)
        h_att_X = torch.cat([h_att_X, h_t_expand], dim=1)

        h_X = self.update_X(h_att_X)

        return h_X

class GMHLayer(nn.Module):
    def __init__(self,
                 hidden_X,
                 hidden_t,
                 dropout,
                 num_heads=4):
        super().__init__()

        
        self.hidden_X = hidden_X
        self.hidden_t = hidden_t
        self.num_heads = num_heads

        self.fc_q = nn.Linear(hidden_X, hidden_X)
        self.fc_k = GATConv(hidden_X, hidden_X)
        self.fc_v = GATConv(hidden_X, hidden_X)

        self.att = nn.MultiheadAttention(hidden_X, num_heads)

        self.update_X = nn.Sequential(
            nn.Linear(hidden_X + hidden_t, hidden_X),
            nn.ReLU(),
            nn.LayerNorm(hidden_X),
            nn.Dropout(dropout)
        )
        self.activation = torch.tanh 

    def forward(self, edge_index, h_X, h_t):
        num_nodes = h_X.size(0)
        # h_att_X = self.att(h_X, edge_index)
        Q = self.fc_q(h_X)
        K = self.fc_k(h_X, edge_index)
        V = self.fc_v(h_X, edge_index)

        # dim_split = self.hidden_X // self.num_heads
        # Q_ = torch.stack(Q.split(dim_split, -1), 0)
        # K_ = torch.stack(K.split(dim_split, -1), 0)
        # V_ = torch.stack(V.split(dim_split, -1), 0)
        # # print(Q_.size(), K_.size(), V_.size()); exit()
        # A = torch.softmax(Q_.bmm(K_.transpose(1, 2))/math.sqrt(self.hidden_X), -1)
        # h_att_X = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), -1)

        Q_ = Q
        K_ = K
        V_ = V
        attention_score = Q_.mm(K_.transpose(0, 1)) / math.sqrt(self.hidden_X)
        A = torch.softmax(attention_score, dim=-1)
        h_att_X = A.mm(V_)
        
        # h_att_X, a = self.att(Q, K, V)

        num_nodes = h_X.size(0)
        h_t_expand = h_t.expand(num_nodes, -1)
        h_att_X = torch.cat([h_att_X, h_t_expand], dim=1)

        h_X = self.update_X(h_att_X)

        return h_X


class GNNTower(nn.Module):
    def __init__(self,
                 in_X,
                 hidden_t,
                 hidden_X,
                 out_size,
                 num_gnn_layers,
                #  num_att_layers,
                 dropout):
        super().__init__()


        self.mlp_in_t = nn.Sequential(
            nn.Linear(1, hidden_t),
            nn.ReLU(),
            nn.Linear(hidden_t, hidden_t),
            nn.ReLU())
        
        self.mlp_in_X = nn.Sequential(
            nn.Linear(in_X, hidden_X),
            nn.ReLU(),
            nn.Linear(hidden_X, hidden_X),
            nn.ReLU()
        )

        self.gnn_layers = nn.ModuleList([
            GNNLayer(hidden_X,
                     hidden_t,
                     dropout)
            for _ in range(num_gnn_layers)])
        
        # self.att_layers = nn.ModuleList([
        #     ATTLayer(hidden_X,
        #              hidden_t,
        #              dropout)
        #     for _ in range(num_att_layers)])

        # +1 for the input attributes
        # hidden_cat = (num_gnn_layers + num_att_layers + 1) * (hidden_X) + hidden_t
        # hidden_cat = (num_att_layers + 1) * (hidden_X) + hidden_t
        hidden_cat = (num_gnn_layers + 1) * (hidden_X) + hidden_t
        self.mlp_out = nn.Sequential(
            nn.Linear(hidden_cat, hidden_cat),
            nn.ReLU(),
            nn.Linear(hidden_cat, out_size)
        )
        

    def forward(self,
                t_float,
                X_t_one_hot,
                A_t):
        # Input projection.
        # (1, hidden_t)
        h_t = self.mlp_in_t(t_float).unsqueeze(0)
        h_X = self.mlp_in_X(X_t_one_hot)

        h_X_list = [h_X]
        for gnn in self.gnn_layers:
            h_X = gnn(A_t, h_X, h_t)
            h_X_list.append(h_X)
        
        # row, col = A_t.coo()
        # edge_index = torch.stack([row, col], dim=0)
        # for att in self.att_layers:
        #     h_X = att(edge_index, h_X, h_t)
        #     h_X_list.append(h_X)

        # (|V|, hidden_t)
        h_t = h_t.expand(h_X.size(0), -1)
        h_cat = torch.cat(h_X_list + [h_t], dim=1)

        return self.mlp_out(h_cat)
    

class LinkPredictor(nn.Module):
    def __init__(self,
                 in_X,
                 num_classes_E,
                 hidden_t,
                 hidden_X,
                 hidden_E,
                 num_gnn_layers,
                #  num_att_layers,
                 dropout):
        super().__init__()

        self.gnn_encoder = GNNTower(in_X,
                                    hidden_t,
                                    hidden_X,
                                    hidden_E,
                                    num_gnn_layers,
                                    # num_att_layers,
                                    dropout)
        self.mlp_out = nn.Sequential(
            nn.Linear(hidden_E, hidden_E),
            nn.ReLU(),
            nn.Linear(hidden_E, num_classes_E)
        )

    def forward(self,
                t_float,
                X_t_one_hot,
                A_t,
                src,
                dst):
        # (|V|, hidden_E)
        h = self.gnn_encoder(t_float,
                             X_t_one_hot,
                             A_t)
        
        # (|E|, hidden_E)
        h = h[src] * h[dst]
        # (|E|, num_classes_E)
        logit = self.mlp_out(h)

        return logit


class GNNAsymm(nn.Module):
    def __init__(self,
                 in_X,
                 num_classes_E,
                 gnn_E_config):
        super().__init__()


        self.pred_E = LinkPredictor(in_X,
                                    num_classes_E,
                                    **gnn_E_config)

    def forward(self,
                t_float_E,
                X_one_hot_2d,
                A_t,
                batch_src,
                batch_dst):
        
        logit_E = self.pred_E(t_float_E,
                              X_one_hot_2d,
                              A_t,
                              batch_src,
                              batch_dst)

        return logit_E
    
