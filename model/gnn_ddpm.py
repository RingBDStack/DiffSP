import torch
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

class GNNTower(nn.Module):
    def __init__(self,
                 num_attrs_X,
                 num_classes_X,
                 hidden_t,
                 hidden_X,
                 out_size,
                 num_gnn_layers,
                 dropout):
        super().__init__()

        in_X = num_attrs_X * num_classes_X

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

        # +1 for the input attributes
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

        # (|V|, hidden_t)
        h_t = h_t.expand(h_X.size(0), -1)
        h_cat = torch.cat(h_X_list + [h_t], dim=1)

        return self.mlp_out(h_cat)

class LinkPredictor(nn.Module):
    def __init__(self,
                 num_attrs_X,
                 num_classes_E,
                 num_classes_X,
                 hidden_t,
                 hidden_X,
                 hidden_E,
                 num_gnn_layers,
                 dropout):
        super().__init__()

        self.gnn_encoder = GNNTower(num_attrs_X,
                                    num_classes_X,
                                    hidden_t,
                                    hidden_X,
                                    hidden_E,
                                    num_gnn_layers,
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

class MLPLayer(nn.Module):
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

    def forward(self, h_X, h_t):
        num_nodes = h_X.size(0)
        h_t_expand = h_t.expand(num_nodes, -1)
        h_X = torch.cat([h_X, h_t_expand], dim=1)

        h_X = self.update_X(h_X)

        return h_X

class MLPTower(nn.Module):
    def __init__(self,
                 num_attrs_X,
                 num_classes_X,
                 hidden_t,
                 hidden_X,
                 num_mlp_layers,
                 dropout):
        super().__init__()

        self.num_attrs_X = num_attrs_X
        in_X = num_attrs_X * num_classes_X

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

        self.mlp_layers = nn.ModuleList([
            MLPLayer(hidden_X,
                     hidden_t,
                     dropout)
            for _ in range(num_mlp_layers)])

        # +1 for the input features
        hidden_cat = (num_mlp_layers + 1) * (hidden_X) + hidden_t
        self.mlp_out = nn.Sequential(
            nn.Linear(hidden_cat, hidden_cat),
            nn.ReLU(),
            nn.Linear(hidden_cat, in_X)
        )

    def forward(self,
                t_float,
                X_t_one_hot):
        # Input projection.
        h_t = self.mlp_in_t(t_float).unsqueeze(0)
        h_X = self.mlp_in_X(X_t_one_hot)
        
        h_X_list = [h_X]
        for mlp in self.mlp_layers:
            h_X = mlp(h_X, h_t)
            h_X_list.append(h_X)
        
        h_t = h_t.expand(h_X.size(0), -1)
        h_cat = torch.cat(h_X_list + [h_t], dim=1)
        logit = self.mlp_out(h_cat)

        # (|V|, F, C)
        logit = logit.reshape(X_t_one_hot.size(0), self.num_attrs_X, -1)

        return logit

class GNNAsymm_WithX(nn.Module):
    def __init__(self,
                 num_attrs_X,
                 num_classes_E,
                 num_classes_X,
                 mlp_X_config,
                 gnn_E_config):
        super().__init__()

        self.pred_X = MLPTower(num_attrs_X,
                               num_classes_X,
                               **mlp_X_config)

        self.pred_E = LinkPredictor(num_attrs_X,
                                    num_classes_E,
                                    num_classes_X,
                                    **gnn_E_config)

    def forward(self,
                t_float_X,
                t_float_E,
                X_t_one_hot,
                X_one_hot_2d,
                A_t,
                batch_src,
                batch_dst):

        logit_X = self.pred_X(t_float_X,
                              X_t_one_hot)
        
        logit_E = self.pred_E(t_float_E,
                              X_one_hot_2d,
                              A_t,
                              batch_src,
                              batch_dst)

        return logit_X, logit_E
    

class GNNAsymm_WoutX(nn.Module):
    def __init__(self,
                 num_attrs_X,
                 num_classes_E,
                 num_classes_X,
                 gnn_E_config):
        super().__init__()


        self.pred_E = LinkPredictor(num_attrs_X,
                                    num_classes_E,
                                    num_classes_X,
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