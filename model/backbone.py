import torch
from torch_geometric.nn.models import GCN, GAT, GIN
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

class Classifier(torch.nn.Module):
    def __init__(self, nin, nhid, nout, nlayer, type="GIN"):
        super().__init__()
        self.nin = nin
        self.nhid = nhid
        self.nout = nout

        if type == "GCN":
            self.model = GCN(nin, nhid, nlayer, nhid)
        elif type == "GAT":
            self.model = GAT(nin, nhid, nlayer, nhid)
        elif type == "GIN":
            self.model = GIN(nin, nhid, nlayer, nhid)
        
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(nhid * 2, nhid),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(nhid, nout)
        )

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        _x = self.model(x, edge_index, edge_weight, batch)
        
        if batch is None:
            batch = torch.zeros(_x.size(0), dtype=torch.int64, device=_x.device)
        
        _x = torch.cat([gmp(_x, batch), gap(_x, batch)], dim=1)
        _x = self.mlp(_x)
        return _x