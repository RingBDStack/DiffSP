import os
import sys
import warnings
warnings.filterwarnings('ignore')
sys.path.append(os.path.abspath('./'))

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from easydict import EasyDict as edict
from torch_geometric.nn import global_mean_pool


def get_data(path, device):
    data = torch.load(os.path.abspath(path), map_location=device)
    train_dataset = data['train']
    val_dataset = data['val']
    test_dataset = data['test']
    attack_dataset = data['attack']
    num_classes = data['num_classes']
    return train_dataset, val_dataset, test_dataset, attack_dataset, num_classes


def get_marginal(dataset):
    E_marginal = [0, 0]
    for graph in dataset:
        num_nodes = graph.x.size(0)
        num_edges = graph.edge_index.size(1)
        
        E_marginal[0] += num_nodes * num_nodes - num_edges
        E_marginal[1] += num_edges
    
    E_marginal = torch.tensor(E_marginal)
    E_marginal = E_marginal / torch.sum(E_marginal)
    
    return E_marginal


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.norm = gcn_norm
        self.conv1 = GCNConv(in_channels, hidden_channels, normalize=False)
        self.conv2 = GCNConv(hidden_channels, hidden_channels, normalize=False)
        self.fc = torch.nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None, **kwargs):
        # Normalize edge indices only once:
        if not kwargs.get('skip_norm', False):
            edge_index, edge_weight = self.norm(
                edge_index,
                edge_weight,
                num_nodes=x.size(0),
                add_self_loops=True,
            )

        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        x = global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        x = self.fc(x)
        return x
    
    def get_features(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = self.conv2(x, edge_index, edge_weight)
        # x = self.ln1(x)
        return x

def evaluate(model, val_dataset):
    # model.eval()
    ys = torch.concat([graph.y for graph in val_dataset])
    preds = []
    with torch.no_grad():
        for graph in val_dataset:
            x, edge_index = graph.x, graph.edge_index
            pred = model(x, edge_index)
            preds.append(pred)
    preds = torch.concat(preds)
    accuracy = (preds.argmax(dim=1) == ys).float().mean().item()
    return accuracy


from g.model import GraphDDPM
from g.train import train_ddpm
import random
import numpy as np
def main(args):
    args = edict(args)
    name = args.name
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    train_dataset, val_dataset, test_dataset, attack_dataset, num_classes \
        = get_data(f'./data/{name}/{args.attack}_evasion.pth', device)

    train_val_dataset = train_dataset + val_dataset
    E_marginal = get_marginal(train_val_dataset)
    E_marginal = E_marginal.to(device)

    num_features = train_dataset[0].x.size(1)
    model = GraphDDPM(num_features, args.T_E, E_marginal, args.gnn_E, device).to(device)
    gcn = GCN(num_features, 16, num_classes).to(device)

    ck_path = f'./checkpoint/diffusion/{name}/n_step_E{args.T_E}'
    if args.is_train:
        train_ddpm(args, model, train_val_dataset, device, ck_path)

    save_gcn_path = f'./checkpoint/classifier/{name}/GCN.pth'
    gcn.load_state_dict(torch.load(save_gcn_path, map_location=device))

    checkpoint = torch.load(ck_path + '.pth', map_location=device)
    model.graph_encoder.pred_E.load_state_dict(checkpoint['pred_E_state_dict'])
    # purify_dataset = model.purify(attack_dataset, args.purify.t_E)
    purify_dataset = model.anisotropy_transfer_entropy_guided_purify(gcn, attack_dataset, args.purify.basic_t, args.purify.adaptive_t, args.purify.k, args.purify.lamb, device)
    
    acc_pert = evaluate(gcn, attack_dataset)
    acc_puri = evaluate(gcn, purify_dataset)

    print(f'acc pert {acc_pert:.4f} acc puri {acc_puri:.4f}')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

from parameter import parameter_parser
if __name__ == "__main__":
    config = parameter_parser()
    for _ in range(10):
        seed = random.randint(0, 1000)
        print(seed)
        set_seed(seed)
        set_seed(config.seed)
        main(config)