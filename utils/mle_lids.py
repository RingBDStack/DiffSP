
import torch
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm

def get_neighbors(node, adj):
    return torch.nonzero(adj[node], as_tuple=True)[0].tolist()

def get_neighbors_from_set(node_set, adj):
    neighbors = set()
    for node in node_set:
        neighbors.update(get_neighbors(node, adj))
    return neighbors

def get_hop_neighbors(node, adj, hop=1):
    neighbors = set([node])
    for _ in range(hop):
        neighbors.update(get_neighbors_from_set(neighbors, adj))
    neighbors.remove(node)  # 移除自己
    return list(neighbors)

def get_closest_neighbors(node, x, adj, num_neighbors=3, max_hops=10):
    def find_neighbors(node, current_hop):
        if current_hop > max_hops:
            return []
        neighbors = get_hop_neighbors(node, adj, hop=current_hop)
        if len(neighbors) >= num_neighbors:
            return neighbors
        return find_neighbors(node, current_hop + 1)
    
    neighbors = find_neighbors(node, 1)
    
    if len(neighbors) < num_neighbors:
        # print("find all but not enough neighbors"); exit()
        return None, None
        temp_num_neighbors = len(neighbors)
        neighbors_features = x[neighbors]
        distances = cdist(x[node].unsqueeze(0).numpy(), neighbors_features.numpy()).squeeze()
        distances_tensor = torch.tensor(distances)  
        closest_neighbors = [neighbors[i] for i in torch.argsort(distances_tensor)[:temp_num_neighbors]]
        closest_distances = distances_tensor[torch.argsort(distances_tensor)[:temp_num_neighbors]]
        return closest_neighbors, closest_distances
    
    # 计算最近的 num_neighbors 个邻居
    neighbors_features = x[neighbors]
    distances = cdist(x[node].unsqueeze(0).numpy(), neighbors_features.numpy()).squeeze()
    distances_tensor = torch.tensor(distances)  # 将 distances 转换为 tensor
    closest_neighbors = [neighbors[i] for i in torch.argsort(distances_tensor)[:num_neighbors]]
    closest_distances = distances_tensor[torch.argsort(distances_tensor)[:num_neighbors]]
    return closest_neighbors, closest_distances

from torch_geometric.nn import GCNConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.optim import Adam
import torch.nn.functional as F
import torch.nn as nn

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.norm = gcn_norm
        self.conv1 = GCNConv(in_channels, hidden_channels, normalize=False)
        self.conv2 = GCNConv(hidden_channels, hidden_channels, normalize=False)
        self.ln1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.ln2 = nn.Linear(hidden_channels // 2, out_channels)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None, **kwargs):
        x = self.conv1(x, edge_index, edge_weight)
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        x = self.ln1(x)
        x = self.ln2(x)
        x = F.softmax(x, dim=-1)
        return x

    def get_features(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        # x = self.ln1(x)
        return x

def train_classifier(model, x, edge_index, y, idx_train, epochs=400, lr=0.01, weight_decay=5e-4):
    model.train()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in tqdm(range(epochs)):
        optimizer.zero_grad()
        pred = model(x, edge_index)
        loss = F.cross_entropy(pred[idx_train], y[idx_train])
        loss.backward()
        optimizer.step()

def get_perturbed_and_clean_nodes(edge_index, pert_index, num_nodes):
    original_neighbors = [set() for _ in range(num_nodes)]
    perturbed_neighbors = [set() for _ in range(num_nodes)]
    
    for i in range(edge_index.size(1)):
        node_u, node_v = edge_index[:, i]
        original_neighbors[node_u].add(int(node_v))
        original_neighbors[node_v].add(int(node_u))
    
    for i in range(pert_index.size(1)):
        node_u, node_v = pert_index[:, i]
        perturbed_neighbors[node_u].add(int(node_v))
        perturbed_neighbors[node_v].add(int(node_u))
    
    pert_idx = []
    clean_idx = []
    for i in range(num_nodes):
        if original_neighbors[i] != perturbed_neighbors[i]:
            pert_idx.append(i)
        else:
            clean_idx.append(i)
    # pert_idx = torch.tensor(pert_idx, dtype=torch.long)
    # clean_idx = torch.tensor(clean_idx, dtype=torch.long)
    return pert_idx, clean_idx

def compute_node_disturbance(pert_edge_index, edge_index, num_nodes):
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
    pert_adj = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
    
    for u, v in edge_index.t().tolist():
        adj[u, v] = True
        adj[v, u] = True

    for u, v in pert_edge_index.t().tolist():
        pert_adj[u, v] = True
        pert_adj[v, u] = True
    
    disturbance_count = torch.zeros(num_nodes, dtype=torch.int)
    
    for node in range(num_nodes):
        original_neighbors = adj[node].nonzero(as_tuple=True)[0]
        perturbed_neighbors = pert_adj[node].nonzero(as_tuple=True)[0]
        
        exclusive_edges = len(set(perturbed_neighbors.tolist()) - set(original_neighbors.tolist())) + \
                          len(set(original_neighbors.tolist()) - set(perturbed_neighbors.tolist()))
        disturbance_count[node] = exclusive_edges
    
    sorted_indices = torch.argsort(disturbance_count, descending=True)
    return sorted_indices, disturbance_count

def lid_estimate_mle(feat, k):
    num_nodes = feat.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(feat)
    distances, indices = nbrs.kneighbors(feat)
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    node_lids = torch.zeros(num_nodes)
    for node in range(num_nodes):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            try:
                lid = - k / np.sum(np.log(distances[node]/distances[node][-1]))
            except:
                # print(distances[node])
                # exit()
                continue
            
            node_lids[node] = lid
    return node_lids

import os,sys
import warnings
import numpy as np
import torch_geometric.utils as pygutils
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
sys.path.append(os.path.abspath('./'))
from parameter import parameter_parser

def load_data(args):
    data = torch.load(f'./data/{args.name}/{args.attack}.pth', map_location=args.device)
    if args.attack == 'nettack':
        x, edge_index, pert_edge_index, y, idx_train, idx_val, idx_test = \
            data['x'], data['edge_index'], data['pert_edge_index_list'], data['y'], data['idx_train'], data['idx_val'], data['idx_target']
    else:
        x, edge_index, pert_edge_index, y, idx_train, idx_val, idx_test = \
            data['x'], data['edge_index'], data['pert_edge_index'], data['y'], data['idx_train'], data['idx_val'], data['idx_test']
    return x, edge_index, pert_edge_index, y, idx_train, idx_val, idx_test



def main(config):
    x, edge_index, pert_edge_index, y, idx_train, idx_val, idx_test = load_data(config)
    k = config.k
    log_file = config.log_file
    num_features, num_classes = x.shape[1], len(torch.unique(y))

    save_gcn_path = f'./checkpoint/classifier/{config.name}/GCN.pth'
    gcn = GCN(num_features, 16, num_classes)
    gcn.load_state_dict(torch.load(save_gcn_path, map_location=config.device))
    
    # ======================================================================================================
    feat = gcn.get_features(x, edge_index).detach().cpu().numpy()
    num_nodes = x.size(0)
    
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(feat)
    distances, indices = nbrs.kneighbors(feat)
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    node_lids = torch.zeros(num_nodes)
    for node in range(num_nodes):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            try:
                lid = - k / np.sum(np.log(distances[node]/distances[node][-1]))
            except:
                continue
            if np.isnan(lid):
                print(np.sum(np.log(distances[node]/distances[node][-1])))
                print('Nan exists')
                exit()
            node_lids[node] = lid

    pert_idx, clean_idx = get_perturbed_and_clean_nodes(edge_index, pert_edge_index, x.size(0))
    sorted_indices, disturbance_count = compute_node_disturbance(pert_edge_index, edge_index, num_nodes=x.size(0))

    lids_pert = node_lids[pert_idx]
    lids_pert_nonzero = lids_pert[lids_pert != 0]

    lids_clean = node_lids[clean_idx]
    lids_clean_non_zero = lids_clean[lids_clean != 0]

    pert_slid = lids_pert_nonzero.mean().item()
    clean_slid = lids_clean_non_zero.mean().item()
    print(pert_slid, clean_slid)
    # ======================================================================================================
    with open(log_file, 'a+') as f:
        f.write(f'Before attack: pert node lid {pert_slid:.4f} clean node lid {clean_slid:.4f}\n')

    # ======================================================================================================
    feat = gcn.get_features(x, pert_edge_index).detach().cpu().numpy()
    num_nodes = x.size(0)
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(feat)
    distances, indices = nbrs.kneighbors(feat)
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    node_lids = torch.zeros(num_nodes)
    for node in range(num_nodes):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            try:
                lid = - k / np.sum(np.log(distances[node]/distances[node][-1]))
            except:
                # print(distances[node])
                # exit()
                continue
            
            node_lids[node] = lid
    
    pert_idx, clean_idx = get_perturbed_and_clean_nodes(edge_index, pert_edge_index, x.size(0))
    sorted_indices, disturbance_count = compute_node_disturbance(pert_edge_index, edge_index, num_nodes=x.size(0))

    lids_pert = node_lids[pert_idx]
    lids_pert_nonzero = lids_pert[lids_pert != 0]

    lids_clean = node_lids[clean_idx]
    lids_clean_non_zero = lids_clean[lids_clean != 0]

    # pert_slid = lids_pert_nonzero.mean().item()
    # clean_slid = lids_clean_non_zero.mean().item()
    # print(pert_slid, clean_slid)


    num = 10
    top_pert = np.partition(lids_pert_nonzero, -num)[-num:]
    bottom_clean = np.partition(lids_clean_non_zero, num)[:num]
    pert_slid_top = np.mean(top_pert)
    clean_slid_bottom = np.mean(bottom_clean)
    print(pert_slid, clean_slid)
    # ======================================================================================================

    with open(log_file, 'a+') as f:
        f.write(f'After  attack: pert node lid {pert_slid:.4f} clean node lid {clean_slid:.4f}\n')
    