import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import torch
import os.path as osp
import sys
sys.path.append(osp.abspath('./'))
from tqdm import tqdm
# from nc.utils import *
from n.model import *
from n.train import *
from parameter import parameter_parser
from torch_geometric.nn import GCNConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.optim import Adam


def preprocess(x, edge_index, device):
    # x = g.x
    # edge_index = g.edge_index
    X = x
    # X[X != 0] = 1.

    N = x.size(0)
    src, dst = edge_index

    X_one_hot_list = []
    for f in range(X.size(1)):
        X_f_one_hot = F.one_hot(X[:, f].long(), num_classes=2) # (N, 2)
        X_one_hot_list.append(X_f_one_hot)
    X_one_hot = torch.stack(X_one_hot_list, dim=0).float() # (F, N, 2)

    E = torch.zeros(N, N)
    E[dst, src] = 1.
    E_one_hot = F.one_hot(E.long()).float() # (N, N, 2)

    X_one_hot_count = X_one_hot.sum(dim=1) # (F, 2)
    X_marginal = X_one_hot_count / X_one_hot_count.sum(dim=1, keepdim=True) # (F, 2)

    # (2)
    E_one_hot_count = E_one_hot.sum(dim=0).sum(dim=0)
    E_marginal = E_one_hot_count / E_one_hot_count.sum()

    return X_one_hot.to(device), E_one_hot.to(device), X_marginal.to(device),  E_marginal.to(device)



class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.norm = gcn_norm
        self.conv1 = GCNConv(in_channels, hidden_channels, normalize=False)
        self.conv2 = GCNConv(hidden_channels, out_channels, normalize=False)

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
        return x
    
    def get_features(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        # x = self.conv2(x, edge_index, edge_weight)
        # x = self.ln1(x)
        return x

def train_classifier(model, x, edge_index, y, idx_train, edge_weight=None, epochs=200, lr=0.01, weight_decay=5e-4):
    model.train()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in tqdm(range(epochs)):
        optimizer.zero_grad()
        pred = model(x, edge_index, edge_weight)
        loss = F.cross_entropy(pred[idx_train], y[idx_train])
        loss.backward()
        optimizer.step()


def accuracy(pred, y, mask):
    return (pred.argmax(-1)[mask] == y[mask]).float().mean()

@torch.no_grad()
def test(model, x, edge_index, y, test_mask):
    model.eval()
    pred = model(x, edge_index)
    return float(accuracy(pred, y, test_mask))


def is_undirected(edge_index):
    reversed_edge_index = edge_index.flip(0)
    
    for i in range(edge_index.size(1)):
        edge = edge_index[:, i].view(2, 1)
        if not ((reversed_edge_index == edge).all(dim=0).any()):
            return False
    return True

def has_self_loops(edge_index):
    return (edge_index[0] == edge_index[1]).any()


def load_data(args):
    data = torch.load(f'./data/{args.name}/{args.attack}_evasion.pth', map_location=args.device)
    if args.attack == 'nettack':
        x, edge_index, pert_edge_index, y, idx_train, idx_val, idx_test = \
            data['x'], data['edge_index'], data['pert_edge_index_list'], data['y'], data['idx_train'], data['idx_val'], data['idx_target']
    else:
        x, edge_index, pert_edge_index, y, idx_train, idx_val, idx_test = \
            data['x'], data['edge_index'], data['pert_edge_index'], data['y'], data['idx_train'], data['idx_val'], data['idx_test']
    return x, edge_index, pert_edge_index, y, idx_train, idx_val, idx_test


# transductive
def main(config):
    # ====================config==================== #
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    # config.device = device
    # device = torch.device('cpu')
    print('Using device', device)
    ck_path = f'./checkpoint/diffusion/{config.name}/n_step_E{config.T_E}'
    
    # ====================data==================== #
    x, edge_index, pert_edge_index, y, idx_train, idx_val, idx_test = load_data(config)
    num_features, num_classes = x.shape[1], len(torch.unique(y))
    X_one_hot_3d, E_one_hot, X_marginal, E_marginal = preprocess(x, edge_index, device)

    # ====================model==================== #
    model = GraphDDPM(x.size(1), config.T_E, E_marginal, config.gnn_E, device).to(device)
    classifier = GCN(num_features, 16, num_classes).to(device)

    # ====================model==================== #
    # train_ddpm(config, model, device, x, E_one_hot, ck_path)
    train_classifier(classifier, x, edge_index, y, idx_train)

    # ====================purify==================== #

    checkpoint = torch.load(f"{ck_path}.pth", map_location=device)
    model.graph_encoder.pred_E.load_state_dict(checkpoint['pred_E_state_dict'])

    if config.attack == 'nettack':
        acc_clean = 0.
        acc_pert = 0.
        acc_purify = 0.
        for idx in range(len(idx_test)):
            idx_target = torch.tensor(idx_test[idx]).unsqueeze(0).to(device)
            pert_edge_index_target = pert_edge_index[idx]
            purify_edge_index_target = model.anisotropy_transfer_entropy_guided_purify(classifier, x, pert_edge_index_target, config.purify.basic_t, config.purify.adaptive_t, config.purify.k, config.purify.lamb, device)

            acc_clean += test(classifier, x, edge_index, y, idx_target)
            acc_pert += test(classifier, x, pert_edge_index_target, y, idx_target)
            acc_purify += test(classifier, x, purify_edge_index_target, y, idx_target)
        acc_clean /= len(idx_test)
        acc_pert /= len(idx_test)
        acc_purify /= len(idx_test)
    else:
        acc_clean = test(classifier, x, edge_index, y, idx_test)
        acc_pert = test(classifier, x, pert_edge_index, y, idx_test)
        # purify_edge_index = model.purify(x, pert_edge_index, config.purify.t_E)
        purify_edge_index = model.anisotropy_transfer_entropy_guided_purify(classifier, x, pert_edge_index, config.purify.basic_t, config.purify.adaptive_t, config.purify.k, config.purify.lamb, device)
        acc_purify = test(classifier, x, purify_edge_index, y, idx_test)
    print(f'clean {acc_clean:.4e} pert {acc_pert:.4e} purify {acc_purify:.4e}')
    
    # with open(f'./nc/{config.attack}_{config.name}_result.txt', 'a+') as f:
    #     f.write(f'clean {acc_clean:.4e} pert {acc_pert:.4e} purify {acc_purify:.4e}')
    #     f.write('\n')

import random
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

from easydict import EasyDict
if __name__ == "__main__":
    config = parameter_parser()
    for _ in range(10):
        # config = EasyDict(config)
        seed = random.randint(0, 1000)
        print(seed)
        set_seed(seed)
        set_seed(config.seed)
        main(config)