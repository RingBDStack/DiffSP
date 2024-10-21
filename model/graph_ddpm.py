import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from model.gnn_ddpm import GNNAsymm_WithX, GNNAsymm_WoutX
from utils.ddpm_utils import *

class LossX(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, true_X, logit_X):
        true_X = true_X.transpose(0, 1)               # (|V|, F, C)
        true_X = true_X.reshape(-1, true_X.size(-1))  # (|V| * F, C)
        
        logit_X = logit_X.reshape(true_X.size(0), -1) # (|V| * F, C)

        true_X = torch.argmax(true_X, dim=-1)         # (|V| * F)

        loss_X = F.cross_entropy(logit_X, true_X)

        return loss_X

class LossE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, true_E, logit_E):
        true_E = torch.argmax(true_E, dim=-1)    # (B)
        loss_E = F.cross_entropy(logit_E, true_E)

        return loss_E

class LossY_WithX(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, classifier, logit_X, logit_E, Y, src, dst):
        # X
        logit_X = logit_X.softmax(dim=-1)
        logit_X = torch.transpose(logit_X, 0, 1)
        X = sample_X(logit_X)
        # E
        num_nodes = logit_X.size(1)
        logit_E = logit_E.softmax(dim=-1)
        E_ = logit_E.multinomial(1).squeeze(-1)
        E = torch.zeros(num_nodes, num_nodes).long().to(E_.device)
        E[dst, src] = E_
        E[src, dst] = E_
        E = E + torch.eye(num_nodes, dtype=torch.long, device=E_.device)
        E_index = E.nonzero(as_tuple=False).t()
        # Y
        Y = Y
        
        logit_Y = classifier(X, E_index)
        loss_Y = F.cross_entropy(logit_Y, Y)

        return loss_Y

class LossY_WoutX(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, classifier, X, logit_E, Y, src, dst):
        # E
        num_nodes = X.size(0)
        logit_E = logit_E.softmax(dim=-1)
        E_ = logit_E.multinomial(1).squeeze(-1)
        E = torch.zeros(num_nodes, num_nodes).long().to(E_.device)
        E[dst, src] = E_
        E[src, dst] = E_
        E = E + torch.eye(num_nodes, dtype=torch.long, device=E_.device)
        E_index = E.nonzero(as_tuple=False).t()
        # Y
        Y = Y
        
        logit_Y = classifier(X, E_index)
        loss_Y = F.cross_entropy(logit_Y, Y)

        return loss_Y



class NoiseSchedule(nn.Module):
    def __init__(self, T, device, s=0.008):
        super().__init__()

        # Cosine schedule as proposed in
        # https://arxiv.org/abs/2102.09672
        num_steps = T + 2
        t = np.linspace(0, num_steps, num_steps)
        # Schedule for \bar{alpha}_t = alpha_1 * ... * alpha_t
        alpha_bars = np.cos(0.5 * np.pi * ((t / num_steps) + s) / (1 + s)) ** 2
        # Make the largest value 1.
        alpha_bars = alpha_bars / alpha_bars[0]
        alphas = alpha_bars[1:] / alpha_bars[:-1]

        self.betas = torch.from_numpy(1 - alphas).float().to(device)
        self.alphas = 1 - torch.clamp(self.betas, min=0, max=0.9999)

        log_alphas = torch.log(self.alphas)
        log_alpha_bars = torch.cumsum(log_alphas, dim=0)
        self.alpha_bars = torch.exp(log_alpha_bars)

        self.betas = nn.Parameter(self.betas, requires_grad=False)
        self.alphas = nn.Parameter(self.alphas, requires_grad=False)
        self.alpha_bars = nn.Parameter(self.alpha_bars, requires_grad=False)


class ModelAsync_WithX(torch.nn.Module):
    def __init__(self,
                 num_attrs_X,
                 num_classes_E,
                 num_classes_X,
                 mlp_X_config,
                 gnn_E_config,
                 device):
        super().__init__()

        self.device = device

        self.graph_encoder = GNNAsymm_WithX(num_attrs_X=num_attrs_X,
                                            num_classes_E=num_classes_E,
                                            num_classes_X=num_classes_X,
                                            mlp_X_config=mlp_X_config,
                                            gnn_E_config=gnn_E_config)
        
        self.loss_X = LossX()
        self.loss_E = LossE()
        self.loss_Y = LossY_WithX()

    def get_loss(self,
                X_one_hot_3d,
                X_one_hot_2d,
                X_t_one_hot,
                A_t,
                Y,
                t_float_X,
                t_float_E,
                classifier,
                batch_src,
                batch_dst,
                batch_E_one_hot):
        
        logit_X, logit_E = self.graph_encoder(t_float_X,
                                              t_float_E,
                                              X_t_one_hot,
                                              X_one_hot_2d,
                                              A_t,
                                              batch_src,
                                              batch_dst)
        
        loss_X = self.loss_X(X_one_hot_3d, logit_X)
        loss_E = self.loss_E(batch_E_one_hot, logit_E)
        loss_Y = self.loss_Y(classifier, logit_X, logit_E, Y, batch_src, batch_dst)

        return loss_X, loss_E, loss_Y


    def forward(self,
                X_one_hot_2d,
                X_t_one_hot,
                A_t,
                t_float_X,
                t_float_E,
                batch_src,
                batch_dst,):
        
        logit_X, logit_E = self.graph_encoder(t_float_X,
                                              t_float_E,
                                              X_t_one_hot,
                                              X_one_hot_2d,
                                              A_t,
                                              batch_src,
                                              batch_dst)

        return logit_X, logit_E


class ModelAsync_WoutX(torch.nn.Module):
    def __init__(self,
                 num_attrs_X,
                 num_classes_E,
                 num_classes_X,
                 gnn_E_config,
                 device):
        super().__init__()

        self.device = device

        self.graph_encoder = GNNAsymm_WoutX(num_attrs_X=num_attrs_X,
                                            num_classes_E=num_classes_E,
                                            num_classes_X=num_classes_X,
                                            gnn_E_config=gnn_E_config)
        
        self.loss_E = LossE()
        self.loss_Y = LossY_WoutX()

    def get_loss(self,
                 X_one_hot_2d,
                 A_t,
                 Y,
                 t_float_E,
                 classifier,
                 batch_src,
                 batch_dst,
                 batch_E_one_hot):
        
        logit_E = self.graph_encoder(t_float_E,
                                     X_one_hot_2d,
                                     A_t,
                                     batch_src,
                                     batch_dst)
        
        loss_E = self.loss_E(batch_E_one_hot, logit_E)
        loss_Y = self.loss_Y(classifier, X_one_hot_2d, logit_E, Y, batch_src, batch_dst)

        return loss_E, loss_Y