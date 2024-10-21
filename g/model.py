from utils.mle_lids import lid_estimate_mle
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from g.layer import GNNAsymm
import dgl.sparse as dglsp
from tqdm import tqdm
from torch.utils.data import DataLoader
from copy import deepcopy
from utils.entropy import transfer_entropy

class LossE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, true_E, logit_E):
        true_E = torch.argmax(true_E, dim=-1)    # (B)
        loss_E = F.cross_entropy(logit_E, true_E)

        return loss_E

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

class MarginalTransition(nn.Module):
    def __init__(self,
                 device,
                 E_marginal,
                 num_classes_E):
        super().__init__()

        # (2, 2)
        self.I_E = torch.eye(num_classes_E, device=device)

        # (2, 2)
        self.m_E = E_marginal.unsqueeze(0).expand(num_classes_E, -1).clone()

        self.I_E = nn.Parameter(self.I_E, requires_grad=False)

        self.m_E = nn.Parameter(self.m_E, requires_grad=False)

    def get_Q_bar_E(self, alpha_bar_t):
        Q_bar_t_E = alpha_bar_t * self.I_E + (1 - alpha_bar_t) * self.m_E
        return Q_bar_t_E

class GraphDDPM(nn.Module):
    def __init__(
            self,
            in_X,
            T_E,
            E_marginal,
            gnn_E_config,
            device
    ):
        super().__init__()

        self.device = device

        self.T_E = T_E
        self.num_classes_E = 2

        self.E_marginal = E_marginal
        
        self.transition = MarginalTransition(device,E_marginal, self.num_classes_E)
        self.noise_schedule_E = NoiseSchedule(T_E, device)

        self.graph_encoder = GNNAsymm(in_X=in_X,
                                      num_classes_E=self.num_classes_E,
                                      gnn_E_config=gnn_E_config)

        self.loss_E = LossE()
    

    def sample_E(self, prob_E):
        # (|V|^2, 1)
        E_t = prob_E.reshape(-1, prob_E.size(-1)).multinomial(1)

        # (|V|, |V|)
        num_nodes = prob_E.size(0)
        E_t = E_t.reshape(num_nodes, num_nodes)
        # Make it symmetric for undirected graphs.
        src, dst = torch.triu_indices(
            num_nodes, num_nodes, device=E_t.device)
        E_t[dst, src] = E_t[src, dst]

        return E_t

    def apply_noise_E(self, E_one_hot, t=None):
        if t is None:
            # Sample a timestep t uniformly.
            # Note that the notation is slightly inconsistent with the paper.
            # t=0 corresponds to t=1 in the paper, where corruption has already taken place.
            t = torch.randint(low=0, high=self.T_E + 1, size=(1,), device=E_one_hot.device)

        alpha_bar_t = self.noise_schedule_E.alpha_bars[t]
        Q_bar_t_E = self.transition.get_Q_bar_E(alpha_bar_t) # (2, 2)
        prob_E = E_one_hot @ Q_bar_t_E                       # (|V|, |V|, 2)
        E_t = self.sample_E(prob_E)
        t_float_E = t / self.T_E

        return t_float_E, E_t

    def get_adj(self, E_t):
        # Row normalization.
        edges_t = E_t.nonzero().T
        num_nodes = E_t.size(0)
        A_t = dglsp.spmatrix(edges_t, shape=(num_nodes, num_nodes))
        D_t = dglsp.diag(A_t.sum(1)) ** -1
        return D_t @ A_t

    def log_p_t(self,
                E_one_hot,
                X_one_hot_2d,
                batch_src,
                batch_dst,
                batch_E_one_hot,
                t_E=None):
        
        t_float_E, E_t = self.apply_noise_E(E_one_hot, t_E)
        A_t = self.get_adj(E_t)
        logit_E = self.graph_encoder(t_float_E,
                                              X_one_hot_2d,
                                              A_t,
                                              batch_src,
                                              batch_dst)
        
        loss_E = self.loss_E(batch_E_one_hot, logit_E)

        return loss_E

    def posterior(self,
                  Z_t,
                  Q_t,
                  Q_bar_s,
                  Q_bar_t,
                  prior):
        # (B, 2) or (F, |V|, 2)
        left_term = Z_t @ torch.transpose(Q_t, -1, -2)
        # (B, 1, 2) or (F, |V|, 1, 2)
        left_term = left_term.unsqueeze(dim=-2)
        # (1, 2, 2) or (F, 1, 2, 2)
        right_term = Q_bar_s.unsqueeze(dim=-3)
        # (B, 2, 2) or (F, |V|, 2, 2)
        # Different from denoise_match_z, this function does not
        # compute (Z_t @ Q_t.T) * (Z_0 @ Q_bar_s) for a specific
        # Z_0, but compute for all possible values of Z_0.
        numerator = left_term * right_term

        # (2, B) or (F, 2, |V|)
        prod = Q_bar_t @ torch.transpose(Z_t, -1, -2)
        # (B, 2) or (F, |V|, 2)
        prod = torch.transpose(prod, -1, -2)
        # (B, 2, 1) or (F, |V|, 2, 1)
        denominator = prod.unsqueeze(-1)
        denominator[denominator == 0.] = 1.
        # (B, 2, 2) or (F, |V|, 2, 2)
        out = numerator / denominator

        # (B, 2, 2) or (F, |V|, 2, 2)
        prob = prior.unsqueeze(-1) * out
        # (B, 2) or (F, |V|, C)
        prob = prob.sum(dim=-2)

        return prob
    
    def denoise_match_Z(self,
                        Z_t_one_hot,
                        Q_t_Z,
                        Z_one_hot,
                        Q_bar_s_Z,
                        pred_Z):

        # q(Z^{t-1}| Z, Z^t)
        left_term = Z_t_one_hot @ torch.transpose(Q_t_Z, -1, -2) # (B, C) or (A, B, C)
        right_term = Z_one_hot @ Q_bar_s_Z                       # (B, C) or (A, B, C)
        product = left_term * right_term                         # (B, C) or (A, B, C)
        denom = product.sum(dim=-1)                              # (B,) or (A, B)
        denom[denom == 0.] = 1
        prob_true = product / denom.unsqueeze(-1)                # (B, C) or (A, B, C)

        # q(Z^{t-1}| hat{p}^{Z}, Z^t)
        right_term = pred_Z @ Q_bar_s_Z                          # (B, C) or (A, B, C)
        product = left_term * right_term                         # (B, C) or (A, B, C)
        denom = product.sum(dim=-1)                              # (B,) or (A, B)
        denom[denom == 0.] = 1
        prob_pred = product / denom.unsqueeze(-1)                # (B, C) or (A, B, C)

        epsilon = 1e-10
        prob_true = torch.clamp(prob_true, min=epsilon, max=1 - epsilon)
        prob_pred = torch.clamp(prob_pred, min=epsilon, max=1 - epsilon)

        # KL(q(Z^{t-1}| hat{p}^{Z}, Z^t) || q(Z^{t-1}| Z, Z^t))
        kl = F.kl_div(input=prob_pred.log(), target=prob_true, reduction='none')
        return kl.clamp(min=0).mean().item()

    def denoise_match_E(self,
                        t_float,
                        logit_E,
                        E_t_one_hot,
                        E_one_hot):
        t = int(t_float.item() * self.T_E)
        s = t - 1

        alpha_bar_s = self.noise_schedule_E.alpha_bars[s]
        alpha_t = self.noise_schedule_E.alphas[t]

        Q_bar_s_E = self.transition.get_Q_bar_E(alpha_bar_s)
        # Note that computing Q_bar_t from alpha_bar_t is the same
        # as computing Q_t from alpha_t.
        Q_t_E = self.transition.get_Q_bar_E(alpha_t)

        pred_E = logit_E.softmax(dim=-1)

        return self.denoise_match_Z(E_t_one_hot,
                                    Q_t_E,
                                    E_one_hot,
                                    Q_bar_s_E,
                                    pred_E)

    def val_step(self,
                 E_one_hot,
                 X_one_hot_2d,
                 batch_src,
                 batch_dst,
                 batch_E_one_hot):
        
        device = self.device

        # Case2: E
        denoise_match_E = []
        # t=0 is handled separately.
        for t_sample_E in range(1, self.T_E + 1):
            t_E = torch.LongTensor([t_sample_E]).to(device)
            t_float_E, E_t = self.apply_noise_E(E_one_hot, t_E)
            A_t = self.get_adj(E_t)
            logit_E = self.graph_encoder.pred_E(t_float_E,
                                                X_one_hot_2d,
                                                A_t,
                                                batch_src,
                                                batch_dst)

            E_t_one_hot = F.one_hot(E_t[batch_src, batch_dst],
                                    num_classes=self.num_classes_E).float()
            denoise_match_E_t = self.denoise_match_E(t_float_E,
                                                     logit_E,
                                                     E_t_one_hot,
                                                     batch_E_one_hot)
            denoise_match_E.append(denoise_match_E_t)
        denoise_match_E = float(np.mean(denoise_match_E)) * self.T_E

        # t=0
        t_0 = torch.LongTensor([0]).to(device)
        loss_E = self.log_p_t(E_one_hot,
                                X_one_hot_2d,
                                batch_src,
                                batch_dst,
                                batch_E_one_hot,
                                t_E=t_0)
        log_p_0_E = loss_E.item()

        return denoise_match_E, \
            log_p_0_E

    
    def sample_E_infer(self, prob_E, num_nodes, src, dst):
        E_t_ = prob_E.multinomial(1).squeeze(-1)
        E_t = torch.zeros(num_nodes, num_nodes).long().to(E_t_.device)
        E_t[dst, src] = E_t_
        E_t[src, dst] = E_t_

        return E_t

    def get_E_t(self,
                device,
                batch_src,
                batch_dst,
                pred_E_func,
                t_float,
                X_t_one_hot,
                E_t,
                Q_t_E,
                Q_bar_s_E,
                Q_bar_t_E):
        A_t = self.get_adj(E_t)
        E_prob = torch.zeros(len(self.src), self.num_classes_E).to(device)

        batch_pred_E = pred_E_func(t_float,
                                    X_t_one_hot,
                                    A_t,
                                    batch_src,
                                    batch_dst)

        batch_pred_E = batch_pred_E.softmax(dim=-1)

        # (B, 2)
        batch_E_t_one_hot = F.one_hot(
            E_t[batch_src, batch_dst],
            num_classes=self.num_classes_E).float()
        batch_E_prob_ = self.posterior(batch_E_t_one_hot, Q_t_E,
                                        Q_bar_s_E, Q_bar_t_E, batch_pred_E)

            # end = start + batch_size
            # E_prob[start: end] = batch_E_prob_
            # start = end
        E_prob = batch_E_prob_

        E_t = self.sample_E_infer(E_prob)
        return A_t, E_t
    
    def sample(self, X_t_one_hot, E_t, RT_E, batch_size=327680):
        device = self.device
        self.num_nodes = X_t_one_hot.size(0)


        src, dst = torch.triu_indices(self.num_nodes, self.num_nodes,
                                      offset=1, device=device)
        # (|E|)
        self.dst = dst
        # (|E|)
        self.src = src
        # (|E|, 2)
        edge_index = torch.stack([src, dst], dim=1).to("cpu")
        data_loader = DataLoader(edge_index, batch_size=batch_size)

        # Iteratively sample p(A^s | A^t) for t = 1, ..., T_E, with s = t - 1.
        for s_E in tqdm(list(reversed(range(0, RT_E)))):
            t_E = s_E + 1
            alpha_t_E = self.noise_schedule_E.alphas[t_E]
            alpha_bar_s_E = self.noise_schedule_E.alpha_bars[s_E]
            alpha_bar_t_E = self.noise_schedule_E.alpha_bars[t_E]

            Q_t_E = self.transition.get_Q_bar_E(alpha_t_E)
            Q_bar_s_E = self.transition.get_Q_bar_E(alpha_bar_s_E)
            Q_bar_t_E = self.transition.get_Q_bar_E(alpha_bar_t_E)

            t_float_E = torch.tensor([t_E / self.T_E]).to(device)

            _, E_t = self.get_E_t(device,
                                #   data_loader,
                                  src,
                                  dst,
                                  self.graph_encoder.pred_E,
                                  t_float_E,
                                  X_t_one_hot,
                                  E_t,
                                  Q_t_E,
                                  Q_bar_s_E,
                                  Q_bar_t_E,
                                  batch_size)


        edge_index = E_t.nonzero(as_tuple=False).t()
        return edge_index

    def purify(self, dataset, RT_E):
        purify_dataset = []
        for g in dataset:
            x = g.x; edge_index = g.edge_index
            N = x.size(0)
            src, dst = edge_index.to(self.device)
            E = torch.zeros(N, N)
            E[dst, src] = 1.
            E_one_hot = F.one_hot(E.long(), num_classes=2).float().to(self.device) # (N, N, 2)
            _, E_t = self.apply_noise_E(E_one_hot, RT_E)
            purify_edge_index = self.sample(x, E_t, RT_E)
            purify_g = deepcopy(g)
            purify_g.edge_index = purify_edge_index
            purify_dataset.append(purify_g)
        return purify_dataset
    

    def guide_scale(self, t, budget, lamb=100):
        alpha_bar_t = self.noise_schedule_E.alphas[t]
        scale = budget / alpha_bar_t
        scale *= lamb
        return scale    
    
    def anisotropy_transfer_entropy_guided_purify(self, classifier, dataset, basic_t, adaptive_t, k, lamb, device):
        ### Time
        max_t = basic_t + adaptive_t
        classifier.eval()

        ### Purify
        purify_dataset = []
        for g in tqdm(dataset):
            x, pert_edge_index = g.x, g.edge_index
            num_nodes = x.size(0)

            ### LID based anomoly score
            feat = classifier.get_features(x, pert_edge_index)
            lids = lid_estimate_mle(feat.cpu().detach().numpy(), k)
            adj_anomaly_score = (lids.unsqueeze(0) + lids.unsqueeze(1)) / 2
            adj_anomaly_score.fill_diagonal_(0)
            min_val = adj_anomaly_score.min()
            max_val = adj_anomaly_score.max()
            adj_anomaly_score = (adj_anomaly_score - min_val) / (max_val - min_val)
            adj_time = torch.full((num_nodes, num_nodes), basic_t) + adaptive_t * adj_anomaly_score
            adj_time = torch.round(adj_time)

            ### Preprocess
            pert_src, pert_dst = pert_edge_index
            E = torch.zeros(num_nodes, num_nodes)
            E[pert_src, pert_dst] = 1.
            E_one_hot = F.one_hot(E.long()).float().to(self.device)
            alpha_bar_t = self.noise_schedule_E.alpha_bars[max_t]
            Q_bar_t_E = self.transition.get_Q_bar_E(alpha_bar_t)
            prob_E_t = E_one_hot @ Q_bar_t_E

            src, dst = torch.triu_indices(num_nodes, num_nodes, offset=1, device=device)

            E_t = self.sample_E(prob_E_t)
            A_adv = torch.zeros(num_nodes, num_nodes, dtype=torch.float).to(device)
            A_adv[pert_edge_index[0]][pert_edge_index[1]] = 1.
            feat = A_adv @ x

            ### Diffusion
            for s_E in list(reversed(range(0, max_t))):
                t_E = s_E + 1
                t_float_E = torch.tensor([t_E / self.T_E]).to(device)

                alpha_t_E = self.noise_schedule_E.alphas[t_E]
                alpha_bar_s_E = self.noise_schedule_E.alpha_bars[s_E]
                alpha_bar_t_E = self.noise_schedule_E.alpha_bars[t_E]

                Q_t_E = self.transition.get_Q_bar_E(alpha_t_E)
                Q_bar_s_E = self.transition.get_Q_bar_E(alpha_bar_s_E)
                Q_bar_t_E = self.transition.get_Q_bar_E(alpha_bar_t_E)

                ### Get E_s
                A_t = self.get_adj(E_t)
                pred_E = self.graph_encoder.pred_E(t_float_E, x, A_t, src, dst)
                pred_E = pred_E.softmax(dim=-1)
                E_t_one_hot = F.one_hot(E_t[src, dst], num_classes=self.num_classes_E).float()

                prob_E_s = self.posterior(E_t_one_hot, Q_t_E, Q_bar_s_E, Q_bar_t_E, pred_E)


                ### Transfer entropy guided
                if len(prob_E_t.shape) == 3: prob_E_t = prob_E_t[src, dst]
                
                A_s = torch.zeros(num_nodes, num_nodes).to(device)
                A_t = torch.zeros(num_nodes, num_nodes).to(device)
                A_s[src, dst] = A_s[dst, src] = prob_E_s[:, 1]
                A_t[src, dst] = A_t[dst, src] = prob_E_t[:, 1]

                feat_s = A_s @ x
                feat_t = A_t @ x

                en = -transfer_entropy(feat, feat_t, feat_s, kernel=True)
                grad = torch.autograd.grad(en, prob_E_s)[0]
                scale = self.guide_scale(s_E, budget=0.1, lamb=lamb)

                prob_E_s -= scale * grad
                prob_E_s = torch.relu(prob_E_s)
                prob_E_s = prob_E_s / prob_E_s.sum(dim=-1, keepdim=True)


                #### Non-isotropic
                mask = (adj_time >= t_E).int().to(device)
                
                prob_E_s_forward = E_one_hot @ Q_bar_s_E
                prob_E_s = mask[src, dst, None] * prob_E_s + (1 - mask)[src, dst, None] * prob_E_s_forward[src, dst]


                ### update
                E_s = self.sample_E_infer(prob_E_s, num_nodes, src, dst)
                E_t = E_s
                prob_E_t = prob_E_s
            
            ### Save
            purify_edge_index = E_t.nonzero(as_tuple=False).t().to(device)
            purify_g = deepcopy(g)
            purify_g.edge_index = purify_edge_index
            purify_dataset.append(purify_g)
        
        return purify_dataset
    