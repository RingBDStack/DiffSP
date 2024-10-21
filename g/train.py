from torch.utils.data import DataLoader
from copy import deepcopy
import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

def get_E_one_hot(g, device):
    N = g.x.size(0)
    src, dst = g.edge_index[0], g.edge_index[1]

    E = torch.zeros(N, N)
    E[dst, src] = 1.
    E_one_hot = F.one_hot(E.long()).float()                                 

    return E_one_hot.to(device)

def train_ddpm(config, model, dataset, device, ck_path):
    os.makedirs(os.path.dirname(ck_path), exist_ok=True)
    with open(ck_path +'.txt', 'a+') as f:
        f.write(str(config) + '\n')
    

    optimizer_E = torch.optim.AdamW(model.graph_encoder.pred_E.parameters(), **config.optimizer_E)
    lr_scheduler_E = ReduceLROnPlateau(optimizer_E, mode='min', **config.lr_scheduler)

    best_epoch_E = 0
    best_state_dict_E = deepcopy(model.graph_encoder.pred_E.state_dict())
    best_val_nll_E = float('inf')
    best_log_p_0_E = float('inf')
    best_denoise_match_E = float('inf')

    num_patient_epochs = 0
    for epoch in tqdm(range(config.train.num_epochs), position=0, leave=True):
        # ==================== train ====================
        model.train()
        total_loss = 0
        for g in dataset:
            E_one_hot = get_E_one_hot(g, device)
            X_one_hot_2d = g.x


            num_nodes = g.x.size(0)
            src, dst = torch.triu_indices(num_nodes, num_nodes, offset=1, device=device)
            loss_X = model.log_p_t(E_one_hot,
                                    X_one_hot_2d,
                                    src,
                                    dst,
                                    E_one_hot[src, dst])
            loss = loss_X
            total_loss += loss

            optimizer_E.zero_grad()

            loss.backward()

            nn.utils.clip_grad_norm_(model.graph_encoder.pred_E.parameters(), config.train.max_grad_norm)

            optimizer_E.step()
        tqdm.write(f'loss {total_loss / len(dataset) :.4f}')
        if (epoch + 1) % config.train.val_every_epochs != 0:
            continue

        model.eval()
        num_patient_epochs += 1
        denoise_match_E = []
        log_p_0_E = []
        for g in dataset:
            E_one_hot = get_E_one_hot(g, device)
            X_one_hot_2d = g.x
            num_nodes = g.x.size(0)
            src, dst = torch.triu_indices(num_nodes, num_nodes, offset=1, device=device)

            batch_denoise_match_E, \
                batch_log_p_0_E = model.val_step(
                    E_one_hot,
                    X_one_hot_2d,
                    src,
                    dst,
                    E_one_hot[src, dst])

            denoise_match_E.append(batch_denoise_match_E)
            log_p_0_E.append(batch_log_p_0_E)

        denoise_match_E = np.mean(denoise_match_E)
        log_p_0_E = np.mean(log_p_0_E)

        val_E = denoise_match_E + log_p_0_E

        to_save_cpt = False
        if val_E < best_val_nll_E:
            best_val_nll_E = val_E
            best_epoch_E = epoch
            best_state_dict_E = deepcopy(model.graph_encoder.pred_E.state_dict())
            to_save_cpt = True

        if to_save_cpt:
            torch.save({
                "config": config,
                "best_epoch_E": best_epoch_E,
                "pred_E_state_dict": best_state_dict_E
            }, ck_path + '.pth')
            os.makedirs(os.path.dirname(ck_path), exist_ok=True)
            print(f'model saved {ck_path}.pth')

        if log_p_0_E < best_log_p_0_E:
            best_log_p_0_E = log_p_0_E
            num_patient_epochs = 0

        if denoise_match_E < best_denoise_match_E:
            best_denoise_match_E = denoise_match_E
            num_patient_epochs = 0
        
        print("Epoch {} | best val E {:.7f} | val E {:.7f} | patience {}/{} | E lr {}".format(
            epoch + 1, best_val_nll_E, val_E, num_patient_epochs, config.train.patient_epochs, optimizer_E.param_groups[0]['lr']))
        with open(ck_path +'.txt', 'a+') as f:
            f.write("Epoch {} | best val E {:.7f} | val X {:.7f} | patience {}/{} | E lr{}\n".format(
            epoch + 1, best_val_nll_E, val_E, num_patient_epochs, config.train.patient_epochs, optimizer_E.param_groups[0]['lr']))

        if num_patient_epochs == config.train.patient_epochs:
            break
        
        lr_scheduler_E.step(log_p_0_E)
