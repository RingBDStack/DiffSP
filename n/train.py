
from torch.utils.data import DataLoader
from copy import deepcopy
import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

def train_ddpm(config, model, device, X_one_hot_2d, E_one_hot, ck_path):
    # X_one_hot_3d = X_one_hot_3d.to(device) # (F, |V|, 2)
    # X_one_hot_2d = torch.transpose(X_one_hot_3d, 0, 1) # (|V|, F, 2)
    # X_one_hot_2d = X_one_hot_2d.reshape(X_one_hot_2d.size(0), -1) # (|V|, 2 * F)
    
    N = X_one_hot_2d.size(0)
    src, dst = torch.triu_indices(N, N, offset=1, device=device)
    edge_index = torch.stack([dst, src], dim=1)
    data_loader = DataLoader(edge_index.cpu(), batch_size=config.data.batch_size, shuffle=True, num_workers=4)
    # val_data_loader = DataLoader(edge_index, batch_size=config.data.val_batch_size, shuffle=False)

    optimizer_E = torch.optim.AdamW(model.graph_encoder.pred_E.parameters(), **config.optimizer_E)
    lr_scheduler_E = ReduceLROnPlateau(optimizer_E, mode='min', **config.lr_scheduler)

    best_epoch_E = 0
    best_state_dict_E = deepcopy(model.graph_encoder.pred_E.state_dict())
    best_val_nll_E = float('inf')
    best_log_p_0_E = float('inf')
    best_denoise_match_E = float('inf')

    num_patient_epochs = 0
    for epoch in tqdm(range(config.train.num_epochs)):
        model.train()

        for batch_edge_index in tqdm(data_loader):
            batch_edge_index = batch_edge_index.to(device)
            batch_dst, batch_src = batch_edge_index.T
            batch_src, batch_dst = src, dst
            loss_E = model.log_p_t(E_one_hot,
                                    X_one_hot_2d,
                                    batch_src,
                                    batch_dst,
                                    E_one_hot[batch_src, batch_dst])
            loss = loss_E
            optimizer_E.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.graph_encoder.pred_E.parameters(), config.train.max_grad_norm)
            optimizer_E.step()
        
        if (epoch + 1) % config.train.val_every_epochs != 0:
            continue

        model.eval()
        num_patient_epochs += 1
        denoise_match_E = []
        log_p_0_E = []
        # for batch_edge_index in tqdm(val_data_loader):
        #     batch_dst, batch_src = batch_edge_index.T
        batch_denoise_match_E, batch_log_p_0_E = model.val_step(
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
            os.makedirs(os.path.dirname(ck_path), exist_ok=True)
            torch.save({
                "config": config,
                "best_epoch_E": best_epoch_E,
                "pred_E_state_dict": best_state_dict_E
            }, ck_path + '.pth')
            print(f'model saved {ck_path}.pth')

        if log_p_0_E < best_log_p_0_E:
            best_log_p_0_E = log_p_0_E
            num_patient_epochs = 0

        if denoise_match_E < best_denoise_match_E:
            best_denoise_match_E = denoise_match_E
            num_patient_epochs = 0
        
        print("Epoch {} | {} | best val E {:.7f} | val E {:.7f} | patience {}/{} | E lr {}".format(epoch + 1, config.name, best_val_nll_E, val_E, num_patient_epochs, config.train.patient_epochs, optimizer_E.param_groups[0]['lr']))
        with open(ck_path +'.txt', 'a+') as f:
            f.write("Epoch {} | best val E {:.7f} | val E {:.7f} | patience {}/{} | E lr{}\n".format(epoch + 1, best_val_nll_E, val_E, num_patient_epochs, config.train.patient_epochs, optimizer_E.param_groups[0]['lr']))

        if num_patient_epochs == config.train.patient_epochs:
            break
        
        lr_scheduler_E.step(log_p_0_E)
