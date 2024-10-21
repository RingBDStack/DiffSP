import os
import time
import pickle
import math
import torch
import sys
sys.path.append("/home/LAB/luojy/work-main/Diffusion-Robust/")

import numpy as np
from tqdm import trange
from model.sde import VPSDE, VESDE, subVPSDE
from model.anomaly_score import get_batch_link_anomaly_score
from utils.graph import mask_x, mask_adjs, gen_noise
from utils.entropy import transfer_entropy

def guide_scale_back(config, t, sde_x, sde_adj):
  # scale = diffusion.compute_scale(x_in,t, config.attack.ptb*2/255. / 3. / config.purification.guide_scale)
  # def compute_scale(self,x, t, m):
  #   alpha_bar = self.alphas_cumprod[t]
  #   return np.sqrt(1-alpha_bar) / (m*np.sqrt(alpha_bar))
  budget = config.attack.budget

  adj_log_mean_coeff = -0.25 * t ** 2 * (sde_adj.beta_1 - sde_adj.beta_0) - 0.5 * t * sde_adj.beta_0
  adj_attack_scale = torch.exp(adj_log_mean_coeff) * budget
  adj_noise_scale = torch.sqrt(1. - torch.exp(2. * adj_log_mean_coeff))
  adj_coefficient = adj_noise_scale / adj_attack_scale
  adj_scale = config.pure.guide_rate * adj_coefficient

  x_log_mean_coeff = -0.25 * t ** 2 * (sde_x.beta_1 - sde_x.beta_0) - 0.5 * t * sde_x.beta_0
  x_coefficient = torch.sqrt(1. - torch.exp(2. * x_log_mean_coeff))
  x_scale = config.pure.guide_rate * x_coefficient
  # print(adj_scale, x_scale)
  # print(adj_noise_scale, adj_attack_scale)
  # exit()
  return x_scale, adj_scale

def guide_scale(config, t, sde_adj):
  # scale = diffusion.compute_scale(x_in,t, config.attack.ptb*2/255. / 3. / config.purification.guide_scale)
  # def compute_scale(self,x, t, m):
  #   alpha_bar = self.alphas_cumprod[t]
  #   return np.sqrt(1-alpha_bar) / (m*np.sqrt(alpha_bar))
  budget = config.attack.budget

  adj_log_mean_coeff = -0.25 * t ** 2 * (sde_adj.beta_1 - sde_adj.beta_0) - 0.5 * t * sde_adj.beta_0
  adj_attack_scale = torch.exp(adj_log_mean_coeff) * budget
  adj_noise_scale = torch.sqrt(1. - torch.exp(2. * adj_log_mean_coeff))
  adj_coefficient = adj_noise_scale / adj_attack_scale
  adj_scale = config.pure.guide_rate * adj_coefficient

  # x_log_mean_coeff = -0.25 * t ** 2 * (sde_x.beta_1 - sde_x.beta_0) - 0.5 * t * sde_x.beta_0
  # x_coefficient = torch.sqrt(1. - torch.exp(2. * x_log_mean_coeff))
  # x_scale = config.pure.guide_rate * x_coefficient
  # print(adj_scale, x_scale)
  # print(adj_noise_scale, adj_attack_scale)
  # exit()
  return adj_scale

def get_score_fn(sde, model, train=True, continuous=True):

  if not train:
    model.eval()
  model_fn = model

  if isinstance(sde, VPSDE) or isinstance(sde, subVPSDE):
    def score_fn(x, adj, flags, t):
      # Scale neural network output by standard deviation and flip sign
      if continuous:
        score = model_fn(x, adj, flags)
        std = sde.marginal_prob(torch.zeros_like(adj), t)[1]
      else:
        raise NotImplementedError(f"Discrete not supported")
      score = -score / std[:, None, None]
      return score

  elif isinstance(sde, VESDE):
    def score_fn(x, adj, flags, t):
      if continuous:
        score = model_fn(x, adj, flags)
      else:  
        raise NotImplementedError(f"Discrete not supported")
      return score

  else:
    raise NotImplementedError(f"SDE class {sde.__class__.__name__} not supported.")

  return score_fn


def diff_purify_back(config, x, adj, flags, score_model_x, score_model_adj, sde_x, sde_adj):
    
    score_fn_x = get_score_fn(sde_x, score_model_x, train=False, continuous=True)
    score_fn_adj = get_score_fn(sde_adj, score_model_adj, train=False, continuous=True)

    pure_steps = config.pure.basic_steps
    timesteps = torch.linspace(sde_adj.T, config.pure.eps, pure_steps, device=config.device)
    dt = -1. / config.sde.x.num_scales

    # === get x_t adj_t ===
    t = torch.full((adj.shape[0], ), pure_steps / config.sde.x.num_scales, device=config.device)

    z_x = gen_noise(x, flags, sym=False)
    mean_x, std_x = sde_x.marginal_prob(x, t)
    x_t = mask_x(mean_x + std_x[:, None, None] * z_x, flags)

    z_adj = gen_noise(adj, flags, sym=False)
    mean_adj, std_adj = sde_adj.marginal_prob(adj, t)
    adj_t = mask_adjs(mean_adj + std_adj[:, None, None] * z_adj, flags)

    # === purefication ===
    for i in trange(0, (pure_steps), desc='[Sampling]', position=1, leave=False):
        x_t_1 = x_t.clone().detach()
        adj_t_1 = adj_t.clone().detach()

        # === build time ===
        t = timesteps[i]
        vec_t = torch.ones(adj_t.size(0), device=config.device) * t
        vec_dt = torch.ones(adj_t.size(0), device=config.device) * (dt / 2)

        # === score comutation ===
        score_x = score_fn_x(x_t, adj_t, flags, vec_t)
        score_adj = score_fn_adj(x_t, adj_t, flags, vec_t)

        sdrift_x = -sde_x.sde(x_t, vec_t)[1][:, None, None] ** 2 * score_x # -g(t)^2 * score
        sdrift_adj = -sde_adj.sde(adj_t, vec_t)[1][:, None, None] ** 2 * score_adj
        
        # === Correction ===
        timestep = (vec_t * (sde_x.N - 1) / sde_x.T).long() # t=1 sde_x.N=1000 sde_X.T=1
        
        noise = gen_noise(x_t, flags, sym=False)
        grad_norm = torch.norm(score_x.reshape(score_x.shape[0], -1), dim=-1).mean()
        noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
        if isinstance(sde_x, VPSDE):
          alpha = sde_x.alphas.to(vec_t.device)[timestep]
        else:
          alpha = torch.ones_like(vec_t)
      
        step_size = (config.pure.snr * noise_norm / grad_norm) ** 2 * 2 * alpha
        x_mean = x_t + step_size[:, None, None] * score_x
        x_t = x_mean + torch.sqrt(step_size * 2)[:, None, None] * noise * config.pure.scale_eps

        noise = gen_noise(adj_t, flags)
        grad_norm = torch.norm(score_adj.reshape(score_adj.shape[0], -1), dim=-1).mean()
        noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
        if isinstance(sde_adj, VPSDE):
          alpha = sde_adj.alphas.to(vec_t.device)[timestep] # VP
        else:
          alpha = torch.ones_like(vec_t) # VE
        step_size = (config.pure.snr * noise_norm / grad_norm) ** 2 * 2 * alpha
        adj_mean = adj_t + step_size[:, None, None] * score_adj
        adj_t = adj_mean + torch.sqrt(step_size * 2)[:, None, None] * noise * config.pure.scale_eps

        # === prediction ===
        x_mean = x_t
        adj_mean = adj_t
        mu_x, sigma_x = sde_x.transition(x_t, vec_t, vec_dt)
        mu_adj, sigma_adj = sde_adj.transition(adj_t, vec_t, vec_dt) 
        x_t = mu_x + sigma_x[:, None, None] * gen_noise(x_t, flags, sym=False)
        adj_t = mu_adj + sigma_adj[:, None, None] * gen_noise(adj_t, flags)
        
        x_t = x_t + sdrift_x * dt
        adj_t = adj_t + sdrift_adj * dt

        mu_x, sigma_x = sde_x.transition(x_t, vec_t + vec_dt, vec_dt) 
        mu_adj, sigma_adj = sde_adj.transition(adj_t, vec_t + vec_dt, vec_dt) 
        
        # ====================
        # transfer entropy guided
        if config.pure.is_guide:
          with torch.enable_grad():
            # X_t = mu_x.clone().detach().requires_grad_(True)
            A_t = mu_adj.clone().detach().requires_grad_(True)
            x.reruires_grad, x_t_1.requires_grad, adj.requires_grad, adj_t_1.requires_grad, flags.requires_grad = False, False, False, False, False
            te = -transfer_entropy(adj, A_t, adj_t_1, flags)
            # te = -transfer_entropy(x, X_t, x_t_1, adj, A_t, adj_t_1, flags)
            # te.backward()
            # X_t_grad = torch.autograd.grad(te, X_t)[0]
            A_t_grad = torch.autograd.grad(te, A_t)[0]
            adj_scale = guide_scale(config, t, sde_adj)
            # x_scale, adj_scale = guide_scale(config, t, sde_x, sde_adj)

            # mu_x -= config.pure.guide_rate * X_t.grad
            # mu_adj -= config.pure.guide_rate * A_t.grad

            # mu_x -= x_scale * X_t_grad
            mu_adj -= adj_scale * A_t_grad

        # print(te)
        x_t = mu_x + sigma_x[:, None, None] * gen_noise(x_t, flags, sym=False)
        adj_t = mu_adj + sigma_adj[:, None, None] * gen_noise(adj_t, flags)

        x_mean = mu_x
        adj_mean = mu_adj

    print(' ')
    return x_mean if config.pure.noise_removal else x, adj_mean if config.pure.noise_removal else adj


def diff_purify(config, x, adj, flags, score_model_x, score_model_adj, sde_x, sde_adj):
    
    score_fn_x = get_score_fn(sde_x, score_model_x, train=False, continuous=True)
    score_fn_adj = get_score_fn(sde_adj, score_model_adj, train=False, continuous=True)

    # adaptive steps
    pure_basic_steps = config.pure.basic_steps
    pure_adaptive_steps = config.pure.adaptive_steps
    pure_max_steps = pure_basic_steps + pure_adaptive_steps
    # t_adaptive_adj = pure_basic_steps + adj_anomaly_score * pure_adaptive_steps

    # time
    if config.pure.old_version:
      timesteps = torch.linspace(sde_adj.T, config.pure.eps, pure_max_steps, device=config.device)
    else:
      timesteps = torch.linspace(sde_adj.T, config.diffusion.eps, sde_adj.N, device=config.device)[sde_adj.N - pure_max_steps:] # Default: (1, 0.001, 1000)

    dt = -1. / sde_adj.N

    # === get x_t adj_t ===
    t = torch.full((adj.shape[0], ), pure_max_steps / sde_adj.N, device=config.device)

    z_x = gen_noise(x, flags, sym=False)
    mean_x, std_x = sde_x.marginal_prob(x, t)
    x_t = mask_x(mean_x + std_x[:, None, None] * z_x, flags)

    z_adj = gen_noise(adj, flags, sym=False)
    mean_adj, std_adj = sde_adj.marginal_prob(adj, t)
    adj_t = mask_adjs(mean_adj + std_adj[:, None, None] * z_adj, flags)

    # === purefication ===
    for i in trange(0, (pure_max_steps), desc='[Sampling]', position=1, leave=False):
        # === build time ===
        t = timesteps[i]
        vec_t = torch.ones(adj.size(0), device=config.device) * t
        vec_dt = torch.ones(adj.size(0), device=config.device) * (dt / 2)

        # === adaptive denoise ===
        # reference: Constructing Non-isotropic Gaussian Diffusion Model Using Isotropic Gaussian Diffusion Model for Image Editing [Algorithm 1]
        # if config.pure.adaptive and i < pure_adaptive_steps:
        #   mask_adj_adaptive = (t_adaptive_adj >= (t * sde_adj.N)).int()

        #   mean_x_orig, std_x_orig = sde_x.marginal_prob(x, vec_t)
        #   x_t = mask_x(mean_x_orig + std_x_orig[:, None, None] * z_x, flags)

        #   mean_adj_orig, std_adj_orig = sde_adj.marginal_prob(adj, vec_t)
        #   adj_t_orig = mask_adjs(mean_adj_orig + std_adj_orig[:, None, None] * z_adj, flags)

        #   adj_t = (mask_adj_adaptive * adj_t) + (1 - mask_adj_adaptive) * adj_t_orig

        x_t_1 = x_t.clone().detach()
        adj_t_1 = adj_t.clone().detach()
        # === score comutation ===
        score_x = score_fn_x(x_t, adj_t, flags, vec_t)
        score_adj = score_fn_adj(x_t, adj_t, flags, vec_t)

        sdrift_x = -sde_x.sde(x_t, vec_t)[1][:, None, None] ** 2 * score_x # -g(t)^2 * score
        sdrift_adj = -sde_adj.sde(adj_t, vec_t)[1][:, None, None] ** 2 * score_adj
        
        # === Correction ===
        timestep = (vec_t * (sde_x.N - 1) / sde_x.T).long() # t=1 sde_x.N=1000 sde_X.T=1
        
        noise = gen_noise(x_t, flags, sym=False)
        grad_norm = torch.norm(score_x.reshape(score_x.shape[0], -1), dim=-1).mean()
        noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
        if isinstance(sde_x, VPSDE):
          alpha = sde_x.alphas.to(vec_t.device)[timestep]
        else:
          alpha = torch.ones_like(vec_t)
      
        step_size = (config.pure.snr * noise_norm / grad_norm) ** 2 * 2 * alpha
        x_mean = x_t + step_size[:, None, None] * score_x
        x_t = x_mean + torch.sqrt(step_size * 2)[:, None, None] * noise * config.pure.scale_eps

        noise = gen_noise(adj_t, flags)
        grad_norm = torch.norm(score_adj.reshape(score_adj.shape[0], -1), dim=-1).mean()
        noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
        if isinstance(sde_adj, VPSDE):
          alpha = sde_adj.alphas.to(vec_t.device)[timestep] # VP
        else:
          alpha = torch.ones_like(vec_t) # VE
        step_size = (config.pure.snr * noise_norm / grad_norm) ** 2 * 2 * alpha
        adj_mean = adj_t + step_size[:, None, None] * score_adj
        adj_t = adj_mean + torch.sqrt(step_size * 2)[:, None, None] * noise * config.pure.scale_eps

        # === prediction ===
        x_mean = x_t
        adj_mean = adj_t
        mu_x, sigma_x = sde_x.transition(x_t, vec_t, vec_dt)
        mu_adj, sigma_adj = sde_adj.transition(adj_t, vec_t, vec_dt) 
        x_t = mu_x + sigma_x[:, None, None] * gen_noise(x_t, flags, sym=False)
        adj_t = mu_adj + sigma_adj[:, None, None] * gen_noise(adj_t, flags)
        
        x_t = x_t + sdrift_x * dt
        adj_t = adj_t + sdrift_adj * dt

        mu_x, sigma_x = sde_x.transition(x_t, vec_t + vec_dt, vec_dt) 
        mu_adj, sigma_adj = sde_adj.transition(adj_t, vec_t + vec_dt, vec_dt) 
        
        # ====================
        # transfer entropy guided
        if config.pure.is_guide and (i + 1) % config.pure.guide_step == 0:
          with torch.enable_grad():
            # X_t = mu_x.clone().detach().requires_grad_(True)
            A_t = mu_adj.clone().detach().requires_grad_(True)
            x.reruires_grad, x_t_1.requires_grad, adj.requires_grad, adj_t_1.requires_grad, flags.requires_grad = False, False, False, False, False
            te = -transfer_entropy(adj, A_t, adj_t_1, flags)
            # te = -transfer_entropy(x, X_t, x_t_1, adj, A_t, adj_t_1, flags)
            # te.backward()
            # X_t_grad = torch.autograd.grad(te, X_t)[0]
            A_t_grad = torch.autograd.grad(te, A_t)[0]
            adj_scale = guide_scale(config, t, sde_adj)
            # x_scale, adj_scale = guide_scale(config, t, sde_x, sde_adj)

            # mu_x -= config.pure.guide_rate * X_t.grad
            # mu_adj -= config.pure.guide_rate * A_t.grad

            # mu_x -= x_scale * X_t_grad
            mu_adj -= adj_scale * A_t_grad

        # print(te)
        x_t = mu_x + sigma_x[:, None, None] * gen_noise(x_t, flags, sym=False)
        adj_t = mu_adj + sigma_adj[:, None, None] * gen_noise(adj_t, flags)

        x_mean = mu_x
        adj_mean = mu_adj

    print(' ')
    return x_mean if config.pure.noise_removal else x, adj_mean if config.pure.noise_removal else adj

def sample(config, shape_x, shape_adj, flags, score_model_x, score_model_adj, sde_x, sde_adj):
  score_fn_x = get_score_fn(sde_x, score_model_x, train=False, continuous=True)
  score_fn_adj = get_score_fn(sde_adj, score_model_adj, train=False, continuous=True)

  with torch.no_grad():
    # -------- Initial sample --------
    x = sde_x.prior_sampling(shape_x).to(config.device)
    adj = sde_adj.prior_sampling_sym(shape_adj).to(config.device)
    x = mask_x(x, flags)
    adj = mask_adjs(adj, flags)
    diff_steps = sde_adj.N
    timesteps = torch.linspace(sde_adj.T, config.diffusion.eps, diff_steps, device=config.device)
    dt = -1. / diff_steps

    # -------- Rverse diffusion process --------
    for i in trange(0, (diff_steps), desc = '[Sampling]', position = 1, leave=False):
      t = timesteps[i]
      vec_t = torch.ones(shape_adj[0], device=t.device) * t
      vec_dt = torch.ones(shape_adj[0], device=t.device) * (dt/2)

      # -------- Score computation --------
      score_x = score_fn_x(x, adj, flags, vec_t)
      score_adj = score_fn_adj(x, adj, flags, vec_t)

      Sdrift_x = -sde_x.sde(x, vec_t)[1][:, None, None] ** 2 * score_x
      Sdrift_adj  = -sde_adj.sde(adj, vec_t)[1][:, None, None] ** 2 * score_adj

      # -------- Correction step --------
      timestep = (vec_t * (sde_x.N - 1) / sde_x.T).long()

      noise = gen_noise(x, flags, sym=False)
      grad_norm = torch.norm(score_x.reshape(score_x.shape[0], -1), dim=-1).mean()
      noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
      if isinstance(sde_x, VPSDE):
        alpha = sde_x.alphas.to(vec_t.device)[timestep]
      else:
        alpha = torch.ones_like(vec_t)
    
      step_size = (config.diffusion.snr * noise_norm / grad_norm) ** 2 * 2 * alpha
      x_mean = x + step_size[:, None, None] * score_x
      x = x_mean + torch.sqrt(step_size * 2)[:, None, None] * noise * config.diffusion.scale_eps

      noise = gen_noise(adj, flags)
      grad_norm = torch.norm(score_adj.reshape(score_adj.shape[0], -1), dim=-1).mean()
      noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
      if isinstance(sde_adj, VPSDE):
        alpha = sde_adj.alphas.to(vec_t.device)[timestep] # VP
      else:
        alpha = torch.ones_like(vec_t) # VE
      step_size = (config.diffusion.snr * noise_norm / grad_norm) ** 2 * 2 * alpha
      adj_mean = adj + step_size[:, None, None] * score_adj
      adj = adj_mean + torch.sqrt(step_size * 2)[:, None, None] * noise * config.diffusion.scale_eps

      # -------- Prediction step --------
      x_mean = x
      adj_mean = adj
      mu_x, sigma_x = sde_x.transition(x, vec_t, vec_dt)
      mu_adj, sigma_adj = sde_adj.transition(adj, vec_t, vec_dt) 
      x = mu_x + sigma_x[:, None, None] * gen_noise(x, flags, sym=False)
      adj = mu_adj + sigma_adj[:, None, None] * gen_noise(adj, flags)
      
      x = x + Sdrift_x * dt
      adj = adj + Sdrift_adj * dt

      mu_x, sigma_x = sde_x.transition(x, vec_t + vec_dt, vec_dt) 
      mu_adj, sigma_adj = sde_adj.transition(adj, vec_t + vec_dt, vec_dt) 
      x = mu_x + sigma_x[:, None, None] * gen_noise(x, flags, sym=False)
      adj = mu_adj + sigma_adj[:, None, None] * gen_noise(adj, flags)

      x_mean = mu_x
      adj_mean = mu_adj

    print(' ')

    return x_mean, adj_mean