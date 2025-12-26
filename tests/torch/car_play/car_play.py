#!/usr/bin/env python3
# MIT License
# Copyright (c) 2025 Anton Schreiner

"""
World Models for CarRacing (Ha & Schmidhuber 2018 style)

Three-stage training:
1. VAE: Learn to compress observations to latent Z
2. World Model (MDN-RNN): Learn dynamics p(z_{t+1}, r, done | z_t, a_t, h_t)
3. Controller: Train policy entirely in "dreams" using the world model

Usage:
    # Stage 1: Collect random rollouts and train VAE
    python world_models.py command=train_vae
    
    # Stage 2: Train world model on collected data
    python world_models.py command=train_world_model
    
    # Stage 3: Train controller in dreams
    python world_models.py command=train_controller
    
    # Generate video with trained agent
    python world_models.py command=generate_video
    
    # Run all stages sequentially
    python world_models.py command=train_all
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.distributions import Normal, Categorical, MixtureSameFamily
from pathlib import Path
import pickle
import math
import time
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.config_store import ConfigStore
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from gymnasium.vector import SyncVectorEnv


# ============================================================================
# Config
# ============================================================================

@dataclass
class VAEConfig:
    latent_dim: int = 32
    kld_weight: float = 0.0001


@dataclass
class WorldModelConfig:
    hidden_dim      : int = 256
    num_layers      : int = 4
    num_gaussians   : int = 5  # For MDN output
    num_quantiles   : int = 32  # For distributional value prediction
    temperature     : float = 1.0

@dataclass
class ControllerConfig:
    hidden_dim: int = 64


@dataclass
class TrainingConfig:
    # Outer loop
    num_major_iterations: int = 10
    
    # Data collection
    num_rollouts            : int = 20  # Per iteration
    max_rollouts            : int = 50  # Max to keep in buffer
    max_steps_per_rollout   : int = 1000
    num_envs                : int = 8
    
    # VAE training
    vae_epochs       : int = 5
    vae_batch_size   : int = 64
    vae_lr           : float = 1e-3
    
    # World model training
    wm_epochs       : int = 10
    wm_batch_size   : int = 32
    wm_seq_len      : int = 64
    wm_lr           : float = 1e-3
    
    # Controller training (PPO in dreams)
    ctrl_iterations    : int        = 100  # Number of PPO iterations
    ctrl_lr            : float      = 3e-4
    dream_horizon      : int        = 1024  # Steps per dream rollout
    num_dream_envs     : int        = 8  # Parallel dream rollouts
    ppo_epochs         : int        = 4  # Epochs per PPO update
    ppo_batch_size     : int        = 64  # Minibatch size for PPO
    gamma              : float      = 0.99  # Discount factor
    gae_lambda         : float      = 0.95  # GAE lambda
    lambda_horizon     : float      = 0.95  # Dreamer-style λ for value estimation
    clip_epsilon       : float      = 0.2  # PPO clip range
    entropy_coef       : float      = 0.01  # Entropy bonus
    value_coef         : float      = 0.5  # Value loss coefficient
    max_grad_norm      : float      = 0.5  # Gradient clipping
    risk_aversion      : float      = 0.5  # 0.0 = risk-neutral, 1.0 = very risk-averse (CVaR)
    
    # General
    grad_clip: float = 1.0
    save_interval: int = 10


@dataclass
class CheckpointConfig:
    data_path               : str = ".tmp/rollout_data.pkl"
    vae_path                : str = ".tmp/vae.pth"
    world_model_path        : str = ".tmp/world_model.pth"
    controller_path         : str = ".tmp/controller.pth"
    fresh: bool = False


@dataclass
class Config:
    vae                 : VAEConfig = field(default_factory=VAEConfig)
    world_model         : WorldModelConfig = field(default_factory=WorldModelConfig)
    controller          : ControllerConfig = field(default_factory=ControllerConfig)
    training            : TrainingConfig = field(default_factory=TrainingConfig)
    checkpoint          : CheckpointConfig = field(default_factory=CheckpointConfig)
    command             : str = "train_all"
    device              : str = "cuda" if torch.cuda.is_available() else "cpu"


cs = ConfigStore.instance()
cs.store(name="config", node=Config)


# ============================================================================
# Action Space
# ============================================================================

NUM_ACTIONS = 5
DISCRETE_ACTIONS = {
    0: np.array([0.0, 0.0, 0.0]),   # nothing
    1: np.array([-1.0, 0.0, 0.0]),  # left
    2: np.array([1.0, 0.0, 0.0]),   # right  
    3: np.array([0.0, 1.0, 0.0]),   # gas
    4: np.array([0.0, 0.0, 0.8]),   # brake
}


def action_to_env(action_idx: int) -> np.ndarray:
    return DISCRETE_ACTIONS[action_idx]


def actions_to_env(action_indices: np.ndarray) -> np.ndarray:
    return np.array([DISCRETE_ACTIONS[int(a)] for a in action_indices])


# ============================================================================
# VAE
# ============================================================================

class Encoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        # Input: (3, 96, 96)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),    # -> 32x48x48
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),   # -> 64x24x24
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),  # -> 128x12x12
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), # -> 256x6x6
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256 * 6 * 6, latent_dim)
        self.fc_logvar = nn.Linear(256 * 6 * 6, latent_dim)
    
    def forward(self, x):
        h = self.conv(x)
        h = h.reshape(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 6 * 6)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # -> 128x12x12
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # -> 64x24x24
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # -> 32x48x48
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),     # -> 3x96x96
            nn.Sigmoid(),
        )
    
    def forward(self, z):
        h = self.fc(z)
        h = h.reshape(h.size(0), 256, 6, 6)
        return self.deconv(h)


class VAE(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        latent_dim = cfg.latent_dim if hasattr(cfg, 'latent_dim') else cfg['latent_dim']
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
    
    def encode(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)
    
    def decode(self, z) -> torch.Tensor:
        return self.decoder(z)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
    def loss(self, x, recon, mu, logvar, kld_weight=0.0001):
        recon_loss = F.mse_loss(recon, x, reduction='mean')
        kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kld_weight * kld_loss, recon_loss, kld_loss


# ============================================================================
# World Model (MDN-RNN)
# ============================================================================

class MDNRNN(nn.Module):
    """
    Mixture Density Network RNN.
    Predicts p(z_{t+1} | z_t, a_t, h_t) as a mixture of Gaussians.
    Also predicts reward, done probability, and distributional value (quantiles).
    """
    def __init__(self, latent_dim: int, action_dim: int, cfg: WorldModelConfig):
        super().__init__()
        hidden_dim = cfg.hidden_dim if hasattr(cfg, 'hidden_dim') else cfg['hidden_dim']
        num_layers = cfg.num_layers if hasattr(cfg, 'num_layers') else cfg['num_layers']
        num_gaussians = cfg.num_gaussians if hasattr(cfg, 'num_gaussians') else cfg['num_gaussians']
        num_quantiles = cfg.num_quantiles if hasattr(cfg, 'num_quantiles') else cfg.get('num_quantiles', 32)
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_gaussians = num_gaussians
        self.num_quantiles = num_quantiles
        
        # Fixed quantile levels τ ∈ (0, 1)
        # Midpoints of uniform intervals: [0.5/N, 1.5/N, ..., (N-0.5)/N]
        self.register_buffer(
            'taus',
            torch.linspace(0.5 / num_quantiles, 1 - 0.5 / num_quantiles, num_quantiles)
        )
        
        # RNN
        self.rnn = nn.LSTM(
            input_size=latent_dim + action_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # MDN outputs for next z
        # For each Gaussian: pi (weight), mu (mean), sigma (std)
        self.fc_pi = nn.Linear(hidden_dim, num_gaussians)
        self.fc_mu = nn.Linear(hidden_dim, num_gaussians * latent_dim)
        self.fc_sigma = nn.Linear(hidden_dim, num_gaussians * latent_dim)
        
        # Reward and done prediction
        self.fc_reward = nn.Linear(hidden_dim, 1)
        self.fc_done = nn.Linear(hidden_dim, 1)
        
        # Distributional value prediction (quantiles instead of mean)
        self.fc_quantiles = nn.Linear(hidden_dim, num_quantiles)
    
    def init_hidden(self, batch_size: int, device: torch.device):
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device),
            torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        )
    
    def forward(self, z, action, hidden=None):
        """
        Args:
            z: (B, T, latent_dim)
            action: (B, T, action_dim) one-hot
            hidden: optional (h, c) tuple
            
        Returns:
            pi: (B, T, num_gaussians) mixture weights
            mu: (B, T, num_gaussians, latent_dim)
            sigma: (B, T, num_gaussians, latent_dim)
            reward: (B, T)
            done: (B, T)
            hidden: new hidden state
        """
        B, T, _ = z.shape
        
        x = torch.cat([z, action], dim=-1)
        
        if hidden is None:
            hidden = self.init_hidden(B, z.device)
        
        out, hidden = self.rnn(x, hidden)  # (B, T, hidden_dim)
        
        # MDN outputs
        pi = F.softmax(self.fc_pi(out), dim=-1)  # (B, T, num_gaussians)
        mu = self.fc_mu(out).reshape(B, T, self.num_gaussians, self.latent_dim)
        sigma = F.softplus(self.fc_sigma(out)).reshape(B, T, self.num_gaussians, self.latent_dim)
        sigma = sigma + 1e-3  # minimum std
        
        # Reward and done
        reward = self.fc_reward(out).squeeze(-1)
        done = torch.sigmoid(self.fc_done(out).squeeze(-1))
        
        # Distributional value: predict quantiles
        quantiles = self.fc_quantiles(out)  # (B, T, num_quantiles)
        
        return pi, mu, sigma, reward, done, quantiles, hidden
    
    def sample(self, z, action, hidden, temperature=1.0):
        """
        Sample next z from the MDN.
        
        Args:
            z: (B, latent_dim)
            action: (B, action_dim) one-hot
            hidden: (h, c) tuple
            temperature: sampling temperature
            
        Returns:
            next_z: (B, latent_dim)
            reward: (B,)
            done: (B,)
            quantiles: (B, num_quantiles) - value distribution
            hidden: new hidden state
        """
        # Add time dimension
        z = z.unsqueeze(1)
        action = action.unsqueeze(1)
        
        pi, mu, sigma, reward, done, quantiles, hidden = self.forward(z, action, hidden)
        
        # Remove time dimension
        pi = pi.squeeze(1)        # (B, num_gaussians)
        mu = mu.squeeze(1)        # (B, num_gaussians, latent_dim)
        sigma = sigma.squeeze(1)  # (B, num_gaussians, latent_dim)
        reward = reward.squeeze(1)
        done = done.squeeze(1)
        quantiles = quantiles.squeeze(1)  # (B, num_quantiles)
        
        # Sample from mixture
        if temperature > 0:
            # Sample which Gaussian to use
            pi_temp = F.log_softmax(torch.log(pi + 1e-8) / temperature, dim=-1)
            component = torch.multinomial(torch.exp(pi_temp), 1).squeeze(-1)  # (B,)
            
            # Get parameters for selected component
            batch_idx = torch.arange(mu.size(0), device=mu.device)
            mu_selected = mu[batch_idx, component]      # (B, latent_dim)
            sigma_selected = sigma[batch_idx, component]  # (B, latent_dim)
            
            # Sample
            next_z = mu_selected + sigma_selected * torch.randn_like(sigma_selected) * temperature
        else:
            # Deterministic: use most likely component's mean
            component = pi.argmax(dim=-1)
            batch_idx = torch.arange(mu.size(0), device=mu.device)
            next_z = mu[batch_idx, component]
        
        return next_z, reward, done, quantiles, hidden
    
    def get_risk_adjusted_value(self, quantiles, risk_aversion=0.0):
        """
        Compute risk-adjusted value from quantiles.
        
        Args:
            quantiles: (..., num_quantiles) predicted quantile values
            risk_aversion: 0.0 = risk-neutral (mean), 1.0 = very risk-averse (min quantile)
            
        Returns:
            value: (...) risk-adjusted scalar value
        """
        if risk_aversion <= 0:
            # Risk-neutral: return mean of distribution
            return quantiles.mean(dim=-1)
        
        # CVaR (Conditional Value at Risk): average of bottom quantiles
        # risk_aversion controls what fraction of worst outcomes to consider
        num_q = quantiles.shape[-1]
        k = max(1, int(num_q * (1 - risk_aversion * 0.9)))  # Keep at least 1, at most all
        
        # Sort quantiles and take mean of bottom k
        sorted_q, _ = quantiles.sort(dim=-1)
        return sorted_q[..., :k].mean(dim=-1)
    
    def quantile_regression_loss(self, pred_quantiles, target_returns, mask=None):
        """
        Quantile regression loss (Huber variant for stability).
        
        Args:
            pred_quantiles: (B, T, num_quantiles) predicted quantiles
            target_returns: (B, T) actual returns
            mask: (B, T) optional mask for valid timesteps
            
        Returns:
            loss: scalar
        """
        # Expand target to compare with each quantile
        # target: (B, T) -> (B, T, 1)
        # pred: (B, T, num_quantiles)
        target = target_returns.unsqueeze(-1)  # (B, T, 1)
        
        # Error for each quantile
        errors = target - pred_quantiles  # (B, T, num_quantiles)
        
        # Huber loss (more stable than pure quantile loss)
        kappa = 1.0  # Huber threshold
        abs_errors = errors.abs()
        huber = torch.where(
            abs_errors <= kappa,
            0.5 * errors.pow(2),
            kappa * (abs_errors - 0.5 * kappa)
        )
        
        # Asymmetric weighting by tau
        # Below quantile (error > 0): weight by tau
        # Above quantile (error < 0): weight by (1 - tau)
        taus = self.taus.view(1, 1, -1)  # (1, 1, num_quantiles)
        quantile_weights = torch.where(errors > 0, taus, 1 - taus)
        
        loss = (quantile_weights * huber)
        
        if mask is not None:
            loss = loss * mask.unsqueeze(-1)
            return loss.sum() / (mask.sum() * self.num_quantiles + 1e-8)
        
        return loss.mean()
    
    def mdn_loss(self, pi, mu, sigma, target):
        """
        Negative log-likelihood of target under the mixture.
        
        Args:
            pi: (B, T, num_gaussians)
            mu: (B, T, num_gaussians, latent_dim)
            sigma: (B, T, num_gaussians, latent_dim)
            target: (B, T, latent_dim)
        """
        target = target.unsqueeze(2)  # (B, T, 1, latent_dim)
        
        # Log probability under each Gaussian
        log_prob = -0.5 * (
            ((target - mu) / sigma).pow(2) + 
            2 * torch.log(sigma) + 
            math.log(2 * math.pi)
        ).sum(dim=-1)  # (B, T, num_gaussians)
        
        # Log mixture probability
        log_pi = torch.log(pi + 1e-8)
        log_prob_mixture = torch.logsumexp(log_pi + log_prob, dim=-1)  # (B, T)
        
        return -log_prob_mixture.mean()


# ============================================================================
# Controller (Actor-Critic Policy)
# ============================================================================

class Controller(nn.Module):
    """
    Actor-Critic controller for PPO training.
    Takes z and RNN hidden state, outputs action distribution and value.
    """
    def __init__(self, latent_dim: int, hidden_dim: int, rnn_hidden_dim: int, action_dim: int):
        super().__init__()
        input_dim = latent_dim + rnn_hidden_dim
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
        )
        
        # Actor head (policy)
        self.actor = nn.Linear(hidden_dim, action_dim)
        
        # Critic head (value function)
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, z, h):
        """
        Args:
            z: (B, latent_dim)
            h: (B, rnn_hidden_dim)
        Returns:
            action_logits: (B, action_dim)
            value: (B,)
        """
        x = torch.cat([z, h], dim=-1)
        features = self.shared(x)
        logits = self.actor(features)
        value = self.critic(features).squeeze(-1)
        return logits, value
    
    def get_action_and_value(self, z, h, action=None):
        """
        Sample action and compute log_prob, entropy, value.
        If action provided, compute log_prob for that action.
        """
        logits, value = self.forward(z, h)
        dist = torch.distributions.Categorical(logits=logits)
        
        if action is None:
            action = dist.sample()
        
        return action, dist.log_prob(action), dist.entropy(), value
    
    def get_value(self, z, h):
        """Get value estimate only."""
        _, value = self.forward(z, h)
        return value
    
    def get_action(self, z, h, deterministic=False):
        """Simple action getter for evaluation."""
        logits, _ = self.forward(z, h)
        if deterministic:
            return logits.argmax(dim=-1)
        else:
            dist = torch.distributions.Categorical(logits=logits)
            return dist.sample()


# ============================================================================
# Data Collection
# ============================================================================

def collect_rollouts_with_policy(cfg: Config, vae=None, world_model=None, controller=None) -> dict:
    """Collect rollouts using current policy (or random if no controller)."""
    print("=== Collecting Rollouts ===")
    
    num_envs = cfg.training.num_envs
    num_rollouts = cfg.training.num_rollouts
    max_steps = cfg.training.max_steps_per_rollout
    device = cfg.device
    
    use_policy = controller is not None and vae is not None and world_model is not None
    if use_policy:
        print("  Using learned policy")
        vae.eval()
        world_model.eval()
        controller.eval()
    else:
        print("  Using random policy")
    
    env = SyncVectorEnv([lambda: gym.make("CarRacing-v3", render_mode="rgb_array") for _ in range(num_envs)])
    
    all_obs = []
    all_actions = []
    all_rewards = []
    all_dones = []
    
    rollouts_collected = 0
    obs, _ = env.reset()
    
    # Per-env hidden states
    if use_policy:
        hiddens = [world_model.init_hidden(1, device) for _ in range(num_envs)]
    
    episode_obs = [[] for _ in range(num_envs)]
    episode_actions = [[] for _ in range(num_envs)]
    episode_rewards = [[] for _ in range(num_envs)]
    episode_dones = [[] for _ in range(num_envs)]
    
    while rollouts_collected < num_rollouts:
        if use_policy:
            # Get actions from policy
            actions = []
            with torch.no_grad():
                for i in range(num_envs):
                    obs_t = torch.from_numpy(obs[i]).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
                    z, _ = vae.encode(obs_t)
                    h = hiddens[i][0][0]  # (1, hidden_dim)
                    
                    # Mix of deterministic and stochastic for exploration
                    action = controller.get_action(z, h, deterministic=False)
                    action_oh = F.one_hot(action, NUM_ACTIONS).float()
                    
                    # Update hidden
                    _, _, _, _, _, _, hiddens[i] = world_model(z.unsqueeze(1), action_oh.unsqueeze(1), hiddens[i])
                    
                    actions.append(action.item())
            actions = np.array(actions)
        else:
            actions = np.random.randint(0, NUM_ACTIONS, size=num_envs)
        
        for i in range(num_envs):
            episode_obs[i].append(obs[i])
            episode_actions[i].append(actions[i])
        
        env_actions = actions_to_env(actions)
        next_obs, rewards, terminated, truncated, _ = env.step(env_actions)
        dones = terminated | truncated
        
        for i in range(num_envs):
            episode_rewards[i].append(rewards[i])
            episode_dones[i].append(dones[i])
            
            if dones[i] or len(episode_obs[i]) >= max_steps:
                if len(episode_obs[i]) > 10:
                    all_obs.append(np.stack(episode_obs[i]))
                    all_actions.append(np.array(episode_actions[i]))
                    all_rewards.append(np.array(episode_rewards[i]))
                    all_dones.append(np.array(episode_dones[i]))
                    rollouts_collected += 1
                    
                    if rollouts_collected % 10 == 0:
                        total_reward = sum(episode_rewards[i])
                        print(f"  Collected {rollouts_collected}/{num_rollouts} rollouts (last reward: {total_reward:.1f})")
                
                episode_obs[i] = []
                episode_actions[i] = []
                episode_rewards[i] = []
                episode_dones[i] = []
                
                # Reset hidden for this env
                if use_policy:
                    hiddens[i] = world_model.init_hidden(1, device)
        
        obs = next_obs
        
        if rollouts_collected >= num_rollouts:
            break
    
    env.close()
    
    data = {
        'obs': all_obs,
        'actions': all_actions,
        'rewards': all_rewards,
        'dones': all_dones,
    }
    
    total_frames = sum(len(o) for o in all_obs)
    print(f"Collected {len(all_obs)} rollouts, {total_frames} total frames")
    
    return data


def merge_data(old_data: dict, new_data: dict, max_rollouts: int) -> dict:
    """Merge new rollouts into existing data, keeping most recent."""
    merged = {
        'obs': old_data['obs'] + new_data['obs'],
        'actions': old_data['actions'] + new_data['actions'],
        'rewards': old_data['rewards'] + new_data['rewards'],
        'dones': old_data['dones'] + new_data['dones'],
    }
    
    # Keep only most recent rollouts
    if len(merged['obs']) > max_rollouts:
        merged = {
            'obs': merged['obs'][-max_rollouts:],
            'actions': merged['actions'][-max_rollouts:],
            'rewards': merged['rewards'][-max_rollouts:],
            'dones': merged['dones'][-max_rollouts:],
        }
    
    return merged


def collect_random_rollouts(cfg: Config) -> dict:
    """Collect rollouts with random policy."""
    return collect_rollouts_with_policy(cfg, None, None, None)


def save_data(data: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved data to {path}")


def load_data(path: str) -> dict:
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded data from {path}")
    return data


# ============================================================================
# Training Functions
# ============================================================================

def train_vae(cfg: Config, vae=None, world_model=None, controller=None):
    """Stage 1: Collect data and train VAE."""
    print("=== Stage 1: Training VAE ===")
    device = cfg.device
    
    # Check if we can skip data collection (all checkpoints exist)
    has_data = Path(cfg.checkpoint.data_path).exists()
    has_vae = Path(cfg.checkpoint.vae_path).exists()
    has_wm = Path(cfg.checkpoint.world_model_path).exists()
    has_ctrl = Path(cfg.checkpoint.controller_path).exists()
    
    if has_data and has_vae and has_wm and has_ctrl and not cfg.checkpoint.fresh:
        print("  Found existing checkpoints, loading models for on-policy collection...")
        # Load models if not provided
        if vae is None:
            vae = VAE(cfg.vae).to(device)
            vae.load_state_dict(torch.load(cfg.checkpoint.vae_path, weights_only=True))
        if world_model is None:
            world_model = MDNRNN(cfg.vae.latent_dim, NUM_ACTIONS, cfg.world_model).to(device)
            try:
                world_model.load_state_dict(torch.load(cfg.checkpoint.world_model_path, weights_only=True), strict=False)
            except RuntimeError:
                print("  Warning: Could not fully load world model state dict (possible architecture mismatch).")
        if controller is None:
            controller = Controller(cfg.vae.latent_dim, cfg.controller.hidden_dim, 
                                   cfg.world_model.hidden_dim, NUM_ACTIONS).to(device)
            try:
                controller.load_state_dict(torch.load(cfg.checkpoint.controller_path, weights_only=True))
            except RuntimeError:
                print("  Warning: Could not fully load controller state dict (possible architecture mismatch).")
    elif has_data and not cfg.checkpoint.fresh:
        print("  Found existing data, skipping initial collection...")
        # Just load existing data, don't collect new
        data = load_data(cfg.checkpoint.data_path)
    
    # Collect new rollouts (with policy if available, else random)
    if controller is not None and vae is not None and world_model is not None:
        new_data = collect_rollouts_with_policy(cfg, vae, world_model, controller)
        # Load existing data and merge
        if Path(cfg.checkpoint.data_path).exists() and not cfg.checkpoint.fresh:
            old_data = load_data(cfg.checkpoint.data_path)
            data = merge_data(old_data, new_data, cfg.training.max_rollouts)
            print(f"Merged data: {len(data['obs'])} total rollouts")
        else:
            data = new_data
        save_data(data, cfg.checkpoint.data_path)
    elif not has_data or cfg.checkpoint.fresh:
        # No controller yet, collect random rollouts
        new_data = collect_rollouts_with_policy(cfg, None, None, None)
        data = new_data
        save_data(data, cfg.checkpoint.data_path)
    # else: data already loaded above
    
    # Stack all observations
    all_obs = np.concatenate(data['obs'], axis=0)  # (N, 96, 96, 3)
    all_obs = torch.from_numpy(all_obs).permute(0, 3, 1, 2).float() / 255.0  # (N, 3, 96, 96)
    
    print(f"Training VAE on {len(all_obs)} frames")
    
    dataset = TensorDataset(all_obs)
    loader = DataLoader(dataset, batch_size=cfg.training.vae_batch_size, shuffle=True, num_workers=0)
    
    # Create or reuse VAE
    if vae is None:
        vae = VAE(cfg.vae).to(device)
        if Path(cfg.checkpoint.vae_path).exists() and not cfg.checkpoint.fresh:
            vae.load_state_dict(torch.load(cfg.checkpoint.vae_path, weights_only=True))
            print(f"Loaded VAE from {cfg.checkpoint.vae_path}")
    
    optimizer = torch.optim.Adam(vae.parameters(), lr=cfg.training.vae_lr)
    
    for epoch in range(cfg.training.vae_epochs):
        total_loss = 0
        total_recon = 0
        total_kld = 0
        
        for batch_idx, (x,) in enumerate(loader):
            x = x.to(device)
            recon, mu, logvar = vae(x)
            loss, recon_loss, kld_loss = vae.loss(x, recon, mu, logvar, cfg.vae.kld_weight)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kld += kld_loss.item()
            
            # Save checkpoint and DDS every 100 batches
            if batch_idx % 100 == 0:
                from py.torch_utils import dds_from_tensor, DelayedKeyboardInterrupt
                with DelayedKeyboardInterrupt():
                    while True:
                        try:
                            torch.save(vae.state_dict(), cfg.checkpoint.vae_path)
                            break
                        except Exception as e:
                            pass  # Ignore file access issues or keyboard interrupts
                try:
                    dds = dds_from_tensor(recon[0:1])
                    dds.save(".tmp/vae_recon.dds")
                    dds = dds_from_tensor(x[0:1])
                    dds.save(".tmp/vae_orig.dds")
                except Exception as e:
                    print(f"Failed to save DDS: {e}")
        
        avg_loss = total_loss / len(loader)
        avg_recon = total_recon / len(loader)
        avg_kld = total_kld / len(loader)
        print(f"Epoch {epoch+1}/{cfg.training.vae_epochs}: loss={avg_loss:.4f}, recon={avg_recon:.4f}, kld={avg_kld:.4f}")
        
        from py.torch_utils import DelayedKeyboardInterrupt
        with DelayedKeyboardInterrupt():
            while True:
                try:
                    torch.save(vae.state_dict(), cfg.checkpoint.vae_path)
                    break
                except Exception as e:
                    pass
    
    print(f"VAE saved to {cfg.checkpoint.vae_path}")
    return vae


def train_world_model(cfg: Config, vae=None):
    """Stage 2: Train MDN-RNN world model."""
    print("=== Stage 2: Training World Model ===")
    device = cfg.device
    
    # Load data
    data = load_data(cfg.checkpoint.data_path)
    
    # Load or reuse VAE
    if vae is None:
        vae = VAE(cfg.vae).to(device)
        vae.load_state_dict(torch.load(cfg.checkpoint.vae_path, weights_only=True))
        print(f"Loaded VAE from {cfg.checkpoint.vae_path}")
    vae.eval()
    
    # Encode all observations to latents
    print("Encoding observations to latents...")
    encoded_rollouts = []
    
    with torch.no_grad():
        for obs in data['obs']:
            obs_t = torch.from_numpy(obs).permute(0, 3, 1, 2).float().to(device) / 255.0
            mu, _ = vae.encode(obs_t)
            encoded_rollouts.append(mu.cpu())
    
    # Create world model
    world_model = MDNRNN(
        latent_dim=cfg.vae.latent_dim,
        action_dim=NUM_ACTIONS,
        cfg=cfg.world_model
    ).to(device)
    
    if Path(cfg.checkpoint.world_model_path).exists() and not cfg.checkpoint.fresh:
        world_model.load_state_dict(torch.load(cfg.checkpoint.world_model_path, weights_only=True), strict=False)
        print(f"Loaded world model from {cfg.checkpoint.world_model_path}")
    
    optimizer = torch.optim.Adam(world_model.parameters(), lr=cfg.training.wm_lr)
    
    # Training loop
    seq_len = cfg.training.wm_seq_len
    batch_size = cfg.training.wm_batch_size
    
    for epoch in range(cfg.training.wm_epochs):
        total_z_loss = 0
        total_r_loss = 0
        total_d_loss = 0
        num_batches = 0
        
        # Sample random sequences
        for batch_num in range(100):  # 100 batches per epoch
            # Sample batch_size random rollouts and starting positions
            batch_z = []
            batch_a = []
            batch_r = []
            batch_d = []
            
            for _ in range(batch_size):
                # Pick random rollout
                idx = np.random.randint(len(encoded_rollouts))
                z_seq = encoded_rollouts[idx]
                a_seq = data['actions'][idx]
                r_seq = data['rewards'][idx]
                d_seq = data['dones'][idx]
                
                # Pick random starting point
                max_start = max(0, len(z_seq) - seq_len - 1)
                start = np.random.randint(0, max_start + 1)
                end = min(start + seq_len, len(z_seq) - 1)
                
                batch_z.append(z_seq[start:end])
                batch_a.append(a_seq[start:end])
                batch_r.append(r_seq[start:end])
                batch_d.append(d_seq[start:end])
            
            # Pad sequences to same length
            max_len = max(len(z) for z in batch_z)
            z_padded = torch.zeros(batch_size, max_len, cfg.vae.latent_dim)
            a_padded = torch.zeros(batch_size, max_len, NUM_ACTIONS)
            r_padded = torch.zeros(batch_size, max_len)
            d_padded = torch.zeros(batch_size, max_len)
            mask = torch.zeros(batch_size, max_len)
            
            for i in range(batch_size):
                L = len(batch_z[i])
                z_padded[i, :L] = batch_z[i]
                a_padded[i, :L] = F.one_hot(torch.from_numpy(batch_a[i]).long(), NUM_ACTIONS).float()
                r_padded[i, :L] = torch.from_numpy(batch_r[i]).float()
                d_padded[i, :L] = torch.from_numpy(batch_d[i]).float()
                mask[i, :L] = 1.0
            
            z_padded = z_padded.to(device)
            a_padded = a_padded.to(device)
            r_padded = r_padded.to(device)
            d_padded = d_padded.to(device)
            mask = mask.to(device)
            
            # Forward pass (predict next step from current)
            z_input = z_padded[:, :-1]
            a_input = a_padded[:, :-1]
            z_target = z_padded[:, 1:]
            r_target = r_padded[:, 1:]
            d_target = d_padded[:, 1:]
            mask = mask[:, 1:]
            
            if z_input.size(1) == 0:
                continue
            
            pi, mu, sigma, r_pred, d_pred, q_pred, _ = world_model(z_input, a_input)
            
            # Compute TD targets for distributional value (Dreamer-style)
            # For quantile regression: target = r + γ * (1 - done) * V_mean(s')
            # We use the mean of next state's distribution as bootstrap
            with torch.no_grad():
                # Get quantiles of next states
                _, _, _, _, _, q_next, _ = world_model(z_target, a_padded[:, 1:])
                # Use mean of quantiles as bootstrap value
                v_next = q_next.mean(dim=-1)
                v_target = r_target + cfg.training.gamma * (1 - d_target) * v_next
            
            # Losses
            z_loss = world_model.mdn_loss(pi, mu, sigma, z_target)
            r_loss = F.mse_loss(r_pred * mask, r_target * mask)
            d_loss = F.binary_cross_entropy(d_pred * mask, d_target * mask)
            q_loss = world_model.quantile_regression_loss(q_pred, v_target, mask)  # Distributional value loss
            
            loss = z_loss + r_loss + 0.1 * d_loss + 0.5 * q_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(world_model.parameters(), cfg.training.grad_clip)
            optimizer.step()
            
            total_z_loss += z_loss.item()
            total_r_loss += r_loss.item()
            total_d_loss += d_loss.item()
            num_batches += 1
            
            # Save checkpoint and visualize predictions periodically
            if batch_num % 50 == 0:
                from py.torch_utils import dds_from_tensor, DelayedKeyboardInterrupt
                with DelayedKeyboardInterrupt():
                    while True:
                        try:
                            torch.save(world_model.state_dict(), cfg.checkpoint.world_model_path)
                            break
                        except Exception as e:
                            pass  # Ignore file access issues or keyboard interrupts
                
                # Visualize: decode predicted z vs actual z
                try:
                    with torch.no_grad():
                        # Get the mean of the most likely Gaussian
                        best_component = pi[0, 0].argmax()
                        z_pred = mu[0, 0, best_component]  # (latent_dim,)
                        z_actual = z_target[0, 0]  # (latent_dim,)
                        
                        recon_pred = vae.decode(z_pred.unsqueeze(0))
                        recon_actual = vae.decode(z_actual.unsqueeze(0))
                        
                        dds = dds_from_tensor(recon_pred)
                        dds.save(".tmp/wm_pred.dds")
                        dds = dds_from_tensor(recon_actual)
                        dds.save(".tmp/wm_actual.dds")
                except Exception as e:
                    print(f"Failed to save DDS: {e}")
        
        print(f"Epoch {epoch+1}/{cfg.training.wm_epochs}: "
              f"z_loss={total_z_loss/num_batches:.4f}, "
              f"r_loss={total_r_loss/num_batches:.4f}, "
              f"d_loss={total_d_loss/num_batches:.4f}")
        
        from py.torch_utils import DelayedKeyboardInterrupt
        with DelayedKeyboardInterrupt():
            while True:
                try:
                    torch.save(world_model.state_dict(), cfg.checkpoint.world_model_path)
                    break
                except Exception as e:
                    pass
    
    print(f"World model saved to {cfg.checkpoint.world_model_path}")
    return vae, world_model


def collect_dream_rollouts(vae, world_model, controller, cfg: Config):
    """
    Collect rollouts in dreams for PPO training.
    Uses Dreamer-style λ-returns from world model's distributional value head.
    Supports risk-sensitive action selection via CVaR.
    """
    device = cfg.device
    num_envs = cfg.training.num_dream_envs
    horizon = cfg.training.dream_horizon
    temperature = cfg.world_model.temperature
    risk_aversion = cfg.training.risk_aversion
    
    # Storage for rollout data
    obs_z = torch.zeros(horizon, num_envs, cfg.vae.latent_dim, device=device)
    obs_h = torch.zeros(horizon, num_envs, cfg.world_model.hidden_dim, device=device)
    actions = torch.zeros(horizon, num_envs, dtype=torch.long, device=device)
    log_probs = torch.zeros(horizon, num_envs, device=device)
    rewards = torch.zeros(horizon, num_envs, device=device)
    dones = torch.zeros(horizon, num_envs, device=device)
    ctrl_values = torch.zeros(horizon, num_envs, device=device)  # Controller's value estimates
    wm_values = torch.zeros(horizon, num_envs, device=device)    # World model's risk-adjusted value estimates
    
    # Initialize from real observations
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    
    # Get initial observations for each dream env
    z_batch = []
    for _ in range(num_envs):
        obs, _ = env.reset()
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
            z, _ = vae.encode(obs_t)
            z_batch.append(z)
    env.close()
    
    z = torch.cat(z_batch, dim=0)  # (num_envs, latent_dim)
    hidden = world_model.init_hidden(num_envs, device)
    
    # Collect rollout
    for t in range(horizon):
        h = hidden[0][0]  # (num_envs, hidden_dim)
        
        with torch.no_grad():
            action, log_prob, _, ctrl_value = controller.get_action_and_value(z, h)
            action_oh = F.one_hot(action, NUM_ACTIONS).float()
        
        # Store
        obs_z[t] = z
        obs_h[t] = h
        actions[t] = action
        log_probs[t] = log_prob
        ctrl_values[t] = ctrl_value
        
        # Dream step - returns quantiles for distributional value
        with torch.no_grad():
            next_z, reward, done, wm_quantiles, hidden = world_model.sample(z, action_oh, hidden, temperature)
            # Convert quantiles to risk-adjusted scalar value
            wm_value = world_model.get_risk_adjusted_value(wm_quantiles, risk_aversion)
        
        rewards[t] = reward
        dones[t] = (done > 0.5).float()
        wm_values[t] = wm_value  # Store risk-adjusted value for λ-return computation
        
        z = next_z
        
        # Reset done envs (get new starting observations)
        done_mask = done > 0.5
        if done_mask.any():
            # For simplicity, just resample z from prior for done envs
            num_done = done_mask.sum().item()
            z[done_mask] = torch.randn(num_done, cfg.vae.latent_dim, device=device) * 0.5
            # Reset hidden states for done envs
            hidden[0][:, done_mask] = 0
            hidden[1][:, done_mask] = 0
    
    # Get final values for bootstrapping
    with torch.no_grad():
        h = hidden[0][0]
        _, _, _, final_ctrl_value = controller.get_action_and_value(z, h)
        # Get world model's final value via a dummy forward
        action_oh = F.one_hot(torch.zeros(num_envs, dtype=torch.long, device=device), NUM_ACTIONS).float()
        _, _, _, final_wm_quantiles, _ = world_model.sample(z, action_oh, hidden, temperature=0)
        final_wm_value = world_model.get_risk_adjusted_value(final_wm_quantiles, risk_aversion)
    
    return obs_z, obs_h, actions, log_probs, rewards, dones, ctrl_values, wm_values, final_ctrl_value, final_wm_value


def compute_lambda_returns(rewards, wm_values, dones, final_wm_value, gamma, lam):
    """
    Compute Dreamer-style λ-returns using world model's value estimates.
    
    V_λ(s_t) = r_t + γ * ((1-λ) * V_wm(s_{t+1}) + λ * V_λ(s_{t+1}))
    
    This provides dense training signal even with sparse rewards.
    
    Args:
        rewards: (T, N) rewards from world model
        wm_values: (T, N) world model value estimates
        dones: (T, N) done flags
        final_wm_value: (N,) bootstrap value from world model
        gamma: discount factor
        lam: λ for mixing n-step returns
        
    Returns:
        lambda_returns: (T, N) - dense value targets
    """
    T, N = rewards.shape
    lambda_returns = torch.zeros_like(rewards)
    
    # Bootstrap from world model's value of final state
    next_return = final_wm_value
    
    for t in reversed(range(T)):
        next_non_terminal = 1.0 - dones[t]
        
        # V_λ = r + γ * ((1-λ) * V_wm + λ * V_λ_next)
        if t == T - 1:
            next_wm_value = final_wm_value
        else:
            next_wm_value = wm_values[t + 1]
        
        lambda_returns[t] = rewards[t] + gamma * next_non_terminal * (
            (1 - lam) * next_wm_value + lam * next_return
        )
        next_return = lambda_returns[t]
    
    return lambda_returns


def compute_gae(rewards, values, dones, final_value, gamma, gae_lambda):
    """
    Compute Generalized Advantage Estimation.
    
    Args:
        rewards: (T, N) rewards
        values: (T, N) value estimates
        dones: (T, N) done flags
        final_value: (N,) bootstrap value
        gamma: discount factor
        gae_lambda: GAE lambda
        
    Returns:
        advantages: (T, N)
        returns: (T, N)
    """
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = 0
    
    for t in reversed(range(T)):
        if t == T - 1:
            next_value = final_value
        else:
            next_value = values[t + 1]
        
        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
    
    returns = advantages + values
    return advantages, returns


def save_dream_rollout(vae, world_model, controller, cfg: Config, save_path: str, num_frames: int = 64):
    """
    Generate and save a dream rollout as a video.
    Shows what the world model hallucinates during training.
    
    Args:
        vae: VAE model for decoding latents to images
        world_model: MDN-RNN world model
        controller: Controller policy
        cfg: Config
        save_path: Path to save the video (without extension)
        num_frames: Number of frames to generate
    """
    import imageio
    
    device = cfg.device
    temperature = cfg.world_model.temperature
    
    # Initialize from a real observation
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    obs, _ = env.reset()
    env.close()
    
    with torch.no_grad():
        obs_t = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        z, _ = vae.encode(obs_t)
    
    hidden = world_model.init_hidden(1, device)
    
    frames = []
    rewards_list = []
    actions_list = []
    
    # Generate dream sequence
    with torch.no_grad():
        for t in range(num_frames):
            # Decode current state
            frame = vae.decode(z)
            frame_np = (frame[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            
            # Get action from controller
            h = hidden[0][0]
            action = controller.get_action(z, h, deterministic=False)
            action_oh = F.one_hot(action, NUM_ACTIONS).float()
            
            # Dream step - returns quantiles (we don't use value here, just for stepping)
            next_z, reward, done, wm_quantiles, hidden = world_model.sample(z, action_oh, hidden, temperature)
            
            # Store info
            rewards_list.append(reward.item())
            actions_list.append(action.item())
            
            # Add text overlay with action and reward
            action_names = ['noop', 'left', 'right', 'gas', 'brake']
            text = f"t={t} a={action_names[action.item()]} r={reward.item():.2f}"
            
            # Simple text overlay (just append to frame for now)
            frames.append(frame_np)
            
            z = next_z
            
            # Reset if done
            if done.item() > 0.5:
                # Resample from prior
                z = torch.randn(1, cfg.vae.latent_dim, device=device) * 0.5
                hidden = world_model.init_hidden(1, device)
    
    # Save as video
    video_path = f"{save_path}.mp4"
    imageio.mimsave(video_path, frames, fps=15)
    
    # Also save as a grid of frames for quick viewing
    try:
        from py.torch_utils import dds_from_tensor
        # Create a grid of frames (8x8 = 64 frames)
        grid_size = min(8, int(np.sqrt(num_frames)))
        grid_frames = frames[:grid_size * grid_size]
        
        # Stack into grid
        rows = []
        for i in range(grid_size):
            row = np.concatenate(grid_frames[i*grid_size:(i+1)*grid_size], axis=1)
            rows.append(row)
        grid = np.concatenate(rows, axis=0)
        
        # Save grid as DDS
        grid_tensor = torch.from_numpy(grid).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        dds = dds_from_tensor(grid_tensor)
        dds.save(f"{save_path}_grid.dds")
    except Exception as e:
        print(f"Failed to save dream grid: {e}")
    
    total_reward = sum(rewards_list)
    print(f"Dream rollout saved: {video_path} (total_reward={total_reward:.1f})")
    
    return frames, rewards_list, actions_list


def train_controller(cfg: Config, vae=None, world_model=None):
    """Stage 3: Train controller in dreams using PPO."""
    print("=== Stage 3: Training Controller (PPO in Dreams) ===")
    device = cfg.device
    
    # Load or reuse VAE
    if vae is None:
        vae = VAE(cfg.vae).to(device)
        vae.load_state_dict(torch.load(cfg.checkpoint.vae_path, weights_only=True))
        print(f"Loaded VAE from {cfg.checkpoint.vae_path}")
    vae.eval()
    
    # Load or reuse world model
    if world_model is None:
        world_model = MDNRNN(
            latent_dim=cfg.vae.latent_dim,
            action_dim=NUM_ACTIONS,
            cfg=cfg.world_model
        ).to(device)
        world_model.load_state_dict(torch.load(cfg.checkpoint.world_model_path, weights_only=True))
        print(f"Loaded world model from {cfg.checkpoint.world_model_path}")
    world_model.eval()
    
    # Initialize controller
    controller = Controller(
        latent_dim=cfg.vae.latent_dim,
        hidden_dim=cfg.controller.hidden_dim,
        rnn_hidden_dim=cfg.world_model.hidden_dim,
        action_dim=NUM_ACTIONS
    ).to(device)
    
    if Path(cfg.checkpoint.controller_path).exists() and not cfg.checkpoint.fresh:
        try:
            controller.load_state_dict(torch.load(cfg.checkpoint.controller_path, weights_only=True))
            print(f"Loaded controller from {cfg.checkpoint.controller_path}")
        except Exception as e:
            print(f"Could not load controller: {e}, starting fresh")
    
    num_params = sum(p.numel() for p in controller.parameters())
    print(f"Controller has {num_params} parameters")
    
    risk_mode = "risk-neutral" if cfg.training.risk_aversion <= 0 else f"risk-averse (CVaR α={1-cfg.training.risk_aversion*0.9:.2f})"
    print(f"Training with {risk_mode} value estimation")
    
    optimizer = torch.optim.Adam(controller.parameters(), lr=cfg.training.ctrl_lr)
    
    # Training config
    num_envs = cfg.training.num_dream_envs
    horizon = cfg.training.dream_horizon
    batch_size = cfg.training.ppo_batch_size
    num_minibatches = (num_envs * horizon) // batch_size
    
    best_reward = -float('inf')
    
    for iteration in range(cfg.training.ctrl_iterations):
        # Collect rollouts in dreams
        obs_z, obs_h, actions, old_log_probs, rewards, dones, ctrl_values, wm_values, final_ctrl_value, final_wm_value = \
            collect_dream_rollouts(vae, world_model, controller, cfg)
        
        # Compute Dreamer-style λ-returns using world model's value estimates
        # This gives dense signal even with sparse rewards
        lambda_returns = compute_lambda_returns(
            rewards, wm_values, dones, final_wm_value,
            cfg.training.gamma, cfg.training.lambda_horizon
        )
        
        # Compute advantages: λ-returns - controller's value estimate
        advantages = lambda_returns - ctrl_values
        
        # Flatten batch
        b_obs_z = obs_z.reshape(-1, cfg.vae.latent_dim)
        b_obs_h = obs_h.reshape(-1, cfg.world_model.hidden_dim)
        b_actions = actions.reshape(-1)
        b_old_log_probs = old_log_probs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = lambda_returns.reshape(-1)  # Use λ-returns as value targets
        
        # Normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
        
        # PPO update epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0
        
        for epoch in range(cfg.training.ppo_epochs):
            # Shuffle indices
            indices = torch.randperm(b_obs_z.size(0), device=device)
            
            for start in range(0, b_obs_z.size(0), batch_size):
                end = start + batch_size
                mb_indices = indices[start:end]
                
                mb_z = b_obs_z[mb_indices]
                mb_h = b_obs_h[mb_indices]
                mb_actions = b_actions[mb_indices]
                mb_old_log_probs = b_old_log_probs[mb_indices]
                mb_advantages = b_advantages[mb_indices]
                mb_returns = b_returns[mb_indices]
                
                # Get current policy outputs
                _, new_log_probs, entropy, new_values = controller.get_action_and_value(
                    mb_z, mb_h, mb_actions
                )
                
                # Policy loss (PPO clipped objective)
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - cfg.training.clip_epsilon, 1 + cfg.training.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(new_values, mb_returns)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (
                    policy_loss + 
                    cfg.training.value_coef * value_loss + 
                    cfg.training.entropy_coef * entropy_loss
                )
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(controller.parameters(), cfg.training.max_grad_norm)
                optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1
        
        # Stats
        avg_reward = rewards.sum(dim=0).mean().item()
        avg_policy_loss = total_policy_loss / num_updates
        avg_value_loss = total_value_loss / num_updates
        avg_entropy = total_entropy / num_updates
        
        if avg_reward > best_reward:
            best_reward = avg_reward
            from py.torch_utils import DelayedKeyboardInterrupt
            with DelayedKeyboardInterrupt():
                while True:
                    try:
                        torch.save(controller.state_dict(), cfg.checkpoint.controller_path)
                        break
                    except Exception as e:
                        pass
        
        if iteration % 10 == 0:
            print(f"Iter {iteration+1}/{cfg.training.ctrl_iterations}: "
                  f"reward={avg_reward:.1f}, best={best_reward:.1f}, "
                  f"policy_loss={avg_policy_loss:.4f}, value_loss={avg_value_loss:.4f}, "
                  f"entropy={avg_entropy:.4f}")
        
        # Save visualization periodically
        if iteration % 50 == 0:
            try:
                from py.torch_utils import dds_from_tensor
                with torch.no_grad():
                    # Decode a dream frame
                    z_sample = obs_z[0, 0:1]  # First timestep, first env
                    dream_frame = vae.decode(z_sample)
                    dds = dds_from_tensor(dream_frame)
                    dds.save(f".tmp/dream_iter{iteration:04d}.dds")
                
                # Save a full dream rollout video
                Path(".tmp/dreams").mkdir(parents=True, exist_ok=True)
                save_dream_rollout(
                    vae, world_model, controller, cfg,
                    save_path=f".tmp/dreams/dream_iter{iteration:04d}",
                    num_frames=64
                )
            except Exception as e:
                print(f"Failed to save dream visualization: {e}")
        
        # Periodic checkpoint
        if iteration % cfg.training.save_interval == 0 and iteration > 0:
            from py.torch_utils import DelayedKeyboardInterrupt
            with DelayedKeyboardInterrupt():
                while True:
                    try:
                        torch.save(controller.state_dict(), cfg.checkpoint.controller_path)
                        break
                    except Exception as e:
                        pass
    
    print(f"Controller saved to {cfg.checkpoint.controller_path}")
    return vae, world_model, controller


def generate_video(cfg: Config, vae=None, world_model=None, controller=None):
    """Generate video of trained agent in real environment."""
    print("=== Generating Video ===")
    device = cfg.device
    
    # Load or reuse models
    if vae is None:
        vae = VAE(cfg.vae).to(device)
        vae.load_state_dict(torch.load(cfg.checkpoint.vae_path, weights_only=True))
    vae.eval()
    
    if world_model is None:
        world_model = MDNRNN(
            latent_dim=cfg.vae.latent_dim,
            action_dim=NUM_ACTIONS,
            cfg=cfg.world_model
        ).to(device)
        world_model.load_state_dict(torch.load(cfg.checkpoint.world_model_path, weights_only=True))
    world_model.eval()
    
    if controller is None:
        controller = Controller(
            latent_dim=cfg.vae.latent_dim,
            hidden_dim=cfg.controller.hidden_dim,
            rnn_hidden_dim=cfg.world_model.hidden_dim,
            action_dim=NUM_ACTIONS
        ).to(device)
        controller.load_state_dict(torch.load(cfg.checkpoint.controller_path, weights_only=True))
    controller.eval()
    
    print("Models ready")
    
    # Create environment with video recording
    Path(".tmp/videos").mkdir(parents=True, exist_ok=True)
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    env = RecordVideo(env, video_folder=".tmp/videos", episode_trigger=lambda x: True)
    
    obs, _ = env.reset()
    hidden = world_model.init_hidden(1, device)
    total_reward = 0
    
    for step in range(1000):
        # Encode observation
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
            z, _ = vae.encode(obs_t)
            
            # Get RNN hidden
            h = hidden[0][0]  # (1, hidden_dim)
            
            # Get action
            action = controller.get_action(z, h, deterministic=True)
            action_oh = F.one_hot(action, NUM_ACTIONS).float()
            
            # Update hidden state (even though we're in real env, we need to track it)
            _, _, _, _, _, _, hidden = world_model(z.unsqueeze(1), action_oh.unsqueeze(1), hidden)
            
            # Visualize what the world model predicts vs reality
            if step % 50 == 0:
                try:
                    from py.torch_utils import dds_from_tensor
                    # Get world model prediction
                    next_z_pred, _, _, _, _ = world_model.sample(z, action_oh, hidden, temperature=0.0)
                    recon_pred = vae.decode(next_z_pred)
                    recon_actual = vae.decode(z)
                    
                    dds = dds_from_tensor(recon_pred)
                    dds.save(f".tmp/video_pred_t{step:04d}.dds")
                    dds = dds_from_tensor(recon_actual)
                    dds.save(f".tmp/video_actual_t{step:04d}.dds")
                    dds = dds_from_tensor(obs_t)
                    dds.save(f".tmp/video_real_t{step:04d}.dds")
                except Exception as e:
                    print(f"Failed to save DDS: {e}")
        
        # Step real environment
        env_action = action_to_env(action.item())
        obs, reward, terminated, truncated, _ = env.step(env_action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Episode finished. Total reward: {total_reward:.1f}")
            break
    
    env.close()
    print("Video saved to .tmp/videos/")


def train_all(cfg: Config):
    """Run all training stages in an outer loop."""
    print("=== Running Full Training Pipeline ===\n")
    
    vae = None
    world_model = None
    controller = None
    
    for iteration in range(cfg.training.num_major_iterations):
        print(f"\n{'='*60}")
        print(f"=== Major Iteration {iteration+1}/{cfg.training.num_major_iterations} ===")
        print(f"{'='*60}\n")
        
        # Stage 1: Collect data (using current policy if available) and train VAE
        vae = train_vae(cfg, vae, world_model, controller)
        print()
        
        # Stage 2: Train world model on all collected data
        vae, world_model = train_world_model(cfg, vae)
        print()
        
        # Stage 3: Train controller in dreams
        vae, world_model, controller = train_controller(cfg, vae, world_model)
        print()
        
        # Save a dream rollout to visualize what the agent imagines
        try:
            Path(".tmp/dreams").mkdir(parents=True, exist_ok=True)
            save_dream_rollout(
                vae, world_model, controller, cfg,
                save_path=f".tmp/dreams/major_iter{iteration+1:03d}",
                num_frames=128  # Longer rollout for major iterations
            )
        except Exception as e:
            print(f"Failed to save dream rollout: {e}")
        
        # Generate video to see progress in real environment
        generate_video(cfg, vae, world_model, controller)
        print()
    
    print("=== Training Complete ===")


def generate_dream(cfg: Config, vae=None, world_model=None, controller=None):
    """Generate and save a dream rollout visualization."""
    print("=== Generating Dream Rollout ===")
    device = cfg.device
    
    # Load models
    if vae is None:
        vae = VAE(cfg.vae).to(device)
        vae.load_state_dict(torch.load(cfg.checkpoint.vae_path, weights_only=True))
    vae.eval()
    
    if world_model is None:
        world_model = MDNRNN(
            latent_dim=cfg.vae.latent_dim,
            action_dim=NUM_ACTIONS,
            cfg=cfg.world_model
        ).to(device)
        world_model.load_state_dict(torch.load(cfg.checkpoint.world_model_path, weights_only=True))
    world_model.eval()
    
    if controller is None:
        controller = Controller(
            latent_dim=cfg.vae.latent_dim,
            hidden_dim=cfg.controller.hidden_dim,
            rnn_hidden_dim=cfg.world_model.hidden_dim,
            action_dim=NUM_ACTIONS
        ).to(device)
        controller.load_state_dict(torch.load(cfg.checkpoint.controller_path, weights_only=True))
    controller.eval()
    
    Path(".tmp/dreams").mkdir(parents=True, exist_ok=True)
    save_dream_rollout(
        vae, world_model, controller, cfg,
        save_path=".tmp/dreams/dream_latest",
        num_frames=128
    )


@hydra.main(version_base=None, config_name="config")
def main(cfg: Config):
    print(OmegaConf.to_yaml(cfg))
    Path(".tmp").mkdir(exist_ok=True)
    
    commands = {
        "train_vae": train_vae,
        "train_world_model": train_world_model,
        "train_controller": train_controller,
        "generate_video": generate_video,
        "generate_dream": generate_dream,
        "train_all": train_all,
    }
    
    if cfg.command in commands:
        commands[cfg.command](cfg)
    else:
        print(f"Unknown command: {cfg.command}")
        print(f"Available: {', '.join(commands.keys())}")


if __name__ == '__main__':
    main()