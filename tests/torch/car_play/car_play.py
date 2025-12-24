#!/usr/bin/env python3
# MIT License
# Copyright (c) 2025 Anton Schreiner

"""
Toy Character-Level LLM (ASCII-128)
A minimal GPT-style transformer that operates on raw ASCII characters.
No tokenizer needed - just ord() and chr().

Usage with Hydra (no YAML needed - Python dataclasses are the single source of truth):
    # Train on codebase
    python toy_llm.py command=train data.data_dir=/path/to/code
    
    # Train on Wikipedia
    python toy_llm.py command=train data.wiki=true
    
    # Train on brackets
    python toy_llm.py command=train data.brackets=true
    
    # Override model params
    python toy_llm.py command=train data.brackets=true model.n_layers=2 model.d_model=512
    
    # Generate
    python toy_llm.py command=generate generate.prompt="def "
    
    # Fresh start
    python toy_llm.py command=train data.data_dir=. checkpoint.fresh=true
    
    # Disable Hydra output directory
    python toy_llm.py command=train data.brackets=true hydra.run.dir=.
"""

import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import math
import time
from dataclasses import dataclass, field
from typing import Optional, List
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.config_store import ConfigStore
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from einops import rearrange, reduce, repeat
from piq import SSIMLoss, LPIPS

@dataclass
class ModelConfig:
    dropout      : float    = 0.1


@dataclass
class TrainingConfig:
    start_iteration     : int       = 0
    num_train_steps     : int       = 100000
    batch_size          : int       = 32
    learning_rate       : float     = 3e-4
    weight_decay        : float     = 0.1
    max_iters           : int       = 10000
    eval_interval       : int       = 500
    eval_iters          : int       = 50
    warmup_iters        : int       = 100
    min_lr              : float     = 1e-5
    grad_clip           : float     = 1.0
    train_split         : float     = 0.9
    compile_model       : bool      = False


@dataclass
class AttentionConfig:
    n_heads      : int      = 8
    d_model      : int      = 128
    d_ff         : int      = 512
    dropout      : float    = 0.1

@dataclass
class VAETransformerConfig:
    patch_dim       : int       = 8
    image_size      : int       = 96
    latent_dim      : int       = 128
    num_layers      : int       = 8
    attention_config : AttentionConfig = field(default_factory=AttentionConfig)

@dataclass
class VAECNNConfig:
    latent_dim      : int       = 128

@dataclass
class DataConfig:
    pass


@dataclass
class CheckpointConfig:
    path                : str       = ".tmp/toy_llm.pt"
    fresh               : bool      = False


@dataclass
class GenerateConfig:
    pass


@dataclass 
class Config:
    """Full config combining all sub-configs."""
    model               : ModelConfig = field(default_factory=ModelConfig)
    training            : TrainingConfig = field(default_factory=TrainingConfig)
    data                : DataConfig = field(default_factory=DataConfig)
    checkpoint          : CheckpointConfig = field(default_factory=CheckpointConfig)
    generate            : GenerateConfig = field(default_factory=GenerateConfig)
    vae_cnn             : VAECNNConfig = field(default_factory=VAECNNConfig)
    vae_transformer     : VAETransformerConfig = field(default_factory=VAETransformerConfig)
    command             : str = "train"
    device              : str = "cuda" if torch.cuda.is_available() else "cpu"


# Register with Hydra's ConfigStore - this replaces the YAML file
cs = ConfigStore.instance()
cs.store(name="config", node=Config)

def train(cfg: Config):
    print("Training not yet implemented.")
    import gymnasium as gym

    env = gym.make("CarRacing-v3", render_mode="human")
    obs, info = env.reset()

    for _ in range(1000):
        action = env.action_space.sample()  # random actions
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()

    env.close()


class RMSNorm(nn.Module):
    """Root Mean Square Normalization (simpler than LayerNorm)"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE)"""
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)
    
    def forward(self, seq_len: int):
        if seq_len > self.cos_cached.size(0):
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class SelfAttention(nn.Module):
    def __init__(self, cfg: AttentionConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        
        self.n_heads    = cfg.n_heads
        self.head_dim   = cfg.d_model // cfg.n_heads
        self.qkv        = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out        = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout    = nn.Dropout(cfg.dropout)
        self.rope       = RotaryEmbedding(self.head_dim, cfg.max_seq_len)
    
    def forward(self, x, causal: bool = False):
        B, T, C = x.shape
        
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2) for t in qkv]
        
        # Apply RoPE
        cos, sin = self.rope(T)
        cos, sin = cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)
        q, k = apply_rope(q, k, cos, sin)
        
        # Scaled dot-product attention with causal mask
        out = F.scaled_dot_product_attention(q, k, v, is_causal=causal, dropout_p=self.dropout.p if self.training else 0.0)
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.out(out))
    
class MLP(nn.Module):
    """SwiGLU-style MLP"""
    def __init__(self, cfg: AttentionConfig):
        super().__init__()
        self.w1         = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w2         = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.w3         = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.dropout    = nn.Dropout(cfg.dropout)
    
    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: AttentionConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg)
        self.attn  = SelfAttention(cfg)
        self.norm2 = RMSNorm(cfg)
        self.mlp   = MLP(cfg)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class EncoderBlock(nn.Module):
    def __init__(self, channels, groups=1, skip=True):
        super(EncoderBlock, self).__init__()
        self.dwconv    = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=True, groups=groups)
        self.pwconv1   = nn.Conv2d(in_channels=channels, out_channels=channels * 2, kernel_size=1, stride=1, padding=0, bias=True)
        # self.bn1       = nn.BatchNorm2d(channels * 2)
        # self.act       = nn.GELU()
        self.act       = nn.LeakyReLU(0.01, inplace=True)
        self.pwconv2   = nn.Conv2d(in_channels=channels * 2, out_channels=channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.skip      = skip

    def forward(self, x):
        # x        = self.bn1(x)
        identity = x
        out      = self.dwconv(x)
        out      = self.pwconv1(out)
        # out      = self.bn1(out)
        out      = self.act(out)
        out      = self.pwconv2(out)
        if self.skip:
            out      = out + identity
        return out

class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = self.conv(x)
        return x

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = self.conv(x)
        return x

class VAETransformer(nn.Module):
    def __init__(self, cfg: VAETransformerConfig):
        super(VAETransformer, self).__init__()

        for key, value in cfg.__dict__.items():
            # print(f"VAE Config: {key} = {value}")
            setattr(self, key, value)

        # self.image_size = cfg.vae.image_size
        # self.latent_dim = cfg.vae.latent_dim
        # self.patch_size = cfg.vae.patch_size

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_channels=3, out_channels=self.latent_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.layers      = nn.ModuleList([TransformerBlock(cfg.attention_config) for _ in range(cfg.num_layers)])

        self.patch_prediction_head = nn.ModuleList([
            nn.Linear(in_features=self.latent_dim, out_features=self.latent_dim),
            nn.GELU(),
            nn.Linear(in_features=self.latent_dim, out_features=self.patch_size * self.patch_size * 3),
            nn.GELU(),
        ])
        
    def autoencode(self, x):
        # patchify
        B, C, H, W = x.shape

        # map to 2d image of patches
        x = self.patch_embed(x)  # (B, latent_dim, H/patch_size, W/patch_size)
        # flatten patches
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, latent_dim)

        encoded_token = torch.zeros((B, 1, self.latent_dim), device=x.device)
        x = torch.cat([encoded_token, x], dim=1)  # (B, num_patches + 1, latent_dim)

        for layer in self.layers:
            x = layer(x)

        # predict patches
        patch_preds = []
        # todo
    
class VAECNN(nn.Module):
    def __init__(self, cfg: VAECNNConfig):
        super(VAECNN, self).__init__()

        for key, value in cfg.__dict__["_content"].items():
            # import omegaconf.nodes
            # isinstance(value,
            # omegaconf.nodes.IntegerNode
            # print(f"VAE Config: {key} = {value._val} ({type(value._val)})")
            setattr(self, key, value._val)

        self.encoder = nn.Sequential(
            # 96x96
            DownsampleBlock(in_channels=3, out_channels=16), # 48x48
            EncoderBlock(channels=16),
            DownsampleBlock(in_channels=16, out_channels=32), # 24x24
            EncoderBlock(channels=32),
            DownsampleBlock(in_channels=32, out_channels=64), # 12x12
            EncoderBlock(channels=64),
            DownsampleBlock(in_channels=64, out_channels=128), # 6x6
            EncoderBlock(channels=128),
            DownsampleBlock(in_channels=128, out_channels=128), # 3x3
        )
        self.final_projection = nn.Linear(128 * 3 * 3, self.latent_dim)
        self.fc_mu            = nn.Linear(self.latent_dim, self.latent_dim)
        self.fc_logvar        = nn.Linear(self.latent_dim, self.latent_dim)

        self.decoder_input = nn.Linear(self.latent_dim, 128 * 3 * 3)
        self.decoder = nn.Sequential(
            UpsampleBlock(in_channels=128, out_channels=128), # 6x6
            EncoderBlock(channels=128),
            UpsampleBlock(in_channels=128, out_channels=64), # 12x12
            EncoderBlock(channels=64),
            UpsampleBlock(in_channels=64, out_channels=32), # 24x24
            EncoderBlock(channels=32),
            UpsampleBlock(in_channels=32, out_channels=16), # 48x48
            EncoderBlock(channels=16),
            UpsampleBlock(in_channels=16, out_channels=3), # 96x96
            nn.Sigmoid()
        )

    def encode(self, x):
        B = x.size(0)
        x = self.encoder(x)
        x = x.view(B, -1)
        x = self.final_projection(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        B = z.size(0)
        x = self.decoder_input(z)
        x = x.view(B, 128, 3, 3)
        x = self.decoder(x)
        return x

@hydra.main(version_base=None, config_name="config")
def main(cfg: Config):
    # Print config
    print(OmegaConf.to_yaml(cfg))
    
    if cfg.command == "train":
        train(cfg)
    elif cfg.command == "train_vae":
        vae = VAECNN(cfg.vae_cnn).to(cfg.device)
        optimizer = torch.optim.AdamW(
            vae.parameters(),
            lr=cfg.training.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=cfg.training.weight_decay
        )
        try: # Load the model
            vae.load_state_dict(torch.load(".tmp/vae_cnn.pth"), strict=False)
        except Exception as e:
            print("No model found, starting from scratch.")

        env = gym.make("CarRacing-v3", render_mode="rgb_array")

        lpips_loss = LPIPS().to(cfg.device)

        obs, info = env.reset()
        for epoch in range(cfg.training.num_train_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            image = torch.from_numpy(obs).clone().permute(2, 0, 1).unsqueeze(0).contiguous().float().to(cfg.device) / 255.0

            orig = image

            mu, logvar      = vae.encode(image)
            z               = vae.reparameterize(mu, logvar)
            recon_image     = vae.decode(z)
            loss_recon      = F.mse_loss(recon_image, image) + lpips_loss(recon_image, image)
            loss_kld        = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss            = loss_recon + 0.0001 * loss_kld
            
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            print(f"Step {epoch}: Recon Loss = {loss_recon.item():.4f}, KLD Loss = {loss_kld.item():.4f}")

            if epoch % 16 == 0:
                from py.torch_utils import dds_from_tensor, DelayedKeyboardInterrupt
                with DelayedKeyboardInterrupt():
                    while True:
                        try:
                            torch.save(vae.state_dict(), f".tmp/vae_cnn.pth")
                            break
                        except Exception as e:
                            pass # Ignore file access issues or keyboard interrupts
                try:
                    dds = dds_from_tensor(recon_image[0:1])
                    dds.save(".tmp/recon.dds")
                    dds = dds_from_tensor(orig[0:1])
                    dds.save(".tmp/orig.dds")
                except Exception as e:
                    print(f"Failed to save DDS: {e}")

            if terminated or truncated:
                obs, info = env.reset()

        env.close()

    elif cfg.command == "generate_video":
        

        env = gym.make("CarRacing-v3", render_mode="rgb_array")
        env = RecordVideo(env, video_folder="./.tmp/videos", episode_trigger=lambda x: True)

        obs, info = env.reset()
        for _ in range(1000):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()

        env.close()
    else:
        print(f"Unknown command: {cfg.command}")
        print("Use command=train or command=generate")


if __name__ == '__main__':
    main()