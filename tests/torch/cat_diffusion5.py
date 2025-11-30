# MIT License
# Copyright (c) 2025 Anton Schreiner

import torch
import torch.nn as nn
import os
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import random
from py.torch_utils import dds_from_tensor, DelayedKeyboardInterrupt
from piq import SSIMLoss, LPIPS
import math
import numpy as np
from pytorch_optimizer import Muon, SOAP, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, PolynomialLR
import argparse

args = argparse.ArgumentParser()
args.add_argument("--freeze_vae", default=False, action="store_true", help="Freeze VAE weights during training")
args = args.parse_args()

assert torch.cuda.is_available()
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(torch.cuda.current_device()))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

LATENT_DIM = 3
KPN_FEATURE_DIM = 1 * 5 * 5
PATCH_DIM = 1

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

class Swiglu(nn.Module):
    def __init__(self, num_channels):
        super(Swiglu, self).__init__()
        self.act = nn.SiLU()
        self.ffw_0 = nn.Conv2d(num_channels, num_channels, kernel_size=1)
        self.ffw_1 = nn.Conv2d(num_channels, num_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.ffw_0(x)
        x2 = self.ffw_1(x)
        return x1 * self.act(x2)

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.num_heads  = num_heads
        self.head_dim   = embed_dim // num_heads
        self.scale      = self.head_dim ** -0.5
        
        self.q      = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
        )
        self.k      = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
        )
        self.v      = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
        )
        self.fc     = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
        )
    
        self.last_scores = None

    def forward(self, x):
        B, C, H, W = x.shape
        
        # x = apply_rope_2d(x)

        q, k, v = self.q(x), self.k(x), self.v(x)
        
        # k = apply_rope_2d(k)
        # q = apply_rope_2d(q)

        # W*H - total number of tokens
        q = q.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)  # [B, num_heads, H*W, head_dim]
        k = k.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)  # [B, num_heads, H*W, head_dim] 
        v = v.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)  # [B, num_heads, H*W, head_dim]
        
        # [B, num_heads, H*W, H*W]
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(scores, dim=-1)
        
        # [B, num_heads, H*W, head_dim]
        out = torch.matmul(attn, v)
        
        # [B, num_heads, head_dim, H*W] then [B, C, H, W]
        out = out.permute(0, 1, 3, 2).contiguous().view(B, C, H, W)
        
        self.last_scores = attn
        return self.fc(out)

class LinearAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(LinearAttention, self).__init__()
        self.num_heads  = num_heads
        self.head_dim   = embed_dim // num_heads
        self.scale      = self.head_dim ** -0.5
        
        self.q      = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.k      = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.v      = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.fc     = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
    
        self.last_scores = None

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.q(x), self.k(x), self.v(x)
        
        k = apply_rope_2d(k)
        q = apply_rope_2d(q)

        # Reshape to [B, num_heads, H*W, head_dim]
        q = q.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        k = k.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        v = v.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        
        # Scale queries
        q = q * self.scale
        
        # Apply kernel function (ELU+1 for positivity)
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        
        # Linear attention: compute K^T V then Q(K^T V)
        kv = torch.matmul(k.transpose(-1, -2), v)  # [B, num_heads, head_dim, head_dim]
        
        # Normalize by sum of keys for each query
        k_sum = k.sum(dim=-2, keepdim=True)  # [B, num_heads, 1, head_dim]
        normalizer = torch.matmul(q, k_sum.transpose(-1, -2))  # [B, num_heads, H*W, 1]
        
        out = torch.matmul(q, kv)  # [B, num_heads, H*W, head_dim]
        out = out / (normalizer + 1e-6)  # Normalize
        
        # Reshape back
        out = out.permute(0, 1, 3, 2).contiguous().view(B, C, H, W)
        self.last_scores = None
        return self.fc(out)


class SelfAttentionBlock(nn.Module):
    def __init__(self, options, embed_dim, num_heads):
        super(SelfAttentionBlock, self).__init__()
        self.options = [options]
        # self.attn   = LinearAttention(embed_dim, num_heads)
        self.attn   = SelfAttention(embed_dim, num_heads)
        self.ffw    = nn.Sequential(
            Swiglu(embed_dim),
        )
        self.norm1  = nn.LayerNorm(embed_dim)
        self.norm2  = nn.LayerNorm(embed_dim)

    def forward(self, _x):
        x = _x
        x = x + self.attn(self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        x = x + self.ffw(self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        y = x.contiguous()
        if self.options[0].quantized:
            y = y.to(dtype=torch.float8_e4m3fn).to(dtype=torch.float32)
        return y

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device     = time.device
        half_dim   = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def apply_rope_2d(x, theta=10000.0):
    B, C, H, W = x.shape
    assert C % 4 == 0, "Channel dimension must be divisible by 4 for 2D RoPE"
    
    device = x.device
    dtype = x.dtype
    
    # Split channels: half for height, half for width
    dim_h = C // 2
    dim_w = C // 2
    
    # Compute frequencies for height
    freqs_h = 1.0 / (theta ** (torch.arange(0, dim_h, 2, device=device, dtype=dtype) / dim_h))
    pos_h = torch.arange(H, device=device, dtype=dtype)
    freqs_h = torch.outer(pos_h, freqs_h)  # [H, dim_h//2]
    
    # Compute frequencies for width
    freqs_w = 1.0 / (theta ** (torch.arange(0, dim_w, 2, device=device, dtype=dtype) / dim_w))
    pos_w = torch.arange(W, device=device, dtype=dtype)
    freqs_w = torch.outer(pos_w, freqs_w)  # [W, dim_w//2]
    
    # Create 2D frequency grids
    freqs_h_2d = freqs_h[:, None, :].expand(H, W, -1)  # [H, W, dim_h//2]
    freqs_w_2d = freqs_w[None, :, :].expand(H, W, -1)  # [H, W, dim_w//2]
    
    # Flatten to sequence format
    freqs_h_flat = freqs_h_2d.reshape(H * W, -1)  # [H*W, dim_h//2]
    freqs_w_flat = freqs_w_2d.reshape(H * W, -1)  # [H*W, dim_w//2]
    
    # Reshape x to [B, H*W, C]
    x_seq = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
    
    # Split channels
    x_h, x_w = x_seq.split([dim_h, dim_w], dim=-1)  # Each [B, H*W, C//2]
    
    # Helper function to rotate half
    def rotate_half(t):
        t1, t2 = t.chunk(2, dim=-1)
        return torch.cat((-t2, t1), dim=-1)
    
    # Helper function to apply rotary embedding
    def apply_rotary(t, freqs):
        cos = freqs.cos()
        sin = freqs.sin()
        # Interleave cos and sin: [seq_len, dim//2] -> [seq_len, dim]
        cos = torch.stack([cos, cos], dim=-1).flatten(-2)
        sin = torch.stack([sin, sin], dim=-1).flatten(-2)
        return t * cos + rotate_half(t) * sin
    
    # Apply rotary embeddings
    x_h = apply_rotary(x_h, freqs_h_flat)
    x_w = apply_rotary(x_w, freqs_w_flat)
    
    # Concatenate and reshape back to [B, C, H, W]
    x_out = torch.cat([x_h, x_w], dim=-1)
    x_out = x_out.transpose(1, 2).reshape(B, C, H, W)
    
    return x_out

class ROPE(nn.Module):
    def __init__(self):
        super(ROPE, self).__init__()

    def forward(self, x):
        return apply_rope_2d(x)

class StaticTokens(nn.Module):
    def __init__(self, dim, num_tokens):
        super(StaticTokens, self).__init__()

        self.tokens = nn.Parameter(torch.randn(1, dim, 1, num_tokens))

    def forward(self, x):
        batch_size = x.size(0)

        batched = self.tokens.expand(batch_size, -1, -1, -1)

        return torch.cat([x, batched], dim=3)

class RemoveStaticTokens(nn.Module):
    def __init__(self, num_tokens):
        super(RemoveStaticTokens, self).__init__()
        self.num_tokens = num_tokens

    def forward(self, x):
        return x[:, :, :, :-self.num_tokens]

class ImageToTokens(nn.Module):
    def __init__(self):
        super(ImageToTokens, self).__init__()

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, 1, H * W) # B, C, 1, N

        return x

class TokensToImage(nn.Module):
    def __init__(self):
        super(TokensToImage, self).__init__()

    def forward(self, x):
        B, C, H, W = x.shape
        W = H = int(math.sqrt(W))
        x = x.view(B, C, H, W)  # B, H, W, C
        return x

class InferenceQuantization(nn.Module):
    def __init__(self, options):
        super(InferenceQuantization, self).__init__()
        self.options = [options]

    def forward(self, x):
        if self.options[0].quantized:
            return x.to(dtype=torch.float8_e4m3fn).to(dtype=torch.float32)
        return x

class FilterNetworkDownsample(nn.Module):
    def __init__(self):
        super(FilterNetworkDownsample, self).__init__()

    def forward(self, x, kernel_params):
        B, C, H, W = x.shape
        device = x.device

        image         = x  # RGB image
        input         = x  # RGB image
        
        # kernel_params = torch.sigmoid(kernel_params)
        # kernel_params = kernel_params / (kernel_params.abs().sum(dim=1, keepdim=True) + 1e-6)
        kernel_params = torch.tanh(kernel_params)
        weights       = kernel_params  # 5x5 = 25 weights per output pixel
        patches       = F.unfold(image, kernel_size=5, padding=2)  # Shape: (B, 75, H*W)
        patches       = patches.view(B, 3, 25, H*W)  # (B, RGB, 5x5, H*W)
        weights       = weights.view(B, 1, 25, H*W).expand(B, 3, 25, H*W)  # (B, 3, 5x5, H*W)
        result        = (patches * weights).sum(dim=[2])  # Sum over RGB and 5x5
        result        = result.view(B, 3, H, W)
        downsample    = F.avg_pool2d(result, kernel_size=2, stride=2)
        return result, downsample
    
        # precision matrix form (inverse of covariance)
        # https://arxiv.org/pdf/2112.09318
        # /x\T  / kx * kx        | kxy * kx * ky\   /x\
        # \y/   \ kxy * kx * ky  | ky * ky      /   \y/
        # = { x * kx * kx + y * kxy * kx * ky,  x * kxy * kx * ky + y * ky * ky } dot { x, y}
        # = x * x * kx * kx + x * y * kxy * kx * ky + y * x * kxy * kx * ky + y * y * ky * ky
        # = x^2 * kx^2 + 2xy * kxy * kx * ky + y^2 * ky^2 
        
        kx  = torch.sigmoid(kernel_params[:, 0:1, :, :]) * 4.0
        ky  = torch.sigmoid(kernel_params[:, 1:2, :, :]) * 4.0
        kxy = torch.tanh(kernel_params[:, 2:3, :, :]) * 1.0

        # Gaussian filtering using precision matrix
        xy               = torch.zeros(B, 2, H, W, device=device)
        x                = torch.arange(0, W, 1, dtype=torch.float32, device=device).view(1, 1, 1, W).expand(B, 1, H, W)
        y                = torch.arange(0, H, 1, dtype=torch.float32, device=device).view(1, 1, H, 1).expand(B, 1, H, W)
        xy[:, 0:1, :, :] = (x + 0.5)
        xy[:, 1:2, :, :] = (y + 0.5)
        uv               = xy / torch.tensor([W, H], device=device).view(1, 2, 1, 1)

        acc         = torch.zeros(B, 3, H, W, device=device)
        wacc        = torch.zeros(B, 1, H, W, device=device)
        kernel_size = 5
        for iy in range(kernel_size):
            for ix in range(kernel_size):
                _x     = ix - kernel_size // 2
                _y     = iy - kernel_size // 2
                coord  = uv * torch.tensor([W, H], device=device).view(1, 2, 1, 1) + torch.tensor([_x, _y], device=device).view(1, 2, 1, 1)
                coord  = torch.floor(coord) + 0.5
                ndc    = coord / torch.tensor([W, H], device=device).view(1, 2, 1, 1) * 2.0 - 1.0
                sample = torch.nn.functional.grid_sample(input, ndc.permute(0, 2, 3, 1), mode='nearest', padding_mode='border', align_corners=False)
                dx     = (uv[:, 0:1, :, :] * W - coord[:, 0:1, :, :])
                dy     = (uv[:, 1:2, :, :] * H - coord[:, 1:2, :, :])
                dist   = (dx * dx * kx * kx + dy * dy * ky * kx + 2.0 * dx * dy  * kx * ky * kxy)
                # w      = 2.0 * torch.exp(-2.0 * dist) - torch.exp(-4.0 * dist)
                w      = torch.exp(-2.0 * dist)
                # w      = lanczos_kernel(-1.0 * (dx * dx * kx + dy * dy * ky + 2.0 * dx * dy * kxy), a=2.0)

                acc    = acc + sample * w
                wacc   = wacc + w.abs()
        result = acc / (wacc + 1e-6)

        downsample    = F.avg_pool2d(result, kernel_size=2, stride=2)
        
        return result, downsample

def lanczos_kernel(x, a):
    torch.where(x.abs() < 1e-6, torch.tensor(1e-6, device=x.device), x)
    sinc_x = torch.sin(math.pi * x) / (math.pi * x)
    sinc_x_a = torch.sin(math.pi * x / a) / (math.pi * x / a)
    lanczos = sinc_x * sinc_x_a
    lanczos = torch.where(torch.abs(x) < a, lanczos, torch.zeros_like(x))
    return lanczos


class FilterNetworkUpsample(nn.Module):
    def __init__(self):
        super(FilterNetworkUpsample, self).__init__()

    def forward(self, input, upscale_params, skip_signal, skip_param):
        B, C, H, W = input.shape
        device = input.device
        skip_param    = torch.sigmoid(skip_param)
        
        # precision matrix form (inverse of covariance)
        # https://arxiv.org/pdf/2112.09318
        # /x\T  / kx * kx        | kxy * kx * ky\   /x\
        # \y/   \ kxy * kx * ky  | ky * ky      /   \y/
        # = { x * kx * kx + y * kxy * kx * ky,  x * kxy * kx * ky + y * ky * ky } dot { x, y}
        # = x * x * kx * kx + x * y * kxy * kx * ky + y * x * kxy * kx * ky + y * y * ky * ky
        # = x^2 * kx^2 + 2xy * kxy * kx * ky + y^2 * ky^2 
        
        kx  = torch.sigmoid(upscale_params[:, 0:1, :, :]) * 4.0
        ky  = torch.sigmoid(upscale_params[:, 1:2, :, :]) * 4.0
        kxy = torch.tanh(upscale_params[:, 2:3, :, :]) * 1.0

        # Gaussian upscaling using precision matrix
        xy               = torch.zeros(B, 2, H*2, W*2, device=device)
        x                = torch.arange(0, W*2, 1, dtype=torch.float32, device=device).view(1, 1, 1, W*2).expand(B, 1, H*2, W*2)
        y                = torch.arange(0, H*2, 1, dtype=torch.float32, device=device).view(1, 1, H*2, 1).expand(B, 1, H*2, W*2)
        xy[:, 0:1, :, :] = (x + 0.5)
        xy[:, 1:2, :, :] = (y + 0.5)
        uv               = xy / torch.tensor([W*2, H*2], device=device).view(1, 2, 1, 1)

        acc  = torch.zeros(B, 3, H*2, W*2, device=device)
        wacc = torch.zeros(B, 1, H*2, W*2, device=device)
        kernel_size = 5
        for iy in range(kernel_size):
            for ix in range(kernel_size):
                _x     = ix - kernel_size // 2
                _y     = iy - kernel_size // 2
                coord  = uv * torch.tensor([W, H], device=device).view(1, 2, 1, 1) + torch.tensor([_x, _y], device=device).view(1, 2, 1, 1)
                coord  = torch.floor(coord) + 0.5
                ndc    = coord / torch.tensor([W, H], device=device).view(1, 2, 1, 1) * 2.0 - 1.0
                sample = torch.nn.functional.grid_sample(input, ndc.permute(0, 2, 3, 1), mode='nearest', padding_mode='border', align_corners=False)
                dx     = (uv[:, 0:1, :, :] * W - coord[:, 0:1, :, :])
                dy     = (uv[:, 1:2, :, :] * H - coord[:, 1:2, :, :])
                dist   = (dx * dx * kx * kx + dy * dy * ky * kx + 2.0 * dx * dy  * kx * ky * kxy)
                # w      = 2.0 * torch.exp(-2.0 * dist) - torch.exp(-4.0 * dist)
                w      = torch.exp(-2.0 * dist)
                # w      = torch.exp(-2.0 * (dx * dx * kx * kx + dy * dy * ky * kx + 2.0 * dx * dy  * kx * ky * kxy))
                # w      = lanczos_kernel(-1.0 * (dx * dx * kx + dy * dy * ky + 2.0 * dx * dy * kxy), a=2.0)

                acc    = acc + sample * w
                wacc   = wacc + w.abs()
        upscaled = acc / (wacc + 1e-6)

        return torch.lerp(upscaled, skip_signal, skip_param)

class Diffusion(nn.Module):
    def __init__(self):
        super(Diffusion, self).__init__()

        time_dim = 16
        EXPAND_DIM = 512
        self.expand_dim = EXPAND_DIM

        self.time_dim   = time_dim
        self.time_embed = SinusoidalPositionalEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
        )
        self.quantized = False

        self.expand = nn.Conv2d(LATENT_DIM, time_dim, kernel_size=1, stride=1, padding=0)

        self.wdrift = nn.Sequential(*[
            DownsampleBlock(in_channels=LATENT_DIM, out_channels=16),
            EncoderBlock(channels=16),
            DownsampleBlock(in_channels=16, out_channels=16),
            EncoderBlock(channels=16),
            DownsampleBlock(in_channels=16, out_channels=16),
            EncoderBlock(channels=16),
            
            # nn.Conv2d(16, LATENT_DIM, kernel_size=1, stride=1, padding=0),

            UpsampleBlock(in_channels=16, out_channels=16),
            EncoderBlock(channels=16),
            UpsampleBlock(in_channels=16, out_channels=16),
            EncoderBlock(channels=16),
            UpsampleBlock(in_channels=16, out_channels=LATENT_DIM),

            # EncoderBlock(channels=LATENT_DIM),
            # EncoderBlock(channels=LATENT_DIM),
            # EncoderBlock(channels=LATENT_DIM),
        ])

        self.backbone = nn.Sequential(*[
            InferenceQuantization(self),
            nn.Conv2d(time_dim, EXPAND_DIM, kernel_size=8, stride=8, padding=0),
            InferenceQuantization(self),

            # InferenceQuantization(self),
            # nn.BatchNorm2d(time_dim),
            # DownsampleBlock(in_channels=time_dim, out_channels=32),
            # InferenceQuantization(self),
            # nn.BatchNorm2d(32),
            # DownsampleBlock(in_channels=32, out_channels=64),
            # InferenceQuantization(self),
            # DownsampleBlock(in_channels=64, out_channels=EXPAND_DIM),
            # nn.BatchNorm2d(EXPAND_DIM),
            # InferenceQuantization(self),
            # nn.BatchNorm2d(EXPAND_DIM),
            # EncoderBlock(channels=LATENT_DIM),
            # EncoderBlock(channels=LATENT_DIM),
            # EncoderBlock(channels=LATENT_DIM),

            # *[EncoderBlock(channels=EXPAND_DIM) for _ in range(4)],

            ROPE(),
            InferenceQuantization(self),

            # nn.Conv2d(LATENT_DIM, LATENT_DIM, kernel_size=1, stride=1, padding=0),
            # EncoderBlock(channels=LATENT_DIM),
            # nn.Dropout2d(0.05),
            ImageToTokens(),
            # StaticTokens(dim=EXPAND_DIM, num_tokens=64),
            *[SelfAttentionBlock(self, embed_dim=EXPAND_DIM, num_heads=8) for _ in range(16)],
            # RemoveStaticTokens(num_tokens=64),
            TokensToImage(),
            # nn.Conv2d(EXPAND_DIM, 64, kernel_size=1, stride=1, padding=0),

        ])

        self.upsample_block_0 = nn.Sequential(*[
            # nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
            EncoderBlock(channels=EXPAND_DIM),
            UpsampleBlock(in_channels=EXPAND_DIM, out_channels=EXPAND_DIM // 2),
            InferenceQuantization(self),
        ])
        self.upsample_block_1 = nn.Sequential(*[
            EncoderBlock(channels=EXPAND_DIM // 2),
            UpsampleBlock(in_channels=EXPAND_DIM // 2, out_channels=EXPAND_DIM // 4),
            InferenceQuantization(self),
        ])
        self.upsample_block_2 = nn.Sequential(*[
            EncoderBlock(channels=EXPAND_DIM // 4),
            UpsampleBlock(in_channels=EXPAND_DIM // 4, out_channels=EXPAND_DIM // 8),
            InferenceQuantization(self),
        ])

        self.filter_head_0 = nn.Sequential(*[
            nn.Conv2d(EXPAND_DIM // 2, 32, kernel_size=1, stride=1, padding=0),
            EncoderBlock(channels=32),
            EncoderBlock(channels=32),
            InferenceQuantization(self),
        ])
        self.filter_head_1 = nn.Sequential(*[
            nn.Conv2d(EXPAND_DIM // 4, 32, kernel_size=1, stride=1, padding=0),
            EncoderBlock(channels=32),
            EncoderBlock(channels=32),
            InferenceQuantization(self),
        ])
        self.filter_head_2 = nn.Sequential(*[
            nn.Conv2d(EXPAND_DIM // 8, 32, kernel_size=1, stride=1, padding=0),
            EncoderBlock(channels=32),
            EncoderBlock(channels=32),
            InferenceQuantization(self),
        ])

        # self.contract = nn.Sequential(*[
        #     nn.Conv2d(16, KPN_FEATURE_DIM, kernel_size=1, stride=1, padding=0),
        #     InferenceQuantization(self),
        # ])

        self.down0 = FilterNetworkDownsample()
        self.down1 = FilterNetworkDownsample()
        self.down2 = FilterNetworkDownsample()
        self.up2   = FilterNetworkUpsample()
        self.up1   = FilterNetworkUpsample()
        self.up0   = FilterNetworkUpsample()

        self.safe_dict = {}

    def quantize(self):
        quantized_model = self
        quantized_model.quantized = True
        moduels_to_quantize = [
            quantized_model.time_embed,
            quantized_model.expand,
            quantized_model.backbone,
            # quantized_model.contract,
        ]
        for module in moduels_to_quantize:
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.Linear):
                module.weight.data = module.weight.data.to(dtype=torch.float8_e4m3fn).to(dtype=torch.float32)
                # if module.bias is not None:
                #     module.bias.data = module.bias.data.to(dtype=torch.float8_e4m3fn)

            
        return quantized_model

    def forward(self, _z, noise_level):
        z = self.expand(_z)
        B, C, H, W = z.shape
        time_embedding = self.time_embed(noise_level)
        z = z + self.time_mlp(time_embedding).view(B, self.time_dim, 1, 1)
        z = self.backbone(torch.cat([z, ], dim=1))

        z0 = self.upsample_block_0(z)
        z1 = self.upsample_block_1(z0)
        z2 = self.upsample_block_2(z1)

        f0 = self.filter_head_0(z0)
        f1 = self.filter_head_1(z1)
        f2 = self.filter_head_2(z2)

        filtered0, down_0           = self.down0(_z, kernel_params=f2[:, 0:25, :, :])
        self.safe_dict["filtered0"] = filtered0
        self.safe_dict["down_0"]    = down_0
        filtered1, down_1           = self.down1(down_0, kernel_params=f1[:, 0:25, :, :])
        self.safe_dict["filtered1"] = filtered1
        self.safe_dict["down_1"]    = down_1
        filtered2, down_2           = self.down2(down_1, kernel_params=f0[:, 0:25, :, :])
        self.safe_dict["filtered2"] = filtered2
        self.safe_dict["down_2"]    = down_2
        up_2                        = self.up2(down_2, upscale_params=f0[:, 25:28, :, :], skip_signal=filtered2, skip_param=f0[:, 28:29, :, :])
        self.safe_dict["up_2"]      = up_2
        up_1                        = self.up1(up_2, upscale_params=f1[:, 25:28, :, :], skip_signal=filtered1, skip_param=f1[:, 28:29, :, :])
        self.safe_dict["up_1"]      = up_1
        up_0                        = self.up0(up_1, upscale_params=f2[:, 25:28, :, :], skip_signal=filtered0, skip_param=f2[:, 28:29, :, :])
        self.safe_dict["up_0"]      = up_0
        
        drift                       = self.wdrift(_z)
        # drift                       = F.interpolate(drift, size=(H, W), mode='bilinear', align_corners=False)

        self.safe_dict["drift"]      = drift

        return up_0 + drift

class CatDataloader:
    def __init__(self, path):
        self.paths = []
        self.data  = []
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        for file in os.listdir(path):
            if file.endswith(".png"):
                self.paths.append(os.path.join(path, file))

        
    def next(self, batch_size):
        batch = set()

        def aug(image):
            if random.random() < 0.5:
                # Apply random augmentations
                if random.random() < 0.5:
                    image = torch.flip(image, [3]) # horizontal flip
                # if random.random() < 0.5:
                #     image = torch.flip(image, [2]) # vertical flip
                # angle = random.choice([0, 90, 180, 270])
                # if angle != 0:
                #     image = torch.rot90(image, k=angle // 90, dims=[2, 3])
                
                # if random.random() < 0.5:
                #     # zoom
                #     scale = random.uniform(1.0, 2.2)
                #     new_size = int(64 * scale)
                #     image = F.interpolate(image, size=(new_size, new_size), mode='bicubic', align_corners=False)
                #     crop_x = random.randint(0, new_size - 64)
                #     crop_y = random.randint(0, new_size - 64)
                #     image = image[:, :, crop_x:crop_x+64, crop_y:crop_y+64]

            return image
        while len(batch) < batch_size:
            # Load a new image with a chance, otherwise reuse the old one
            prob = 0.5
            if len(self.paths) > 0 and (random.random() < prob or len(self.data) < 64):
                img_path = random.choice(range(len(self.paths)))
                image    = Image.open(self.paths[img_path])
                image    = self.transform(image).unsqueeze(0) # Map to [B, C, H, W]
                self.paths.remove(self.paths[img_path])
                self.data.append(image)
                batch.add(aug(image))

            else:

                image = random.choice(self.data)
                image = aug(image)
                batch.add(image)

        batch = list(set(batch))

        return torch.cat(batch, dim=0)

dataset = CatDataloader("data\\MNIST\\dataset-part1")

assert dataset.next(1).shape == (1, 3, 64, 64), f"Unexpected shape: {dataset.next(1).shape}"
size=64

num_epochs          = 100000
batch_size          = 64
num_iters_train     = 11024
num_iters           = 16

def assemble_batch():
    batch = dataset.next(batch_size=batch_size)
    return batch.to(device)

def get_beta_schedule(num_steps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, num_steps)

def get_alpha_schedule(betas):
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return alphas, alphas_cumprod

num_diffusion_steps             = num_iters_train
betas                           = get_beta_schedule(num_diffusion_steps).to(device)
alphas, alphas_cumprod          = get_alpha_schedule(betas)
sqrt_alphas_cumprod             = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod   = torch.sqrt(1.0 - alphas_cumprod)

import copy
def lerp(a, b, t):
    return a + (b - a) * t

def create_gaussian_kernel(kernel_size=5, sigma=1.0):
    ax     = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = torch.meshgrid(ax, ax)
    kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / torch.sum(kernel)
    return kernel

def gaussian_blur(img, kernel_size, sigma):
    B, C, H, W = img.shape
    imgs = []
    for b in range(B):
        kernel = create_gaussian_kernel(kernel_size, sigma[b:b+1].item())
        kernel = kernel.view(1, 1, kernel_size, kernel_size).to(img.device)
        kernel = kernel.repeat(C, 1, 1, 1)  # Repeat for each channel
        imgs.append(F.conv2d(img[b:b+1], kernel, padding=kernel_size // 2, groups=img.shape[1]))
    return torch.cat(imgs, dim=0)

diff_model = Diffusion().to(device)

try: # Load the model
    diff_model.load_state_dict(torch.load(".tmp/diff_model.pth"), strict=False)
except Exception as e:
    print("No model found, starting from scratch.")

# filter_network  = FilterNetwork().to(device)
diff_optimizer  = AdamW(params=diff_model.parameters(), lr=3e-4, weight_decay=1e-2)
lpips           = LPIPS().to(device)

num_inference_steps = 64
timestamps = torch.linspace(0, num_inference_steps - 1, num_inference_steps).float().to(device) / num_inference_steps

diff_model_ema = copy.deepcopy(diff_model)
diff_model_ema.requires_grad_(False)

for epoch in range(num_epochs):
    if epoch % 16 == 0:
        with DelayedKeyboardInterrupt():
            while True:
                try:
                    torch.save(diff_model.state_dict(), f".tmp/diff_model.pth")
                    break
                except Exception as e:
                    pass # Ignore file access issues or keyboard interrupts

    b = assemble_batch()
    t   = torch.rand(batch_size, 1, 1, 1, device=device).pow(1.0)
    z_src = b
    z_src = z_src.detach()
    z = z_src
    noise       = torch.randn_like(z)
    z = z_src + (noise - z_src) * (1.0 - t)
    noized = z

    diff_optimizer.zero_grad()
    loss           = 0.0

    z_next = z

    target          = diff_model(z, t)
    loss            = loss + ((target - b)).square().mean() * 1.0 #
    loss            = loss     + lpips(torch.clamp(target * 0.5 + 0.5, 0, 1), torch.clamp(z_src * 0.5 + 0.5, 0, 1))

    if 0: # Self-Forcing loss
        t_next = t
        N      = random.randint(4, 16)
        dt     = (1.0 - t) / 16
        # dt     = 1.0 / 128.0
        for i in range(N):
            target_ema    = diff_model_ema(z_next + torch.rand_like(z) * 0.01, t_next)
            z_next        = z_next + (target_ema - z_next) / (1.0 - t_next + 1e-6) * dt
            # z_next        = target_ema
            t_next        = t_next + dt
        
        target2 = diff_model(z_next, t_next)

        loss            = loss + ((target2 - b) * (-t * 0.0).exp()).square().mean() * 1.0 #


    if 0: # Self-Consistency loss
        t_next = t
        N      = random.randint(4, 16)
        dt     = (1.0 - t) / N
        # dt     = 1.0 / 128.0
        for i in range(N):
            target_ema    = diff_model_ema(z_next + torch.rand_like(z) * 0.01, t_next)
            z_next        = z_next + (target_ema - z_next) / (1.0 - t_next + 1e-6) * dt
            # z_next        = target_ema
            t_next        = t_next + dt

        loss            = loss + ((target - target_ema) * (-t * 0.0).exp()).square().mean() * 1.0 #


    if loss.isnan().any():
        print("NaN loss, exiting.")
        exit(1)

    loss.backward()
    diff_optimizer.step()        

    for param_q, param_k in zip(diff_model.parameters(), diff_model_ema.parameters()):
        gamma = 0.99
        param_k.data = param_k.data * gamma + param_q.data * (1.0 - gamma)

    if epoch % 16 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        with torch.no_grad():
            folder = ".tmp/viz_diffusion"
            os.makedirs(folder, exist_ok=True)

            dds = dds_from_tensor(diff_model.safe_dict["filtered0"][0:1, :, :, :] * 0.5 + 0.5)
            dds.save(f"{folder}/filtered0.dds")
            dds = dds_from_tensor(diff_model.safe_dict["down_0"][0:1, :, :, :] * 0.5 + 0.5)
            dds.save(f"{folder}/down_0.dds")
            dds = dds_from_tensor(diff_model.safe_dict["filtered1"][0:1, :, :, :] * 0.5 + 0.5)
            dds.save(f"{folder}/filtered1.dds")
            dds = dds_from_tensor(diff_model.safe_dict["down_1"][0:1, :, :, :] * 0.5 + 0.5)
            dds.save(f"{folder}/down_1.dds")
            dds = dds_from_tensor(diff_model.safe_dict["filtered2"][0:1, :, :, :] * 0.5 + 0.5)
            dds.save(f"{folder}/filtered2.dds")
            dds = dds_from_tensor(diff_model.safe_dict["down_2"][0:1, :, :, :] * 0.5 + 0.5)
            dds.save(f"{folder}/down_2.dds")
            dds = dds_from_tensor(diff_model.safe_dict["up_2"][0:1, :, :, :] * 0.5 + 0.5)
            dds.save(f"{folder}/up_2.dds")
            dds = dds_from_tensor(diff_model.safe_dict["up_1"][0:1, :, :, :] * 0.5 + 0.5)
            dds.save(f"{folder}/up_1.dds")
            dds = dds_from_tensor(diff_model.safe_dict["up_0"][0:1, :, :, :] * 0.5 + 0.5)
            dds.save(f"{folder}/up_0.dds")
            dds = dds_from_tensor(diff_model.safe_dict["drift"][0:1, :, :, :] * 0.5 + 0.5)
            dds.save(f"{folder}/drift.dds")
            decoded         = z
            stack = torch.zeros((1, 3, 4 * size, batch_size * size), device=device)
            
            for batch_idx in range(batch_size):
                stack[0, :, 0*size:1*size, batch_idx*size:(batch_idx+1)*size] = 0.5 + 0.5 * noized[batch_idx:batch_idx+1, :, :, :]
                stack[0, :, 1*size:2*size, batch_idx*size:(batch_idx+1)*size] = 0.5 + 0.5 * target[batch_idx:batch_idx+1, :, :, :]
                stack[0, :, 2*size:3*size, batch_idx*size:(batch_idx+1)*size] = 0.5 + 0.5 * b[batch_idx:batch_idx+1, :, :, :]
                # stack[0, :, 3*size:4*size, batch_idx*size:(batch_idx+1)*size] = 0.5 + 0.5 * target2[batch_idx:batch_idx+1, :, :, :]
            dds = dds_from_tensor(stack)
            dds.save(".tmp/input.dds")
            x = torch.randn(4, 3, 64, 64, device=device)
            viz_batch_size = 8
            viz_size       = size * 1
            _x              = torch.randn((viz_batch_size, 3, viz_size, viz_size), device=device)
            x = _x
            skip = 4
            viz_stack      = torch.zeros((1, 3, (num_inference_steps // skip + 1) * viz_size, viz_batch_size * viz_size), device=device)
            for batch_idx in range(viz_batch_size):
                viz_stack[0, :, 0*viz_size:1*viz_size, batch_idx*viz_size:(batch_idx+1)*viz_size] = 0.5 + 0.5 * x[batch_idx:batch_idx+1, :, :, :]
            gif_stack = []
            gif_stack.append(torch.nn.functional.avg_pool2d(0.5 + 0.5 * x[0, :, :, :], kernel_size=2, stride=2))
            dt = 1.0 / (num_inference_steps - 1)
            prev_x = x
            nesterov_step = 0.0
            inf_z = None
            inf_z = x
    
            quantized_inference_model = Diffusion().to(device)
            state_dict = diff_model.state_dict()
            quantized_inference_model.load_state_dict(state_dict)
            # quantized_inference_model = quantized_inference_model.quantize()

            for step in range(num_inference_steps):
                t           = (step) / (num_inference_steps)
                t           = torch.tensor([t], device=device).repeat(viz_batch_size).reshape(viz_batch_size, 1, 1, 1)
                shift       = 1.0
                t           = shift * t / (1.0 + (shift - 1.0) * t)
                dt          = shift / (1.0 + (shift - 1.0) * t)**2 * 1.0 / num_inference_steps
                target      = quantized_inference_model(inf_z, t)
                inf_z       = inf_z + (target - inf_z) / (1.0 - t) * dt
                # inf_z       = target
                x           = inf_z
                prev_x      = x
                # if step < num_inference_steps - 1:
                #     sigma_noise = 0.01
                #     noise       = torch.randn_like(x)
                #     inf_z       = inf_z + noise * sigma_noise

                gif_stack.append(torch.nn.functional.avg_pool2d(0.5 + 0.5 * inf_z[0, :, :, :].detach().cpu(), kernel_size=2, stride=2))
                if (step + 1) % skip == 0:
                    for batch_idx in range(viz_batch_size):
                        viz_stack[0, :, (step // skip+1)*viz_size:(step//skip+2)*viz_size, batch_idx*viz_size:(batch_idx+1)*viz_size] = 0.5 + 0.5 * inf_z[batch_idx:batch_idx+1, :, :, :]
                        

            dds = dds_from_tensor(viz_stack)
            dds.save(".tmp/output.dds")

            from PIL import Image
            pil_images = []
            for tensor in gif_stack:
                # Move to CPU, permute to [H, W, C], scale to [0, 255], convert to uint8
                img_array = (tensor.permute(1, 2, 0).cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
                pil_img   = Image.fromarray(img_array)
                pil_images.append(pil_img)

            # Save as animated GIF
        
            pil_images[0].save(
                '.tmp/output.gif',
                save_all=True,
                append_images=pil_images[0:],
                duration=100,  # milliseconds per frame
                loop=0  # 0 for infinite loop
            )
