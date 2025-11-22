# MIT License
# 
# Copyright (c) 2025 Anton Schreiner
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
PATCH_DIM = 1

class EncoderBlock(nn.Module):
    def __init__(self, channels, groups=1, skip=True):
        super(EncoderBlock, self).__init__()
        self.dwconv    = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=True, groups=groups)
        self.pwconv1   = nn.Conv2d(in_channels=channels, out_channels=channels * 2, kernel_size=1, stride=1, padding=0, bias=True)
        # self.bn1       = nn.BatchNorm2d(channels)
        self.act       = nn.GELU()
        # self.act       = nn.LeakyReLU(0.01, inplace=True)
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
    def __init__(self, embed_dim, num_heads):
        super(SelfAttentionBlock, self).__init__()
        # self.attn   = LinearAttention(embed_dim, num_heads)
        self.attn   = SelfAttention(embed_dim, num_heads)
        self.ffw    = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            Swiglu(embed_dim),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
        )
        self.norm1  = nn.LayerNorm(embed_dim)
        self.norm2  = nn.LayerNorm(embed_dim)

    def forward(self, _x):
        x = _x
        x = x + self.attn(self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        x = x + self.ffw(self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        return x.contiguous()

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


class VAEDiffusion(nn.Module):
    def __init__(self):
        super(VAEDiffusion, self).__init__()

        time_dim = 128
        EXPAND_DIM = 512
        self.expand_dim = EXPAND_DIM

        self.time_dim   = time_dim
        self.time_embed = SinusoidalPositionalEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, EXPAND_DIM),
        )

        self.expand = nn.Conv2d(LATENT_DIM, EXPAND_DIM, kernel_size=1, stride=1, padding=0)

        self.backbone = nn.Sequential(*[
            nn.BatchNorm2d(EXPAND_DIM),
            DownsampleBlock(in_channels=EXPAND_DIM, out_channels=EXPAND_DIM),
            nn.BatchNorm2d(EXPAND_DIM),
            DownsampleBlock(in_channels=EXPAND_DIM, out_channels=EXPAND_DIM),
            nn.BatchNorm2d(EXPAND_DIM),

            # EncoderBlock(channels=LATENT_DIM),
            # EncoderBlock(channels=LATENT_DIM),
            # EncoderBlock(channels=LATENT_DIM),

            *[EncoderBlock(channels=EXPAND_DIM) for _ in range(4)],

            ROPE(),

            # nn.Conv2d(LATENT_DIM, LATENT_DIM, kernel_size=1, stride=1, padding=0),
            # EncoderBlock(channels=LATENT_DIM),
            # nn.Dropout2d(0.05),
            ImageToTokens(),
            StaticTokens(dim=EXPAND_DIM, num_tokens=64),
            *[SelfAttentionBlock(embed_dim=EXPAND_DIM, num_heads=8) for _ in range(4)],
            RemoveStaticTokens(num_tokens=64),
            TokensToImage(),
            # nn.Conv2d(LATENT_DIM, LATENT_DIM, kernel_size=1, stride=1, padding=0),
            
            nn.BatchNorm2d(EXPAND_DIM),
            UpsampleBlock(in_channels=EXPAND_DIM, out_channels=EXPAND_DIM),
            nn.BatchNorm2d(EXPAND_DIM),
            UpsampleBlock(in_channels=EXPAND_DIM, out_channels=EXPAND_DIM),
        ])

        self.contract = nn.Conv2d(EXPAND_DIM, LATENT_DIM, kernel_size=1, stride=1, padding=0)

    def get_grad(self, _z, noise_level):
        z = self.expand(_z)
        B, C, H, W = z.shape
        time_embedding = self.time_embed(noise_level)
        z = z + self.time_mlp(time_embedding).view(B, self.expand_dim, 1, 1)
        # z = self.backbone(torch.cat([z, recurrent], dim=1))
        z = self.backbone(torch.cat([z, ], dim=1))
        # z, recurrent = z.chunk(2, dim=1)

        return self.contract(z)# , torch.nn.functional.tanh(recurrent)

    def forward(self, x):
        assert False, "Not implemented"


class GAN(nn.Module):
    def __init__(self):
        super(GAN, self).__init__()

        self.dm_self_attn_bk = nn.Sequential(*[
            nn.Conv2d(3, PATCH_DIM*PATCH_DIM*3, kernel_size=PATCH_DIM, stride=PATCH_DIM, padding=0),
            ROPE(),
            *[SelfAttentionBlock(embed_dim=PATCH_DIM*PATCH_DIM*3, num_heads=8) for _ in range(8)],
            nn.Conv2d(PATCH_DIM*PATCH_DIM*3, LATENT_DIM, kernel_size=1, stride=1, padding=0),
        ])
        
        num_patches = (64 // PATCH_DIM) * (64 // PATCH_DIM)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=LATENT_DIM * num_patches, out_features=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = x
        B, C, H, W = z.shape
        z = self.dm_self_attn_bk(z)

        z = z.view(B, -1)
        validity = self.classifier(z)

        return validity

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
batch_size          = 32
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
# cloned_model = copy.deepcopy(model)
# # Disable gradients for the cloned model
# for param in cloned_model.parameters():
#     param.requires_grad = False

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

# def gradient_magniture(t):
#     return t

kodim = transforms.ToTensor()(Image.open(".tmp\\kodim23.png")).unsqueeze(0).to(device) * 2 - 1
# kodim = kodim[:, :, 0:512, 100:100+512]
kodim = kodim[:, :, 0:512, 0:512]

if 1:
    
    diff_model = VAEDiffusion().to(device)
    # gan_model = GAN().to(device)

    try: # Load the model
        diff_model.load_state_dict(torch.load(".tmp/diff_model.pth"), strict=False)
        # gan_model.load_state_dict(torch.load(".tmp/gan_model.pth"), strict=False)
    except Exception as e:
        print("No model found, starting from scratch.")


    diff_model_ema = copy.deepcopy(diff_model)
    diff_model_ema.requires_grad_(False)

    diff_optimizer = AdamW(params=diff_model.parameters(), lr=1e-4, weight_decay=1e-2)
    lr_scheduler2 = PolynomialLR(diff_optimizer, total_iters=num_epochs, power=1.0)
    # lr_scheduler3 = PolynomialLR(gan_optimizer, total_iters=num_epochs, power=1.0)

    lpips  = LPIPS().to(device)

    do_neg_training = False
    # model_param_checkpoint = copy.deepcopy(model.state_dict())

    num_inference_steps = 64
    timestamps = torch.linspace(0, num_inference_steps - 1, num_inference_steps).float().to(device) / num_inference_steps

    for epoch in range(num_epochs):
        if epoch % 16 == 0:
            with DelayedKeyboardInterrupt():
                while True:
                    try:
                        torch.save(diff_model.state_dict(), f".tmp/diff_model.pth")
                        # torch.save(gan_model.state_dict(), f".tmp/gan_model.pth")
                        break
                    except Exception as e:
                        pass # Ignore file access issues or keyboard interrupts

        # if epoch == 0 or random.random() < 0.05:
        if 1:
            b = assemble_batch()
            
            t   = torch.rand(batch_size, 1, 1, 1, device=device).pow(1.0)

            # pick randomly from timestamps
            # for i in range(batch_size):
            #     t_idx               = random.randint(0, num_inference_steps - 1)
            #     t[i:i + 1, :, :, :]  = timestamps[t_idx:t_idx+1]

            # shift = 1.0
            # t     = shift * t / (1.0 + (shift - 1.0) * t)
            # dt    = shift / (1.0 + (shift - 1.0) * t)**2 * 1.0 / num_inference_steps
            # dt = 1.0 / num_inference_steps

            # noise_level_1 = noise_level * torch.rand(batch_size, 1, 1, 1, device=device)
            # noise_level_1 = noise_level * 0.2
            z_src = b
            z_src = z_src.detach()
            z = z_src
            noise       = torch.randn_like(z)
            z = z_src + (noise - z_src) * (1.0 - t)

        # Diffusion model step

        diff_optimizer.zero_grad()
        loss           = 0.0
        for i in range(1):

            # dt            = 1.0 / 2.0 * torch.rand_like(t)

            target          = diff_model.get_grad(z, t)

            # grad_next = diff_model.get_grad(z + grad * dt, t + dt)
            # grad_prev = diff_model.get_grad(z - grad * dt, t - dt)
            z_next = z
            if 1:
                t_next = t
                N      = 2 # random.randint(4, 16)
                # dt     = (1.0 - t) / N
                dt     = 1.0 / 128.0
                v_avg = torch.zeros_like(z_next)
                for i in range(N):
                    target_ema    = diff_model_ema.get_grad(z_next + torch.rand_like(z) * 0.01, t_next)
                    z_next        = z_next + (target_ema - z_next) * dt
                    t_next        = t_next + dt
                    v_avg         = v_avg + target_ema / N

                loss            = loss + ((target - v_avg) * (-t * 0.0).exp()).square().mean() * 1.0 #

            
            # grad_ema_next = diff_model_ema.get_grad(z + grad_ema * dt + torch.rand_like(z) * 0.001, t + dt)
            # grad_ema_prev = diff_model_ema.get_grad(z - grad_ema * dt + torch.rand_like(z) * 0.001, t - dt)

            # k =  (math.exp(iter / 16.0))
            loss            = loss + ((target - z_src)).square().mean() * 1.0 #
            # loss            = loss + ((grad - (grad_ema_next + grad_ema_prev) / 2.0)).square().mean() * 1.0 #
            # loss            = loss + ((grad - (grad_ema * 2.0 + grad_ema_next + grad_ema_prev) / 4.0) * (-t * 4.0).exp()).square().mean() * 16.0 #
            # loss            = loss + ((grad - (grad_ema * 1.0 + grad_ema_next + grad_ema_prev) / 3.0)).square().mean() * 1.0 #
            
            # loss            = loss + (grad + noise).square().mean() * 1.0 #
            # loss            = loss + (1.0 - torch.nn.functional.cosine_similarity(grad, (z_src - z), dim=1)).mean() * 1.0 #
            # z = (z + grad * noise_level.sqrt()) / (1.0e-6 + (1 - noise_level).abs()).sqrt()
            # z               = z + grad * (1.0 - t)

            # loss            = loss + ((z - z_src) / (1.0e-3 + (1.0 - t))).square().mean() * 1.25# penalize distance to target

            # decoded         = z
            # loss            = loss     + lpips(torch.clamp(target * 0.5 + 0.5, 0, 1), torch.clamp(z_src * 0.5 + 0.5, 0, 1))

            # loss            = loss + (decoded - noisy_input).square().mean() * 0.15 # penalize big steps
            # loss            = loss + ((decoded - b) / (1.0e-3 + noise_level)).square().mean() * 1.25# penalize distance to target
            # loss            = loss + ((decoded - b) / (1.0e-3 + (1.0 - t))).square().mean() * 1.25# penalize distance to target

            # t = t + dt
            # if i < 15:
            #     sigma = 0.01
            #     z = z * (1.0 - sigma) + torch.randn_like(z) * sigma
            
            # loss            = loss + ((z - z_src) / (1.0e-3 + noise_level)).square().mean() * 1.25# penalize distance to target
            # loss            = loss + ((grad - (z_src - noise)) * torch.exp(-2.0 * noise_level)).square().mean() * 1.25# penalize distance to target
            # loss            = loss + ((grad - (z_src - noise))).square().mean() * 1.0 # penalize distance to target

            # noise_level = noise_level * 0.5

        
        # validity = gan_model(decoded)
        # loss = loss + F.binary_cross_entropy(validity, torch.ones_like(validity)) * 1.0

            # decoded_fake         = decoded.detach()

        for param_q, param_k in zip(diff_model.parameters(), diff_model_ema.parameters()):
            gamma = 0.99
            param_k.data = param_k.data * gamma + param_q.data * (1.0 - gamma)

        z               = z.detach()
        # recurrent       = recurrent.detach()

        if loss.isnan().any():
            print("NaN loss, exiting.")
            exit(1)

        # sigma = 0.
        # z     = z * (1.0 - sigma) + torch.randn_like(z) * sigma

        loss.backward()
        diff_optimizer.step()        

        lr_scheduler2.step()


        if epoch % 16 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            
            
            with torch.no_grad():
            # if 1:
                decoded         = z
                stack = torch.zeros((1, 3, 3 * size, batch_size * size), device=device)
                
                for batch_idx in range(batch_size):
                    stack[0, :, 0:size, batch_idx*size:(batch_idx+1)*size]      = 0.5 + 0.5 * target[batch_idx:batch_idx+1, :, :, :]
                    stack[0, :, size:2*size, batch_idx*size:(batch_idx+1)*size] = 0.5 + 0.5 * b[batch_idx:batch_idx+1, :, :, :]
                    stack[0, :, 2*size:3*size, batch_idx*size:(batch_idx+1)*size] = 0.5 + 0.5 * v_avg[batch_idx:batch_idx+1, :, :, :]
                dds = dds_from_tensor(stack)
                dds.save(".tmp/input.dds")

                # dds = dds_from_tensor(recurrent_features[0:1, 0:3, :, :] * 0.5 + 0.5)
                # dds.save(".tmp/recurrent_features.dds")
                
                x = torch.randn(4, 3, 64, 64, device=device)

                # sampling_steps = torch.linspace(num_diffusion_steps - 1, 0, num_iters).long().to(device)
            

                viz_batch_size = 8
                viz_size       = size * 1
                _x              = torch.randn((viz_batch_size, 3, viz_size, viz_size), device=device)
                x = _x
                # x = kodim[:, :, :viz_size, :viz_size]
                skip = 1
                viz_stack      = torch.zeros((1, 3, (num_inference_steps // skip + 1) * viz_size, viz_batch_size * viz_size), device=device)
                # viz_stack2      = torch.zeros((1, 3, (num_inference_steps // skip + 1) * 4, viz_batch_size * 4), device=device)
                for batch_idx in range(viz_batch_size):
                    viz_stack[0, :, 0*viz_size:1*viz_size, batch_idx*viz_size:(batch_idx+1)*viz_size] = 0.5 + 0.5 * x[batch_idx:batch_idx+1, :, :, :]
                gif_stack = []
                gif_stack.append(torch.nn.functional.avg_pool2d(0.5 + 0.5 * x[0, :, :, :], kernel_size=2, stride=2))

                dt = 1.0 / (num_inference_steps - 1)

                # recurrent_features = torch.zeros((viz_batch_size, 4, size, size), device=device)

                prev_x = x
                nesterov_step = 0.0
                # inf_recurrent_features = torch.zeros((viz_batch_size, 4, viz_size, viz_size), device=device)

                inf_z = None
                # z, latents_mu, latents_logvar = model.encode(decoded, sample=False)
                inf_z = torch.randn((viz_batch_size, LATENT_DIM, viz_size // PATCH_DIM, viz_size // PATCH_DIM), device=device)
                recurrent = torch.randn((viz_batch_size, LATENT_DIM, viz_size // PATCH_DIM, viz_size // PATCH_DIM), device=device)

                # inf_noise_level = 1.0
                # inf_noise_level = torch.tensor([inf_noise_level], device=device).repeat(viz_batch_size).reshape(viz_batch_size, 1, 1, 1)

                for step in range(num_inference_steps):
                    # t_batch = torch.full((viz_batch_size,), step_idx, device=device, dtype=torch.long)
                    t = (step) / (num_inference_steps)
                    t = torch.tensor([t], device=device).repeat(viz_batch_size).reshape(viz_batch_size, 1, 1, 1)

                    shift = 1.0
                    t = shift * t / (1.0 + (shift - 1.0) * t)
                    dt = shift / (1.0 + (shift - 1.0) * t)**2 * 1.0 / num_inference_steps

                    # grad = model(x)
                    target                   = diff_model.get_grad(inf_z, t)
                    # inf_z                      = inf_z + grad / (num_inference_steps)
                    # dt = 0.25
                    # dt              = 1.0 / num_inference_steps
                    inf_z           = inf_z + (target - inf_z) / (1.0 - t) * dt
                    # inf_noise_level  = inf_noise_level * (1.0 - dt)
                    # inf_noise_level = inf_noise_level * 0.85
                    # inf_z = (inf_z + grad * (1.0 - t).sqrt()) / (1.0e-6 + (1.0 - (1.0 - t)).abs()).sqrt()
                    # inf_z                      = grad
                    
                    # x                      = x + grad * torch.rand(viz_batch_size, device=device).view(-1, 1, 1, 1)
                    # x                      = x + grad * 1

                    # if step < num_inference_steps - 1:
                    #     sigma = (1.0 - t) * 0.025
                    #     # x = x * (1.0 - sigma) + torch.randn_like(x) * sigma
                    #     inf_z = inf_z * (1.0 - sigma) + torch.randn_like(inf_z) * sigma

                    x                      = inf_z
                    prev_x                 = x

                    gif_stack.append(torch.nn.functional.avg_pool2d(0.5 + 0.5 * x[0, :, :, :].detach().cpu(), kernel_size=2, stride=2))
                    if (step + 1) % skip == 0:
                        for batch_idx in range(viz_batch_size):
                            viz_stack[0, :, (step // skip+1)*viz_size:(step//skip+2)*viz_size, batch_idx*viz_size:(batch_idx+1)*viz_size] = 0.5 + 0.5 * x[batch_idx:batch_idx+1, :, :, :]
                            # viz_stack2[0, :, (step // skip+1)*4:(step//skip+2)*4, batch_idx*4:(batch_idx+1)*4] = 0.5 + 0.5 * recurrent[batch_idx:batch_idx+1, 0:3, :, :]
                    

                if 0:
                    scores_0 = model.self_attn_bk_0.attn.last_scores
                    scores_1 = model.self_attn_bk_1.attn.last_scores
                    scores_2 = model.self_attn_bk_2.attn.last_scores
                    scores_3 = model.self_attn_bk_3.attn.last_scores
                    
                    num_heads = model.self_attn_bk_0.attn.num_heads
                    attn_size   = viz_size // 16  # Assuming the attention is applied at 16x16 resolution
                    score_image = torch.zeros((1, 4, attn_size * attn_size, attn_size * attn_size), device=device)

                    for head_idx in range(num_heads):

                        # print(f"Scores shape: {scores_0.shape}")

                        for y in range(attn_size):
                            for x in range(attn_size):
                                score_image[0:1, 0:1, y * attn_size:(y + 1) * attn_size, x * attn_size:(x + 1) * attn_size] = scores_0[0, head_idx, y * attn_size + x, :].view(attn_size, attn_size)

                        dds = dds_from_tensor(score_image)
                        dds.save(f".tmp/scores_{head_idx}.dds")
                        

               

                dds = dds_from_tensor(viz_stack)
                dds.save(".tmp/output.dds")

                 # Visualize latent space
                viz_stack = torch.zeros((1, 3, size // PATCH_DIM, viz_batch_size * (size // PATCH_DIM)), device=device)
                for i in range(viz_batch_size):
                    viz_stack[0, :, :, i*(size // PATCH_DIM):(i+1)*(size // PATCH_DIM)] = inf_z[i:i+1, 0:3, :, :]

                # Save the visualization
                dds = dds_from_tensor(viz_stack)
                dds.save(".tmp/output_latent_space.dds")

                # dds = dds_from_tensor(viz_stack2)
                # dds.save(".tmp/output_recurrent_features.dds")

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

                # dds = dds_from_tensor(viz_stack)
                # dds.save(".tmp/iterative_diffusion_samples.dds")