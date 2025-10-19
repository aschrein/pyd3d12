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
from piq import SSIMLoss
import math
import numpy as np
from pytorch_optimizer import Muon, SOAP, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, PolynomialLR

assert torch.cuda.is_available()
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(torch.cuda.current_device()))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class EncoderBlock(nn.Module):
    def __init__(self, channels, groups=1):
        super(EncoderBlock, self).__init__()
        # self.bn0       = nn.BatchNorm2d(channels)
        self.dwconv    = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=True, groups=groups)
        self.pwconv1   = nn.Conv2d(in_channels=channels, out_channels=channels * 2, kernel_size=1, stride=1, padding=0, bias=True)
        # self.bn1       = nn.BatchNorm2d(channels * 2)
        # self.act       = nn.SiLU(inplace=True)
        self.act       = nn.LeakyReLU(0.01, inplace=True)
        self.pwconv2   = nn.Conv2d(in_channels=channels * 2, out_channels=channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        # x = self.bn0(x)
        identity = x
        out      = self.dwconv(x)
        out      = self.pwconv1(out)
        # out      = self.bn1(out)
        out      = self.act(out)
        out      = self.pwconv2(out)
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

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
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

class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttentionBlock, self).__init__()
        self.attn   = SelfAttention(embed_dim, num_heads)
        self.ffw    = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
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

class Transformer4x4(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(Transformer4x4, self).__init__()
        self.num_heads  = num_heads
        self.head_dim   = embed_dim // num_heads
        self.scale      = self.head_dim ** -0.5
        
        self.learned_positional_encoding = nn.Parameter(torch.randn(1, embed_dim, 4, 4))
        self.self_attn_bk_0 = SelfAttentionBlock(embed_dim=embed_dim, num_heads=num_heads)
        self.self_attn_bk_1 = SelfAttentionBlock(embed_dim=embed_dim, num_heads=num_heads)
        self.self_attn_bk_2 = SelfAttentionBlock(embed_dim=embed_dim, num_heads=num_heads)
        self.self_attn_bk_3 = SelfAttentionBlock(embed_dim=embed_dim, num_heads=num_heads)
    
        self.last_scores = None
    
    def forward(self, x):
        x = x + self.learned_positional_encoding
        x = self.self_attn_bk_0(x)
        x = self.self_attn_bk_1(x)
        x = self.self_attn_bk_2(x)
        x = self.self_attn_bk_3(x)
        return x

class DiffusionModel(nn.Module):
    def __init__(self, image_channels=3, time_dim=128):
        super(DiffusionModel, self).__init__()
        self.time_dim = time_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=3, stride=1, padding=1),  # 64x64
            EncoderBlock(channels=32),
            EncoderBlock(channels=32),
            EncoderBlock(channels=32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=2, padding=0), # 32x32
            EncoderBlock(channels=64),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=2, padding=0), # 16x16
            EncoderBlock(channels=128),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, stride=2, padding=0), # 8x8
            EncoderBlock(channels=256),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=2, stride=2, padding=0), # 4x4
            Transformer4x4(embed_dim=512, num_heads=4),
            EncoderBlock(channels=512),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=2, stride=2, padding=0), # 2x2
            EncoderBlock(channels=1024),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=2, stride=2, padding=0), # 1x1
            EncoderBlock(channels=1024),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024, out_channels=1024, kernel_size=2, stride=2, padding=0), # 2x2
            EncoderBlock(channels=1024),
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2, padding=0), # 4x4
            EncoderBlock(channels=512),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2, padding=0), # 8x8
            EncoderBlock(channels=256),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2, padding=0), # 8x8
            EncoderBlock(channels=128),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=0), # 16x16
            EncoderBlock(channels=64),
            nn.ConvTranspose2d(in_channels=64, out_channels=image_channels, kernel_size=2, stride=2, padding=0), # 32x32
        )

        # self.learned_positional_encoding = nn.Parameter(torch.randn(1, 256, 4, 4))
        # self.self_attn_bk_0 = SelfAttentionBlock(embed_dim=256, num_heads=4)
        # self.self_attn_bk_1 = SelfAttentionBlock(embed_dim=256, num_heads=4)
        # self.self_attn_bk_2 = SelfAttentionBlock(embed_dim=256, num_heads=4)
        # self.self_attn_bk_3 = SelfAttentionBlock(embed_dim=256, num_heads=4)

        # Latent space projections
        self.fc_mu      = nn.Conv2d(1024, 1024, kernel_size=1, padding=0)
        self.fc_logvar  = nn.Conv2d(1024, 1024, kernel_size=1, padding=0)

        # MLP for final output
        self.mlp0 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1, stride=1, padding=0) # 4x4
        

        # self.mlp0 = nn.Linear(in_features=256, out_features=128, bias=True)
        # self.mlp1 = nn.Linear(in_features=128, out_features=1, bias=True)

    def forward(self, x):
        assert False, "Use gen_score or gen_vae"

    def gen_score(self, x):
        x4 = self.encoder(x) # + t_cond  # 64x64x32
        
        # MLP for final output
        # x0 = x0.view(x0.size(0), -1)
        # x0 = self.mlp0(x0)
        # x0 = nn.functional.leaky_relu(x0, 0.01, inplace=True)
        # x0 = self.mlp1(x0)

        # Bottleneck with attention
        # x4 = x4 + self.learned_positional_encoding
        # x4 = self.self_attn_bk_0(x4)
        # x4 = self.self_attn_bk_1(x4)
        # x4 = self.self_attn_bk_2(x4)
        # x4 = self.self_attn_bk_3(x4)

        # Latent space
        latents_mu     = self.fc_mu(x4)
        latents_logvar = self.fc_logvar(x4)

        # Reparameterization trick
        z = latents_mu

        x0 = self.mlp0(z)
        
        x0 = nn.functional.leaky_relu(x0, 0.01, inplace=True)

        return x0
    
    def gen_vae(self, x):
        x4 = self.encoder(x) # + t_cond  # 64x64x32

        # Latent space
        latents_mu     = self.fc_mu(x4)
        latents_logvar = self.fc_logvar(x4)

        # Reparameterization trick
        if self.training:
            std = torch.exp(0.5 * latents_logvar)
            eps = torch.randn_like(std)
            z   = latents_mu + eps * std
        else:
            z = latents_mu

        z = self.decoder(z)

        return z, latents_mu, latents_logvar

class GradientAdapter(nn.Module):
    def __init__(self, image_channels=3, time_dim=128):
        super(GradientAdapter, self).__init__()
        self.time_dim = time_dim
        
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels*2, 16, kernel_size=3, stride=1, padding=1),
            EncoderBlock(channels=16),
        )
        self.down1 = DownsampleBlock(in_channels=16, out_channels=32)
        self.enc1 = EncoderBlock(channels=32)

        self.down2 = DownsampleBlock(in_channels=32, out_channels=64)
        self.enc2 = EncoderBlock(channels=64)

        self.down3 = DownsampleBlock(in_channels=64, out_channels=128)
        self.enc3 = EncoderBlock(channels=128)

        self.down4 = DownsampleBlock(in_channels=128, out_channels=256)

        # Bottleneck with attention
        self.learned_positional_encoding = nn.Parameter(torch.randn(1, 256, 4, 4))
        self.self_attn_bk_0 = SelfAttentionBlock(embed_dim=256, num_heads=8)
        self.self_attn_bk_1 = SelfAttentionBlock(embed_dim=256, num_heads=8)
        self.self_attn_bk_2 = SelfAttentionBlock(embed_dim=256, num_heads=8)
        self.self_attn_bk_3 = SelfAttentionBlock(embed_dim=256, num_heads=8)

        self.up1  = UpsampleBlock(in_channels=256, out_channels=128)
        self.dec1 = EncoderBlock(channels=128 + 128)  # +128 for skip connection

        self.up2  = UpsampleBlock(in_channels=128 + 128, out_channels=64)
        self.dec2 = EncoderBlock(channels=64 + 64)  # +64 for skip connection

        self.up3  = UpsampleBlock(in_channels=64 + 64, out_channels=32)
        self.dec3 = EncoderBlock(channels=32 + 32)    # +32 for skip connection

        self.up4  = UpsampleBlock(in_channels=32 + 32, out_channels=16)
        self.dec4 = EncoderBlock(channels=16 + 16)    # +16 for skip connection

        self.output_conv = nn.Conv2d(16 + 16, image_channels, kernel_size=3, padding=1)


    def forward(self, x, grad):
       
        x0 = self.encoder(torch.cat([x, grad], dim=1))
        x1 = self.down1(x0)
        x1 = self.enc1(x1)
        
        x2 = self.down2(x1)
        x2 = self.enc2(x2)
        
        x3 = self.down3(x2)
        x3 = self.enc3(x3)
        
        x4 = self.down4(x3)

        # Bottleneck with attention
        x4 = x4 + self.learned_positional_encoding
        x4 = self.self_attn_bk_0(x4)
        x4 = self.self_attn_bk_1(x4)
        x4 = self.self_attn_bk_2(x4)
        x4 = self.self_attn_bk_3(x4)

        z = x4

        x = self.up1(z)
        x = torch.cat([x, x3], dim=1)
        x = self.dec1(x)
        
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x)
        
        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec3(x)
        
        x = self.up4(x)
        x = torch.cat([x, x0], dim=1)
        x = self.dec4(x)
        
        grad = self.output_conv(x)

        return grad


class DiffusionModel1(nn.Module):
    def __init__(self, image_channels=3):
        super(DiffusionModel1, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=3, stride=1, padding=1),
            EncoderBlock(channels=32),
        )
        self.down1 = DownsampleBlock(in_channels=32, out_channels=64)
        self.enc1 = EncoderBlock(channels=64)
    
        self.down2 = DownsampleBlock(in_channels=64, out_channels=128)
        self.enc2 = EncoderBlock(channels=128)
        
        self.down3 = DownsampleBlock(in_channels=128, out_channels=256)
        self.enc3 = EncoderBlock(channels=256)
        
        self.down4 = DownsampleBlock(in_channels=256, out_channels=512)
        
        self.learned_positional_encoding = nn.Parameter(torch.randn(1, 512, 4, 4))
        self.self_attn_bk_0 = SelfAttentionBlock(embed_dim=512, num_heads=8)
        self.self_attn_bk_1 = SelfAttentionBlock(embed_dim=512, num_heads=8)
        self.self_attn_bk_2 = SelfAttentionBlock(embed_dim=512, num_heads=8)
        self.self_attn_bk_3 = SelfAttentionBlock(embed_dim=512, num_heads=8)

        self.up1  = UpsampleBlock(in_channels=512, out_channels=256)
        self.dec1 = EncoderBlock(channels=256 + 256)  # +256 for skip connection
        
        self.up2  = UpsampleBlock(in_channels=256 + 256, out_channels=128)
        self.dec2 = EncoderBlock(channels=128 + 128)  # +128 for skip connection

        self.up3  = UpsampleBlock(in_channels=128 + 128, out_channels=64)
        self.dec3 = EncoderBlock(channels=64 + 64)    # +64 for skip connection

        self.up4  = UpsampleBlock(in_channels=64 + 64, out_channels=32)
        self.dec4 = EncoderBlock(channels=32 + 32)    # +32 for skip connection
        
        self.output_conv = nn.Conv2d(32 + 32, 1, kernel_size=3, padding=1)

        self.mlp0 = nn.Linear(in_features=64*64, out_features=256, bias=True)
        self.mlp1 = nn.Linear(in_features=256, out_features=1, bias=True)


    def forward(self, x):
        x0 = self.encoder(x) # + t_cond  # 64x64x32
        x1 = self.down1(x0)   # 32x32x64
        x1 = self.enc1(x1)
        x2 = self.down2(x1)   # 16x16x128
        x2 = self.enc2(x2)
        x3 = self.down3(x2)   # 8x8x256
        x3 = self.enc3(x3)
        x4 = self.down4(x3)   # 4x4x512
        
        # Bottleneck with attention
        x4 = x4 + self.learned_positional_encoding
        x4 = self.self_attn_bk_0(x4)
        x4 = self.self_attn_bk_1(x4)
        x4 = self.self_attn_bk_2(x4)
        x4 = self.self_attn_bk_3(x4)
        z = x4

        # Decoder with skip connections
        x = self.up1(z)      # 8x8x256
        x = torch.cat([x, x3], dim=1)  # Skip connection
        x = self.dec1(x)
        
        x = self.up2(x)       # 16x16x128
        x = torch.cat([x, x2], dim=1)  # Skip connection
        x = self.dec2(x)
        
        x = self.up3(x)       # 32x32x64
        x = torch.cat([x, x1], dim=1)  # Skip connection
        x = self.dec3(x)
        
        x = self.up4(x)       # 64x64x32
        x = torch.cat([x, x0], dim=1)  # Skip connection
        x = self.dec4(x)
        
        pred = self.output_conv(x)

        # MLP for final output
        x0 = pred.view(pred.size(0), -1)
        # x0 = nn.functional.sigmoid(x0)
        x0 = self.mlp0(x0)
        x0 = nn.functional.leaky_relu(x0, 0.01, inplace=True)
        x0 = self.mlp1(x0)

        return x0

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
        while len(batch) < batch_size:
            # Load a new image with a chance, otherwise reuse the old one
            prob = 0.5
            if len(self.paths) > 0 and (random.random() < prob or len(self.data) < 64):
                img_path = random.choice(range(len(self.paths)))
                image    = Image.open(self.paths[img_path])
                image    = self.transform(image).unsqueeze(0) # Map to [B, C, H, W]
                self.paths.remove(self.paths[img_path])
                self.data.append(image)
                batch.add(image)

            else:

                image = random.choice(self.data)
                batch.add(image)

        batch = list(set(batch))

        return torch.cat(batch, dim=0)

dataset = CatDataloader("data\\MNIST\\dataset-part1")

assert dataset.next(1).shape == (1, 3, 64, 64), f"Unexpected shape: {dataset.next(1).shape}"
size=64

num_epochs          = 10000
batch_size          = 64
num_iters_train     = 1024
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

if 1:
    
    model = DiffusionModel(image_channels=3).to(device)
    # gradient_adapter = GradientAdapter(image_channels=3).to(device)

    try: # Load the model
        model.load_state_dict(torch.load(".tmp/model.pth"), strict=False)
        # gradient_adapter.load_state_dict(torch.load(".tmp/gradient_adapter.pth"), strict=False)
    except Exception as e:
        print(f"No model found, starting from scratch. {e}")


    # optimizer = SOAP(params=model.parameters(), lr=15e-4, weight_decay=1e-2)
    optimizer = AdamW(params=model.parameters(), lr=1e-4, weight_decay=1e-2)
    # grad_adapter_optimizer = AdamW(params=gradient_adapter.parameters(), lr=1e-4, weight_decay=1e-2)

    lr_scheduler = PolynomialLR(optimizer, total_iters=num_epochs, power=1.0)
    
    dx = 0.01

    for epoch in range(num_epochs):
        if epoch % 16 == 0:
            with DelayedKeyboardInterrupt():
                while True:
                    try:
                        torch.save(model.state_dict(), f".tmp/model.pth")
                        # torch.save(gradient_adapter.state_dict(), f".tmp/gradient_adapter.pth")
                        break
                    except Exception as e:
                        print(f"Failed to save model: {e}")
                        pass # Ignore file access issues or keyboard interrupts

        if (epoch == 0) or (random.random() < 0.05):
            b = assemble_batch()
            num_steps   = num_iters_train
            noise_level = torch.rand(batch_size, 1, 1, 1, device=device)
            noise_level = F.interpolate(noise_level, size=(size, size), mode='bilinear', align_corners=False)
            noise       = torch.randn_like(b)
            noisy_input = b * (1 - noise_level) + noise * noise_level
            _noisy_input = noisy_input.clone()
    
        optimizer.zero_grad()
        loss = 0.0
        # VAE loss
        vae_decoded, latents_mu, latents_logvar = model.gen_vae(b)
        loss = loss + F.mse_loss(vae_decoded, b)
        # KLD loss for latent space
        kld_loss    = -0.5 * (1.0 + latents_logvar - latents_mu.pow(2) - latents_logvar.exp()).mean()
        loss        = loss + 0.05 * kld_loss

        if loss.isnan().any():
            print("NaN loss, exiting.")
            exit(1)

        loss.backward()
        # clip grad
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        optimizer.zero_grad()
        loss = 0.0

        noisy_input             = (noisy_input).clone().detach()
        noisy_input.requires_grad    = True

        for ridx in range(2):
            grad_point                  = noisy_input
            potential                   = model.gen_score(grad_point)
            grad_at_point               = torch.autograd.grad(potential.sum(), grad_point, create_graph=True)[0]
            adapted_grad_at_point       = grad_at_point # + gradient_adapter(grad_point, grad_at_point)
            target_grad                 = -(b - noisy_input)
            # grad_mag                    = target_grad.square().sum(dim=(1,2,3), keepdim=True).sqrt()
            # target_grad                 = torch.where(
            #     grad_mag > 1.0 / 16.0,
            #     target_grad / (1.0e-6 + grad_mag) * (1.0 / 16.0),
            #     target_grad
            # )
            # loss                    = loss     + (grad_at_point - target_grad).square().mean()
            # loss                    = loss     + (potential).square().mean() * 0.05
            # noisy_input             = (noisy_input - adapted_grad_at_point * torch.rand(batch_size, device=device).view(-1, 1, 1, 1))
            noisy_input             = (noisy_input - adapted_grad_at_point)
            loss                    = loss     + (noisy_input - b).square().mean() * 2.0
        


        if (epoch % 16) == 0:
            if 0:
                grad_point          = b.clone().detach()
                grad_point.requires_grad = True
                potential           = model(grad_point)
                grad_at_point       = torch.autograd.grad(potential.sum(), grad_point, create_graph=True)[0]
                
                # Second derivatives (diagonal of Hessian)
                # Hutchinson's trace estimator: use random Rademacher vectors
                # tr(H) is approximately E[z^T H z] where z is a Rademacher vector
                z = dx * (torch.randint_like(grad_at_point, 0, 2).float() * 2.0 - 1.0)  # Random Rademacher vector scaled by dx
                
                # Compute z^T H z = z^T (∇(∇E^T z))
                grad_z_product = (grad_at_point * z).sum()
                
                hessian_z = torch.autograd.grad(grad_z_product, grad_point, create_graph=True)[0]

                # Trace estimate per sample
                trace = (hessian_z * z).view(batch_size, -1).sum(dim=1)

                loss = loss + (trace * 0.1).square().mean()
            
            if 0:
                loss = loss + (model(b) * 0.1).square().mean()



        if loss.isnan().any():
            print("NaN loss, exiting.")
            exit(1)

        loss.backward()
        # clip grad
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # torch.nn.utils.clip_grad_norm_(gradient_adapter.parameters(), 1.0)
        optimizer.step()
        # grad_adapter_optimizer.step()
        lr_scheduler.step()

        if 0:
            num_dirs_to_sample = 2
            dx                 = 0.01
            optimizer.zero_grad()
            # grad_adapter_optimizer.zero_grad()
            loss                = 0.0
            gt_potential        = model(b)
            
            if 0:
                gt_laplacian        = -num_dirs_to_sample * gt_potential
                for i in range(num_dirs_to_sample):
                    dir             = torch.randn_like(potential)
                    gt_laplacian    = gt_laplacian + model(noisy_input + dir * dx)
                loss = loss + (gt_laplacian).square().mean()
            
            loss = loss + (gt_potential).square().mean()
            
            if loss.isnan().any():
                print("NaN loss, exiting.")
                exit(1)

            loss.backward()
            optimizer.step()
            # grad_adapter_optimizer.step()
            lr_scheduler.step()

        if epoch % 16 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            
            # Generate samples using your iterative approach
            # with torch.no_grad():
            if 1:
                stack = torch.zeros((1, 3, 4 * size, batch_size * size), device=device)
                
                for batch_idx in range(batch_size):
                    stack[0, :, 0:size, batch_idx*size:(batch_idx+1)*size]        = 0.5 + 0.5 * _noisy_input[batch_idx:batch_idx+1, :, :, :]
                    stack[0, :, size:size*2, batch_idx*size:(batch_idx+1)*size]   = 0.5 + 0.5 * noisy_input[batch_idx:batch_idx+1, :, :, :]
                    stack[0, :, 2*size:3*size, batch_idx*size:(batch_idx+1)*size] = 0.5 + 0.5 * b[batch_idx:batch_idx+1, :, :, :]
                    stack[0, :, 3*size:4*size, batch_idx*size:(batch_idx+1)*size] = 0.5 + 0.5 * vae_decoded[batch_idx:batch_idx+1, :, :, :]
                dds = dds_from_tensor(stack)
                dds.save(".tmp/input.dds")

                # dds = dds_from_tensor(recurrent_features[0:1, 0:3, :, :] * 0.5 + 0.5)
                # dds.save(".tmp/recurrent_features.dds")

                x = torch.randn(4, 3, 64, 64, device=device)

                # sampling_steps = torch.linspace(num_diffusion_steps - 1, 0, num_iters).long().to(device)
                
                num_inference_steps = num_iters * 4

                viz_batch_size = 4
                viz_size       = size
                _x              = torch.randn((viz_batch_size, 3, viz_size, viz_size), device=device)
                x = _x
                skip = 1
                viz_stack      = torch.zeros((1, 3, (num_inference_steps // skip + 1) * viz_size, viz_batch_size * viz_size), device=device)
                # viz_stack2      = torch.zeros((1, 3, (num_inference_steps // skip + 1) * viz_size, viz_batch_size * viz_size), device=device)
                for batch_idx in range(viz_batch_size):
                    viz_stack[0, :, 0*viz_size:1*viz_size, batch_idx*viz_size:(batch_idx+1)*viz_size] = x[batch_idx:batch_idx+1, :, :, :]
                gif_stack = []
                gif_stack.append(x[0, :, :, :])

                dt = 1.0 / (num_inference_steps - 1)

                # recurrent_features = torch.zeros((viz_batch_size, 4, size, size), device=device)

                prev_x = x
                nesterov_step = 0.0

                for step in range(num_inference_steps):
                    # t_batch = torch.full((viz_batch_size,), step_idx, device=device, dtype=torch.long)
                    t = (step + 1) / (num_inference_steps)
                    grad_point                  = x.detach()
                    grad_point.requires_grad    = True
                    potential                   = model.gen_score(grad_point)
                    grad_at_point               = torch.autograd.grad(potential.sum(), grad_point, create_graph=True)[0]
                    grad_at_point               = grad_at_point # + gradient_adapter(grad_point, grad_at_point)
                    prev_x                      = x
                    x                           = x - grad_at_point * torch.rand(viz_batch_size, device=device).view(-1, 1, 1, 1)

                    sigma = 0.01
                    x = x * (1.0 - sigma) + torch.randn_like(x) * sigma

                    gif_stack.append(0.5 + 0.5 * x[0, :, :, :].detach().cpu())
                    if (step + 1) % skip == 0:
                        for batch_idx in range(viz_batch_size):
                            viz_stack[0, :, (step // skip+1)*viz_size:(step//skip+2)*viz_size, batch_idx*viz_size:(batch_idx+1)*viz_size] = 0.5 + 0.5 * x[batch_idx:batch_idx+1, :, :, :]
                            # viz_stack2[0, :, (step // skip+1)*viz_size:(step//skip+2)*viz_size, batch_idx*viz_size:(batch_idx+1)*viz_size] = 0.5 + 0.5 * recurrent_features[batch_idx:batch_idx+1, 0:3, :, :]
                    

                # scores_0 = model.self_attn_bk_0.attn.last_scores
                # scores_1 = model.self_attn_bk_1.attn.last_scores
                # scores_2 = model.self_attn_bk_2.attn.last_scores
                # scores_3 = model.self_attn_bk_3.attn.last_scores
                
                # num_heads = model.self_attn_bk_0.attn.num_heads
                # attn_size   = 4
                # score_image = torch.zeros((1, 4, attn_size * attn_size, attn_size * attn_size), device=device)

                # for head_idx in range(num_heads):

                #     # print(f"Scores shape: {scores_0.shape}")

                #     for y in range(attn_size):
                #         for x in range(attn_size):
                #             score_image[0:1, 0:1, y * attn_size:(y + 1) * attn_size, x * attn_size:(x + 1) * attn_size] = scores_0[0, head_idx, y * attn_size + x, :].view(attn_size, attn_size)

                #     dds = dds_from_tensor(score_image)
                #     dds.save(f".tmp/scores_{head_idx}.dds")
                    
                # Visualize
                # viz_stack = torch.zeros((1, 3, size, 4 * size), device=device)
                # for i in range(4):
                #     viz_stack[0, :, :, i*size:(i+1)*size] = torch.clamp(x[i], 0, 1)
                
                dds = dds_from_tensor(viz_stack)
                dds.save(".tmp/output.dds")
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