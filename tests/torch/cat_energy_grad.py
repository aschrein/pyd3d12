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
    def __init__(self, channels, groups=1, skip=True):
        super(EncoderBlock, self).__init__()
        self.dwconv    = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=True, groups=groups)
        self.pwconv1   = nn.Conv2d(in_channels=channels, out_channels=channels * 2, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn1       = nn.BatchNorm2d(channels * 2)
        self.act       = nn.GELU()
        # self.act       = nn.LeakyReLU(0.01, inplace=True)
        self.pwconv2   = nn.Conv2d(in_channels=channels * 2, out_channels=channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.skip      = skip

    def forward(self, x):
        identity = x
        out      = self.dwconv(x)
        out      = self.pwconv1(out)
        out      = self.bn1(out)
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
            Swiglu(embed_dim),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        )
        self.k      = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            Swiglu(embed_dim),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        )
        self.v      = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            Swiglu(embed_dim),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        )
        self.fc     = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            Swiglu(embed_dim),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        )
    
        self.last_scores = None

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.q(x), self.k(x), self.v(x)
        
        k = apply_rope_2d(k)
        q = apply_rope_2d(q)

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
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
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

class VAE(nn.Module):
    def __init__(self, image_channels=3):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 512, kernel_size=16, stride=16, padding=0),
            nn.BatchNorm2d(512),
            EncoderBlock(channels=512),
        )

        # Latent space projections
        self.fc_mu      = nn.Conv2d(512, 512, kernel_size=1, padding=0)
        self.fc_logvar  = nn.Conv2d(512, 512, kernel_size=1, padding=0)

        # Bottleneck with attention (your existing attention blocks)
        # self.learned_positional_encoding = nn.Parameter(torch.randn(1, 512, 4, 4))
        self.self_attn_bk = nn.Sequential(*[
            SelfAttentionBlock(embed_dim=512, num_heads=8) for _ in range(4)
        ])

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(out_channels=512, in_channels=512, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(512),
            EncoderBlock(channels=512),
            nn.ConvTranspose2d(out_channels=256, in_channels=512, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(256),
            EncoderBlock(channels=256),
            nn.ConvTranspose2d(out_channels=128, in_channels=256, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(128),
            EncoderBlock(channels=128),
            nn.ConvTranspose2d(out_channels=64, in_channels=128, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(64),
            EncoderBlock(channels=64),
        )

        self.output_conv = nn.ConvTranspose2d(out_channels=3, in_channels=64, kernel_size=1, stride=1, padding=0)

    def encode(self, x, sample=True):
        e = self.encoder(x)
        latents_mu     = self.fc_mu(e)
        latents_logvar = self.fc_logvar(e)

        # Reparameterization trick
        if sample:
            std = torch.exp(0.5 * latents_logvar)
            eps = torch.randn_like(std)
            z   = latents_mu + eps * std
        else:
            z = latents_mu

        return z, latents_mu, latents_logvar

    def decode(self, x):
        z = self.self_attn_bk(x)

        d = self.decoder(z)
        d = self.output_conv(d)
        return d

    def forward(self, x):
        assert False, "Not implemented"


class VAEDiffusion(nn.Module):
    def __init__(self, latent_dim=512):
        super(VAEDiffusion, self).__init__()

        self.dm_convs = nn.Sequential(*[
            EncoderBlock(channels=512),
            nn.BatchNorm2d(512),
            EncoderBlock(channels=512),
            nn.BatchNorm2d(512),
            EncoderBlock(channels=512),
            nn.BatchNorm2d(512),
        ])
        self.dm_self_attn_bk = nn.Sequential(*[
            SelfAttentionBlock(embed_dim=512, num_heads=8) for _ in range(4)
        ])
        self.dm_e_conv_3 = EncoderBlock(channels=512)

        self.velocity_decoder = nn.Sequential(
            nn.Conv2d(in_channels=512 + 512, out_channels=512, kernel_size=3, stride=1, padding=1),
            EncoderBlock(channels=512),
            EncoderBlock(channels=512),
            EncoderBlock(channels=512),
        )


    def get_grad(self, _z):
        z = _z
        B, C, H, W = z.shape
        z = self.dm_convs(z)
        z = self.dm_self_attn_bk(z)
        z = self.dm_e_conv_3(z)

        velocity_input = torch.cat([_z, z], dim=1)
        z = self.velocity_decoder(velocity_input)
        return z

    def forward(self, x):
        assert False, "Not implemented"


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
                if random.random() < 0.5:
                    image = torch.flip(image, [2]) # vertical flip
                angle = random.choice([0, 90, 180, 270])
                if angle != 0:
                    image = torch.rot90(image, k=angle // 90, dims=[2, 3])
                
                if random.random() < 0.5:
                    # zoom
                    scale = random.uniform(1.0, 2.2)
                    new_size = int(64 * scale)
                    image = F.interpolate(image, size=(new_size, new_size), mode='bicubic', align_corners=False)
                    crop_x = random.randint(0, new_size - 64)
                    crop_y = random.randint(0, new_size - 64)
                    image = image[:, :, crop_x:crop_x+64, crop_y:crop_y+64]

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

num_epochs          = 10000
batch_size          = 256
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

kodim = transforms.ToTensor()(Image.open(".tmp\\kodim23.png")).unsqueeze(0).to(device) * 2 - 1
# kodim = kodim[:, :, 0:512, 100:100+512]
kodim = kodim[:, :, 0:512, 0:512]

if 1:
    
    vae_model = VAE().to(device)
    diff_model = VAEDiffusion().to(device)

    try: # Load the model
        vae_model.load_state_dict(torch.load(".tmp/vae_model.pth"), strict=False)
        diff_model.load_state_dict(torch.load(".tmp/diff_model.pth"), strict=False)
    except Exception as e:
        print("No model found, starting from scratch.")


    # optimizer = SOAP(params=model.parameters(), lr=15e-4, weight_decay=1e-2)
    vae_optimizer = AdamW(params=vae_model.parameters(), lr=1e-4, weight_decay=1e-2)
    diff_optimizer = AdamW(params=diff_model.parameters(), lr=1e-4, weight_decay=1e-2)

    lr_scheduler = PolynomialLR(vae_optimizer, total_iters=num_epochs, power=1.0)
    lr_scheduler2 = PolynomialLR(diff_optimizer, total_iters=num_epochs, power=1.0)

    lpips  = LPIPS().to(device)

    do_neg_training = False
    # model_param_checkpoint = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        if epoch % 16 == 0:
            with DelayedKeyboardInterrupt():
                while True:
                    try:
                        torch.save(vae_model.state_dict(), f".tmp/vae_model.pth")
                        torch.save(diff_model.state_dict(), f".tmp/diff_model.pth")
                        break
                    except Exception as e:
                        pass # Ignore file access issues or keyboard interrupts

        # if (epoch == 0) or (random.random() < 0.15):
        if 1:

            # if (epoch != 0) and do_neg_training:
            #     for p, cp in zip(model.state_dict().values(), model_param_checkpoint.values()):
            #         param_grad = p.data - cp.data
            #         p.data     = cp.data - param_grad * 0.25 # go in the opposite direction


            # do_neg_training = False
            b = assemble_batch()
            # recurrent_features = torch.zeros((batch_size, 4, size, size), device=device)

            

            # if random.random() < 0.05:
            #     print("Using NEON training step.")
            #     do_neg_training = True
            #     gend = torch.randn_like(b)
            #     with torch.no_grad():
            #         for i in range(4):
            #             grad, recurrent_features = model(gend, recurrent_features)
            #             gend = gend + grad
            #     b = gend.detach()
            #     model_param_checkpoint = copy.deepcopy(model.state_dict())
            #     recurrent_features = torch.zeros((batch_size, 4, size, size), device=device)

            # num_steps   = num_iters_train
            # if random.random() < 0.5:
            #     noise_level = torch.rand(batch_size, 1, 4, 4, device=device)
            # else:
            noise_level = torch.rand(batch_size, 1, 4, 4, device=device)
            # noise_level = F.interpolate(noise_level, size=(size // 16, size // 16), mode='bilinear', align_corners=False)

            z_src, latents_mu, latents_logvar = vae_model.encode(b, sample=False)
            z_src = z_src.detach()
            z = z_src
            noise       = torch.randn_like(z)
            z = z * (1 - noise_level) + noise * noise_level

        iter = 0

        # VAE model step
        vae_optimizer.zero_grad()
        loss            = 0.0

        # VAE encoder
        _z, latents_mu, latents_logvar = vae_model.encode(b, sample=True)
        decoded_ref = vae_model.decode(_z)

        loss = loss + (decoded_ref - b).square().mean() * 1.0
        loss = loss + lpips(torch.clamp(b * 0.5 + 0.5, 0, 1), torch.clamp(decoded_ref * 0.5 + 0.5, 0, 1)) * 1.5

         # KLD loss for latent space
        kld_loss    = -0.5 * (1.0 + latents_logvar - latents_mu.pow(2) - latents_logvar.exp()).mean()
        loss        = loss + 0.5 * kld_loss
        
        if loss.isnan().any():
            print("NaN loss, exiting.")
            exit(1)

        loss.backward()
        vae_optimizer.step()
        
        # Diffusion model step
        loss            = 0.0
        diff_optimizer.zero_grad()
        loss           = 0.0
        grad  = diff_model.get_grad(z)
        z = z + grad
        # z = grad
        decoded = vae_model.decode(z)
        k =  (math.exp(iter / 16.0))
        loss            = loss     + lpips(torch.clamp(b * 0.5 + 0.5, 0, 1), torch.clamp(decoded * 0.5 + 0.5, 0, 1)) * k
        # loss            = loss + (decoded - noisy_input).square().mean() * 0.15 # penalize big steps
        loss            = loss + (decoded - b).square().mean() * k * 0.25 # penalize distance to target
        loss            = loss + (grad - (z_src - z)).square().mean() * k * 1.0 #
        decoded         = decoded.detach()
        z               = z.detach()
        
        if loss.isnan().any():
            print("NaN loss, exiting.")
            exit(1)

        sigma = 0.05
        z     = z * (1.0 - sigma) + torch.randn_like(z) * sigma

        loss.backward()
        diff_optimizer.step()
        
        # Equilibium loss
        diff_optimizer.zero_grad()
        loss           = 0.0
        grad           = diff_model.get_grad(z_src.detach())
        loss           = loss + (grad).square().mean() * k * 1.0 #
        
        if loss.isnan().any():
            print("NaN loss, exiting.")
            exit(1)

        loss.backward()
        diff_optimizer.step()
        
        iter = iter + 1

        # mixin noise
        # sigma = 0.05
        # z     = z * (1.0 - sigma) + torch.randn_like(z) * sigma
        
        
        # recurrent_features     = recurrent_features.detach()
        

        lr_scheduler.step()
        lr_scheduler2.step()


        if epoch % 16 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            
            # Generate samples using your iterative approach
            with torch.no_grad():
            # if 1:
            
                stack = torch.zeros((1, 3, 3 * size, batch_size * size), device=device)
                
                for batch_idx in range(batch_size):
                    stack[0, :, 0:size, batch_idx*size:(batch_idx+1)*size]        = 0.5 + 0.5 * decoded[batch_idx:batch_idx+1, :, :, :]
                    stack[0, :, size:size*2, batch_idx*size:(batch_idx+1)*size]   = 0.5 + 0.5 * decoded_ref[batch_idx:batch_idx+1, :, :, :]
                    stack[0, :, 2*size:3*size, batch_idx*size:(batch_idx+1)*size] = 0.5 + 0.5 * b[batch_idx:batch_idx+1, :, :, :]
                dds = dds_from_tensor(stack)
                dds.save(".tmp/input.dds")

                # dds = dds_from_tensor(recurrent_features[0:1, 0:3, :, :] * 0.5 + 0.5)
                # dds.save(".tmp/recurrent_features.dds")
                
                x = torch.randn(4, 3, 64, 64, device=device)

                # sampling_steps = torch.linspace(num_diffusion_steps - 1, 0, num_iters).long().to(device)
                
                num_inference_steps = num_iters * 4

                viz_batch_size = 8
                viz_size       = size * 1
                _x              = torch.randn((viz_batch_size, 3, viz_size, viz_size), device=device)
                x = _x
                # x = kodim[:, :, :viz_size, :viz_size]
                skip = 4
                viz_stack      = torch.zeros((1, 3, (num_inference_steps // skip + 1) * viz_size, viz_batch_size * viz_size), device=device)
                # viz_stack2      = torch.zeros((1, 3, (num_inference_steps // skip + 1) * viz_size, viz_batch_size * viz_size), device=device)
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
                inf_z = torch.randn((viz_batch_size, 512, 4, 4), device=device)

                for step in range(num_inference_steps):
                    # t_batch = torch.full((viz_batch_size,), step_idx, device=device, dtype=torch.long)
                    t = (step + 1) / (num_inference_steps)
                    # grad = model(x)
                    grad                   = diff_model.get_grad(inf_z)
                    inf_z                      = inf_z + grad
                    # inf_z                      = grad
                    x                      = vae_model.decode(inf_z)
                    prev_x                 = x
                    # x                      = x + grad * torch.rand(viz_batch_size, device=device).view(-1, 1, 1, 1)
                    # x                      = x + grad * 1
                    if step < num_inference_steps - 1:
                        sigma = 0.1
                        # x = x * (1.0 - sigma) + torch.randn_like(x) * sigma
                        inf_z = inf_z * (1.0 - sigma) + torch.randn_like(inf_z) * sigma

                    gif_stack.append(torch.nn.functional.avg_pool2d(0.5 + 0.5 * x[0, :, :, :].detach().cpu(), kernel_size=2, stride=2))
                    if (step + 1) % skip == 0:
                        for batch_idx in range(viz_batch_size):
                            viz_stack[0, :, (step // skip+1)*viz_size:(step//skip+2)*viz_size, batch_idx*viz_size:(batch_idx+1)*viz_size] = 0.5 + 0.5 * x[batch_idx:batch_idx+1, :, :, :]
                            # viz_stack2[0, :, (step // skip+1)*viz_size:(step//skip+2)*viz_size, batch_idx*viz_size:(batch_idx+1)*viz_size] = 0.5 + 0.5 * inf_recurrent_features[batch_idx:batch_idx+1, 0:3, :, :]
                    

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