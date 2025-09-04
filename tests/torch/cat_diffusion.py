import torch
import torch.nn as nn
import os
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import random
from py.torch_utils import dds_from_tensor
from piq import SSIMLoss
import math
import numpy as np

assert torch.cuda.is_available()
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(torch.cuda.current_device()))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderBlock(nn.Module):
    def __init__(self, channels, groups=1):
        super(EncoderBlock, self).__init__()
        self.dwconv    = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=True, groups=groups)
        self.pwconv1   = nn.Conv2d(in_channels=channels, out_channels=channels * 2, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn1       = nn.BatchNorm2d(channels * 2)
        self.act       = nn.GELU()
        self.pwconv2   = nn.Conv2d(in_channels=channels * 2, out_channels=channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        identity = x
        out      = self.dwconv(x)
        out      = self.pwconv1(out)
        out      = self.bn1(out)
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

# Image Patch Embedding
class PatchEmbed(nn.Module):
    def __init__(self, in_channels, embed_dim, kernel_size=4, padding=0):
        super(PatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, padding=padding)

    def forward(self, x):
        x = self.proj(x)
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

    def forward(self, x):
        x = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = x + self.attn(x)
        x = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = x + self.ffw(x)
        return x

class CatDataloader:
    def __init__(self, path):
        self.paths = []
        self.data  = []
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
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

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.feature_extractor_l0           = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.feature_extractor_l0_encoder_0 = EncoderBlock(channels=32)
        self.feature_extractor_l0_encoder_1 = EncoderBlock(channels=32)
        self.feature_extractor_l0_downsample = DownsampleBlock(in_channels=32, out_channels=64)
        self.feature_extractor_l1_encoder_0 = EncoderBlock(channels=64)
        self.feature_extractor_l1_encoder_1 = EncoderBlock(channels=64)
        self.feature_extractor_l1_downsample = DownsampleBlock(in_channels=64, out_channels=128)
        self.feature_extractor_l2_encoder_0 = EncoderBlock(channels=128)
        self.feature_extractor_l2_encoder_1 = EncoderBlock(channels=128)
        self.feature_extractor_l2_upsample = UpsampleBlock(in_channels=128, out_channels=64)
        self.feature_extractor_l1_decoder_0 = EncoderBlock(channels=64)
        self.feature_extractor_l1_decoder_1 = EncoderBlock(channels=64)
        self.feature_extractor_l1_upsample = UpsampleBlock(in_channels=64, out_channels=32)
        self.feature_extractor_l0_decoder_0 = EncoderBlock(channels=32)

        self.learned_positional_encoding = nn.Parameter(torch.randn(1, 512, size // 16, size // 16))

        self.patch_embed       = PatchEmbed(in_channels=32, embed_dim=512, kernel_size=16)
        self.pixel_patch_embed = PatchEmbed(in_channels=3, embed_dim=512, kernel_size=16)
        self.self_attn_bk_0  = SelfAttentionBlock(embed_dim=512, num_heads=8)
        self.self_attn_bk_1  = SelfAttentionBlock(embed_dim=512, num_heads=8)
        self.self_attn_bk_2  = SelfAttentionBlock(embed_dim=512, num_heads=8)
        self.self_attn_bk_3  = SelfAttentionBlock(embed_dim=512, num_heads=8)

        self.post_attn_upsample_0              = UpsampleBlock(in_channels=512, out_channels=256)
        self.post_attn_upsample_0_decoder_0    = EncoderBlock(channels=256)
        self.post_attn_upsample_0_decoder_1    = EncoderBlock(channels=256)
        self.post_attn_upsample_1              = UpsampleBlock(in_channels=256, out_channels=128)
        self.post_attn_upsample_1_decoder_0    = EncoderBlock(channels=128)
        self.post_attn_upsample_1_decoder_1    = EncoderBlock(channels=128)
        self.post_attn_upsample_2              = UpsampleBlock(in_channels=128, out_channels=64)
        self.post_attn_upsample_2_decoder_0    = EncoderBlock(channels=64)
        self.post_attn_upsample_2_decoder_1    = EncoderBlock(channels=64)
        self.post_attn_upsample_3              = UpsampleBlock(in_channels=64, out_channels=32)
        self.post_attn_upsample_3_decoder_0    = EncoderBlock(channels=32)

        self.output_layer     = nn.Conv2d(in_channels=32, out_channels=4, kernel_size=1)


        self.features = {}

        def get_activation(name):
            def hook(model, input, output):
                self.features[name] = output.detach()
            return hook

        for name, layer in self.named_modules():
            layer.register_forward_hook(get_activation(name))

    def extract_features_from_image(self, img):
        self.features = {}
        x               = self.input_encoder(img)

        l0_encoder_0 = self.l0_encoder_0(x)
        l0_encoder_1 = self.l0_encoder_1(l0_encoder_0)
        l0_downsample = self.l0_downsample(l0_encoder_1)

        l1_encoder_0 = self.l1_encoder_0(l0_downsample)
        l1_encoder_1 = self.l1_encoder_1(l1_encoder_0)
        l1_downsample = self.l1_downsample(l1_encoder_1)

        l2_encoder_0 = self.l2_encoder_0(l1_downsample)

        return l2_encoder_0

    def forward(self, input):
        
        if 1:
            # Feature extraction
            feature_extractor_l0                = self.feature_extractor_l0(input)
            feature_extractor_l0_encoder_0      = self.feature_extractor_l0_encoder_0(feature_extractor_l0)
            feature_extractor_l0_encoder_1      = self.feature_extractor_l0_encoder_1(feature_extractor_l0_encoder_0)
            feature_extractor_l0_downsample     = self.feature_extractor_l0_downsample(feature_extractor_l0_encoder_1)
            feature_extractor_l1_encoder_0      = self.feature_extractor_l1_encoder_0(feature_extractor_l0_downsample)
            feature_extractor_l1_encoder_1      = self.feature_extractor_l1_encoder_1(feature_extractor_l1_encoder_0)
            feature_extractor_l1_downsample     = self.feature_extractor_l1_downsample(feature_extractor_l1_encoder_1)
            feature_extractor_l2_encoder_0      = self.feature_extractor_l2_encoder_0(feature_extractor_l1_downsample)
            feature_extractor_l2_encoder_1      = self.feature_extractor_l2_encoder_1(feature_extractor_l2_encoder_0)
            feature_extractor_l2_upsample       = self.feature_extractor_l2_upsample(feature_extractor_l2_encoder_1) + feature_extractor_l1_encoder_1
            feature_extractor_l1_decoder_0      = self.feature_extractor_l1_decoder_0(feature_extractor_l2_upsample)
            feature_extractor_l1_decoder_1      = self.feature_extractor_l1_decoder_1(feature_extractor_l1_decoder_0)
            feature_extractor_l1_upsample       = self.feature_extractor_l1_upsample(feature_extractor_l1_decoder_1) + feature_extractor_l0_encoder_1
            feature_extractor_l0_decoder_0      = self.feature_extractor_l0_decoder_0(feature_extractor_l1_upsample)

            # Self attention
            patch_embed = self.patch_embed(feature_extractor_l0_decoder_0) + self.learned_positional_encoding
        else:
            patch_embed = self.pixel_patch_embed(input) + self.learned_positional_encoding

        x           = self.self_attn_bk_0(patch_embed)
        x           = self.self_attn_bk_1(x)
        x           = self.self_attn_bk_2(x)
        x           = self.self_attn_bk_3(x)

        x           = self.post_attn_upsample_0(x)
        x           = self.post_attn_upsample_0_decoder_0(x)
        x           = self.post_attn_upsample_0_decoder_1(x)
        x           = self.post_attn_upsample_1(x)
        x           = self.post_attn_upsample_1_decoder_0(x)
        x           = self.post_attn_upsample_1_decoder_1(x)
        x           = self.post_attn_upsample_2(x)
        x           = self.post_attn_upsample_2_decoder_0(x)
        x           = self.post_attn_upsample_2_decoder_1(x)
        x           = self.post_attn_upsample_3(x)
        x           = self.post_attn_upsample_3_decoder_0(x)

        x           = self.output_layer(x)

        y = x[:, 0:3, :, :]
        alpha = x[:, 3:4, :, :]
        alpha = F.sigmoid(alpha)

        return torch.lerp(input, y, alpha)
        # return input + y * alpha

# print(assemble_batch().shape)

model     = Model().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

num_epochs = 10000
batch_size = 64
num_iters_train  = 64
num_iters  = 64

def assemble_batch():
    batch = dataset.next(batch_size=batch_size)
    return batch.to(device)

try: # Load the model
    model.load_state_dict(torch.load(".tmp/model.pth"), strict=False)
except Exception as e:
    print("No model found, starting from scratch.")

# import copy
# cloned_model = copy.deepcopy(model)
# # Disable gradients for the cloned model
# for param in cloned_model.parameters():
#     param.requires_grad = False

for epoch in range(num_epochs):
    # Training code here

    b     = assemble_batch()
    loss  = 0.0
    
    # make sure we're I(x) when it comes to the actual attractors
    optimizer.zero_grad()
    o           = model(b)
    loss        = (F.mse_loss(o, b) + SSIMLoss()(o.clamp(0, 1), b.clamp(0, 1))) * 4.0
    loss.backward()
    optimizer.step()

    noise_level = torch.rand(batch_size, 1, 4, 4, device=device)

    noise_level = torch.where(
        noise_level > 0.5,
        torch.ones_like(noise_level),
        torch.ones_like(noise_level) * 0.1
    )

    # Upscale noise level
    noise_level = F.interpolate(noise_level, size=(size, size), mode='bilinear', align_corners=False)
    # noise_level = 0.9
    o       = torch.lerp(b, torch.randn_like(b), noise_level)
    # o = torch.randn_like(b)
    i = o
    
    # o = torch.randn_like(b)
    # o = torch.lerp(b, torch.randn_like(b), (noise_level))
    # o = b * 0.5 + torch.randn_like(b) * 0.5
    # params = {name: param for name, param in cloned_model.named_parameters()}
    # for name, param in model.named_parameters():
    #     params[name].data.copy_(param.data)

    chain = 1 # random.randint(1, 3)
    iters = num_iters_train # random.randint(1, num_iters_train // chain) * chain

    for iter in range(iters // chain):
        loss  = 0.0
        o = o.detach()
        optimizer.zero_grad()
        for chidx in range(chain):
            o           = model(o)
            schedule_weight = (math.exp(4.0 * (iter * chain + chidx - iters ) / (iters))) * 16.0
            loss        = loss + (F.mse_loss(o, b) + SSIMLoss()(o.clamp(0, 1), b.clamp(0, 1))) * schedule_weight
            # Feature loss
            # ref_features = cloned_model.extract_features_from_image(b).detach()
            # tst_features = cloned_model.extract_features_from_image(o)
            # loss        = loss + (1.0 - F.cosine_similarity(ref_features, tst_features)).mean() * schedule_weight
            # loss        = loss + (F.mse_loss(o, b) + SSIMLoss()(o.clamp(0, 1), b.clamp(0, 1))) * ((num_iters_train - iter * chain + chidx) / (num_iters_train)) * 1.0
            # loss        = loss + (F.mse_loss(o, b) + SSIMLoss()(o.clamp(0, 1), b.clamp(0, 1)))
        
        loss.backward()
        optimizer.step()

        # Enforce diversity
        # idx_0 = random.randint(0, batch_size - 1)
        # idx_1 = random.randint(0, batch_size - 1)
        # if idx_1 != idx_0:
        #     img_0 = o[idx_0:idx_0+1, :, :, :]
        #     img_1 = o[idx_1:idx_1+1, :, :, :]
        #     optimizer.zero_grad()
        #     loss        = loss + torch.exp(-F.mse_loss(img_0, img_1)) * 1.0
        #     loss.backward()
        #     optimizer.step()
        # else:
        #     loss.backward()
        #     optimizer.step()
        # loss.backward()
        # optimizer.step()

    if epoch % 16 == 0:
        stack = torch.zeros((1, 3, 3 * size, batch_size * size), device=device)
        for batch_idx in range(batch_size):
            stack[0, :, 0:size, batch_idx*size:(batch_idx+1)*size]        = i[batch_idx:batch_idx+1, :, :, :]
            stack[0, :, size:size*2, batch_idx*size:(batch_idx+1)*size]   = o[batch_idx:batch_idx+1, :, :, :]
            stack[0, :, 2*size:3*size, batch_idx*size:(batch_idx+1)*size] = b[batch_idx:batch_idx+1, :, :, :]

        dds = dds_from_tensor(stack)
        dds.save(".tmp/input.dds")



    

    # x = torch.randn_like(b)
    # for iter in range(num_iters):
    #     noise_level = 1.0 - (iter + 1) / num_iters
    #     noisy       = torch.lerp(b, b + torch.randn_like(b), noise_level)
    #     o           = model(x)
    #     loss        = loss + F.mse_loss(o, noisy) + SSIMLoss()(o.clamp(0, 1), noisy.clamp(0, 1))
    


    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

    if epoch % 16 == 0:
        with torch.no_grad():
            viz_batch_size = 4
            viz_size       = size
            _x              = torch.randn((viz_batch_size, 3, viz_size, viz_size), device=device)
            x = _x
            viz_stack      = torch.zeros((1, 3, (num_iters + 1) * viz_size, viz_batch_size * viz_size), device=device)
            for batch_idx in range(viz_batch_size):
                viz_stack[0, :, 0*viz_size:1*viz_size, batch_idx*viz_size:(batch_idx+1)*viz_size] = x[batch_idx:batch_idx+1, :, :, :]
            gif_stack = []
            gif_stack.append(x[0, :, :, :])
            for iter in range(num_iters):
                x = model(x)
                # if iter == 0:
                #     x = torch.lerp(x, _x, 0.1)
                for batch_idx in range(viz_batch_size):
                    viz_stack[0, :, (iter+1)*viz_size:(iter+2)*viz_size, batch_idx*viz_size:(batch_idx+1)*viz_size] = x[batch_idx:batch_idx+1, :, :, :]
                
                gif_stack.append(x[0, :, :, :])

            scores_0 = model.self_attn_bk_0.attn.last_scores
            scores_1 = model.self_attn_bk_1.attn.last_scores
            scores_2 = model.self_attn_bk_2.attn.last_scores
            scores_3 = model.self_attn_bk_3.attn.last_scores
            
            num_heads = model.self_attn_bk_0.attn.num_heads
            attn_size   = 4
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

            # save gif
            # import imageio
            from PIL import Image
            # imageio.mimsave(".tmp/output.gif", gif_stack, fps=2)
            # Convert tensors to PIL Images
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

        # Save the model checkpoint
        torch.save(model.state_dict(), f".tmp/model.pth")
