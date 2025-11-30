# MIT License
# Copyright (c) 2025 Anton Schreiner

import sys
sys.path.append('C:\\soft\\ComfyUI\\resources\\ComfyUI')

import torch
from nodes import NODE_CLASS_MAPPINGS
from execution import PromptExecutor
import comfy.samplers
from comfy.ldm.wan.model import WanModel
import numpy as np
from PIL import Image
from third_party.ml.wan.modules.vae2_1 import *
from third_party.ml.wan.modules.vae2_2 import *
from safetensors.torch import load_file as load_safetensors
from third_party.ml.wan.modules.t5 import *
import os
from py.torch_utils import *

import logging

logging.basicConfig(level=logging.DEBUG)

# ops = comfy.ops.manual_cast
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

text_len = 512

text_encoder = T5EncoderModel(
    checkpoint_path="C:\\soft\\ComfyUICache\\models\\text_encoders\\models_t5_umt5-xxl-enc-bf16.pth",
    tokenizer_path="C:\\soft\\ComfyUICache\\models\\text_encoders\\umt5_xxl_tokenizer",
    dtype=torch.float16,
    device=device,
    text_len=text_len
)
# text_encoder.requires_grad_(False)

assert text_encoder is not None

model_options = {}
# model_options["dtype"] = torch.float8_e4m3fn

prompt = "An astronaut riding a horse. 4k. holywood. marvel. detailed. trending on artstation."
text_encoder_output = text_encoder([prompt], device=device)
negative_prompt = ""
uncond_text_encoder_output = text_encoder([negative_prompt], device=device)

del text_encoder

VIDEO_WIDTH = 128
VIDEO_HEIGHT = 128
VIDEO_LENGTH = 17
BATCH_SIZE = 1

src_noisy_latents = torch.randn((BATCH_SIZE, 16, (VIDEO_LENGTH - 1) // 4 + 1, VIDEO_HEIGHT // 8, VIDEO_WIDTH // 8), device=device, dtype=torch.float16)
noisy_latents = src_noisy_latents.clone()

import comfy.sample

if 1:
    # Format conditioning properly
    positive_cond = [[text_encoder_output[0], {}]]
    negative_cond = [[uncond_text_encoder_output[0], {}]]
    high_noise_model = comfy.sd.load_diffusion_model("C:\\soft\\ComfyUICache\\models\\diffusion_models\\wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors", model_options=model_options)

    print(f"Text embedding shape: {text_encoder_output[0].shape}")
    print(f"Text embedding mean: {text_encoder_output[0].mean():.4f}")
    print(f"Text embedding std: {text_encoder_output[0].std():.4f}")

    import comfy.model_sampling

    # Create model sampling with shift
    # model_sampling = comfy.model_sampling.ModelSamplingDiscreteFlow()
    # Apply to your model
    # high_noise_model.model.model_sampling = model_sampling
    high_noise_model.model.model_sampling.shift = 8.0

    NUM_STEPS = 1000

    sampler = comfy.samplers.KSampler(
        model=high_noise_model,
        steps=NUM_STEPS,
        device=device,
        sampler="euler",
        scheduler="simple",
        denoise=0.5,
        model_options=model_options,
    )

    samples = sampler.sample(
        noise=src_noisy_latents,
        positive=positive_cond,
        negative=negative_cond,
        cfg=3.5,
        latent_image=noisy_latents,
        start_step=0,
        last_step=NUM_STEPS // 2,
        force_full_denoise=False,
    )

    noisy_latents = samples

    del high_noise_model

    if 1:
        low_noise_model = comfy.sd.load_diffusion_model("C:\\soft\\ComfyUICache\\models\\diffusion_models\\wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors", model_options=model_options)
        low_noise_model.model.model_sampling.shift = 8.0

        sampler = comfy.samplers.KSampler(
            model=low_noise_model,
            steps=NUM_STEPS,
            device=device,
            sampler="euler",
            scheduler="simple",
            denoise=1.0,
            model_options=model_options,
        )

        samples = sampler.sample(
            noise=src_noisy_latents,
            positive=positive_cond,
            negative=negative_cond,
            cfg=3.5,
            latent_image=noisy_latents,
            start_step=NUM_STEPS // 2,
            last_step=NUM_STEPS,
            force_full_denoise=True,
        )
        del low_noise_model

        noisy_latents = samples

# save first frame of noisy_latents for debugging
if 1:
    dds = dds_from_tensor(noisy_latents[0:1, 0:3, 0, :, :].cpu())
    dds.save(".tmp/latents.dds")

if 0:    
    NUM_DENOISING_STEPS = 16
    print("Loading high noise model...")
    high_noise_model = comfy.sd.load_diffusion_model("C:\\soft\\ComfyUICache\\models\\diffusion_models\\wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors", model_options=model_options)
    print("Starting denoising...")

    for step in range(0, NUM_DENOISING_STEPS):
        t = torch.tensor([step], device=device) / (NUM_DENOISING_STEPS * 2)
        with torch.no_grad():
            cond_grad     = high_noise_model.model.diffusion_model(noisy_latents, t, context=text_encoder_output[0])
            uncond_grad   = high_noise_model.model.diffusion_model(noisy_latents, t, context=uncond_text_encoder_output[0])
            noisy_latents = noisy_latents + 1.0 * (uncond_grad + 3.0 * (cond_grad - uncond_grad)) / NUM_DENOISING_STEPS # simplified denoising step

            # save first frame of noisy_latents for debugging
            if 1:
                dds = dds_from_tensor(noisy_latents[0:1, 0:3, 0, :, :].cpu())
                dds.save(".tmp/latents.dds")
        
        print(f"Step {step} completed")

    del high_noise_model

    low_noise_model = comfy.sd.load_diffusion_model("C:\\soft\\ComfyUICache\\models\\diffusion_models\\wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors", model_options=model_options)
    for step in range(0, NUM_DENOISING_STEPS):
        t = torch.tensor([step + NUM_DENOISING_STEPS], device=device) / (NUM_DENOISING_STEPS * 2)
        with torch.no_grad():
            cond_grad       = low_noise_model.model.diffusion_model(noisy_latents, t, context=text_encoder_output[0])
            uncond_grad     = low_noise_model.model.diffusion_model(noisy_latents, t, context=uncond_text_encoder_output[0])
            noisy_latents   = noisy_latents + 1.0 * (uncond_grad + 3.0 * (cond_grad - uncond_grad)) / NUM_DENOISING_STEPS # simplified denoising step

            if 1:
                dds = dds_from_tensor(noisy_latents[0:1, 0:3, 0, :, :].cpu())
                dds.save(".tmp/latents.dds")

        print(f"Step {step} completed")
    del low_noise_model

# decode
vae = Wan2_1_VAE(
    vae_pth="C:\\soft\\ComfyUICache\\models\\vae\\wan_2.1_vae.safetensors",
    dtype=torch.float16,
    device=device,
)

# vae = Wan2_2_VAE(
#     vae_pth="C:\\Users\\aschr\\Downloads\\wan2.2_vae.safetensors",
#     dtype=torch.float16,
#     device=device,
# )

with torch.autocast(device_type='cuda', dtype=torch.float16):
    reconstructed_videos = vae.decode(noisy_latents)

print("Decoding completed")

# reconstructed_videos shape is typically [B, C, T, H, W] or [B, T, C, H, W]
# Convert to numpy and process
video = reconstructed_videos[0]  # Get first batch item

# Move to CPU and convert to numpy
video = video.cpu().float()

# Normalize to [0, 1] range (adjust based on VAE output range)
video = (video + 1) / 2  # If output is in [-1, 1]
video = torch.clamp(video, 0, 1)

# Convert to [0, 255] uint8
video = (video * 255).byte().numpy()

# Rearrange dimensions to [T, H, W, C] for saving
if video.shape[0] == 3:  # If [C, T, H, W]
    video = np.transpose(video, (1, 2, 3, 0))  # [T, H, W, C]
elif video.shape[-1] == 3:  # If already [T, H, W, C]
    pass
else:
    print(f"Unexpected video shape: {video.shape}")

# Save as images or video file
output_dir = ".tmp"
os.makedirs(output_dir, exist_ok=True)
for i, frame in enumerate(video):
    img = Image.fromarray(frame)
    img.save(f"{output_dir}/frame_{i:04d}.png")
print(f"Saved {len(video)} frames to {output_dir}")

# Optional: Save as video using opencv or imageio
try:
    import imageio
    imageio.mimsave(".tmp/output_video.mp4", video, fps=8)
    print("Saved video to output_video.mp4")
except ImportError:
    print("Install imageio to save as video: pip install imageio[ffmpeg]")

print("Decoding completed")

# reconstructed_videos is the final output

exit(0)

low_noise_dict = load_safetensors("C:\\soft\\ComfyUICache\\models\\diffusion_models\\wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors", device='cpu')
high_noise_dict = load_safetensors("C:\\soft\\ComfyUICache\\models\\diffusion_models\\wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors", device='cpu')

low_noise_model = WanModel(
  dim=5120,
  eps=1e-06,
  ffn_dim=13824,
  freq_dim=256,
  in_dim=16,
  model_type="t2v",
  num_heads=40,
  num_layers=40,
  out_dim=16,
  text_len=512,
  operations=ops,
)
low_noise_model.load_state_dict(low_noise_dict).to(dtype=torch.float8_e4m3fn).to(device)


high_noise_model = WanModel(
  dim=5120,
  eps=1e-06,
  ffn_dim=13824,
  freq_dim=256,
  in_dim=16,
  model_type="t2v",
  num_heads=40,
  num_layers=40,
  out_dim=16,
  text_len=512,
  operations=ops,
)
high_noise_model.load_state_dict(high_noise_dict).to(dtype=torch.float8_e4m3fn).to(device)