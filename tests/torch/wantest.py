# MIT License
# Copyright (c) 2025 Anton Schreiner


from idlelib import config
import torch
from third_party.ml.wan.modules.vae2_1 import *
from safetensors.torch import load_file as load_safetensors
from third_party.ml.wan.modules.t5 import *
from third_party.ml.wan.modules.model import WanModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vae = Wan2_1_VAE(
    vae_pth="C:\\soft\\ComfyUICache\\models\\vae\\wan_2.1_vae.safetensors",
    dtype=torch.float16,
    device=device,
)

text_len = 512

text_encoder = T5EncoderModel(
    checkpoint_path="C:\\soft\\ComfyUICache\\models\\text_encoders\\models_t5_umt5-xxl-enc-bf16.pth",
    tokenizer_path="C:\\soft\\ComfyUICache\\models\\text_encoders\\umt5_xxl_tokenizer",
    dtype=torch.bfloat16,
    device=device,
    text_len=text_len
)

assert vae is not None
assert text_encoder is not None

low_noise_dict = load_safetensors("C:\\soft\\ComfyUICache\\models\\diffusion_models\\wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors", device='cpu')

# print(low_noise_dict["scaled_fp8"])
# print(low_noise_dict["text_embedding.0.scale_input"])

for k, v in low_noise_dict.items():
    if "scale_weight" in k or "scale_input" in k or "scaled_fp8" in k:
        print(k, v)

# exit(1)

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
  text_len=512
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
  text_len=512
)
high_noise_model.load_state_dict(load_safetensors("C:\\soft\\ComfyUICache\\models\\diffusion_models\\wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors", device='cpu')).to(dtype=torch.float8_e4m3fn).to(device)


