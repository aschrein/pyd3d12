# MIT License
# Copyright (c) 2025 Anton Schreiner

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms
import random
from einops import rearrange


torch.manual_seed(42)


class EggrollModel(nn.Module):
    """Simple model for testing EGGROLL training"""
    
    def __init__(self, num_patches=64, patch_size=8, hidden_dim=64):
        super().__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        
        # Learnable patches: [64, 3, 8, 8]
        self.learnable_patches = nn.Parameter(torch.randn(num_patches, 3, patch_size, patch_size))
        
        # Simple network to predict patch weights from input
        self.patch_proj = nn.Linear(3 * patch_size * patch_size, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, num_patches)
        
    def forward(self, x):
        """
        x: [batch, 64, 64, 3] HWC image
        returns: [batch, 64, 64, 3] reconstructed image
        """
        batch_size = x.shape[0]
        
        # Extract patches: [batch, 64, 64, 3] -> [batch, 64, 192]
        patches = rearrange(x, 'b (h ph) (w pw) c -> b (h w) (ph pw c)', 
                           ph=self.patch_size, pw=self.patch_size)
        
        # Compute patch weights
        h = torch.tanh(self.patch_proj(patches))  # [batch, 64, hidden_dim]
        weights = self.out_proj(h)  # [batch, 64, 64]
        # weights = torch.softmax(weights, dim=-1)  # Softmax over patches
        weights = weights.tanh()


        # Blend learnable patches using weights
        # learnable_patches: [64, 3, 8, 8] -> [64, 192]
        lp_flat = rearrange(self.learnable_patches, 'p c ph pw -> p (ph pw c)')
        blended = torch.matmul(weights, lp_flat)  # [batch, 64, 192]
        
        # Reshape to image: [batch, 64, 192] -> [batch, 64, 64, 3]
        output = rearrange(blended, 'b (h w) (ph pw c) -> b (h ph) (w pw) c',
                          h=8, w=8, ph=self.patch_size, pw=self.patch_size, c=3)
        
        return output


class EggrollTrainer:
    """
    EGGROLL trainer implementing the paper's algorithm.
    
    Key formula (Equation 8):
        μ_{t+1} = μ_t + (α / N) * Σ E_i * f_i
    
    Where E_i = (1/sqrt(r)) * A_i @ B_i.T is a rank-r perturbation.
    """
    
    def __init__(self, model, lr=0.01, sigma=0.1, rank=1, 
                 use_adam=False, beta1=0.9, beta2=0.999, eps=1e-8):
        self.model = model
        self.lr = lr
        self.sigma = sigma  # Perturbation scale
        self.rank = rank
        
        # Adam state (optional)
        self.use_adam = use_adam
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        
        # Initialize Adam moments if needed
        if use_adam:
            self.m = {}  # First moment
            self.v = {}  # Second moment
            for name, param in model.named_parameters():
                self.m[name] = torch.zeros_like(param, dtype=torch.float32)
                self.v[name] = torch.zeros_like(param, dtype=torch.float32)
    
    def get_rank1_perturbation(self, param, seed, batch_idx, layer_id):
        """
        Generate rank-1 perturbation: E = A @ B.T / sqrt(r)
        
        For a parameter of shape (m, n) or reshaped to (m, n):
        - A is a random vector of shape (m,)
        - B is a random vector of shape (n,)
        - E = outer(A, B) / sqrt(rank)
        """
        gen = torch.Generator(device=param.device)
        # Unique seed per (batch, layer, global_seed)
        gen.manual_seed(seed + batch_idx * 10000 + layer_id * 100)
        
        # Reshape param to 2D for outer product
        if param.dim() == 1:
            # For 1D params (biases), treat as (n, 1)
            m, n = param.shape[0], 1
        elif param.dim() == 2:
            m, n = param.shape
        else:
            # For higher-dim params, reshape to 2D
            m = param.shape[0]
            n = param.numel() // m
        
        # Generate random vectors A and B
        A = torch.randn(m, generator=gen, device=param.device, dtype=torch.float32)
        B = torch.randn(n, generator=gen, device=param.device, dtype=torch.float32)
        
        # Rank-1 perturbation: outer(A, B) / sqrt(rank)
        # For rank > 1, we'd sum multiple outer products
        E = torch.outer(A, B) / math.sqrt(self.rank)
        
        # Scale by sigma
        E = self.sigma * E
        
        # Reshape back to param shape
        E = E.view(param.shape)
        
        return E
    
    def forward_with_perturbation(self, x, seed, batch_idx):
        """Forward pass with perturbations applied to all parameters"""
        original_params = {}
        
        # Apply perturbations
        for layer_id, (name, param) in enumerate(self.model.named_parameters()):
            original_params[name] = param.data.clone()
            E = self.get_rank1_perturbation(param, seed, batch_idx, layer_id)
            param.data = param.data + E.to(param.dtype)
        
        # Forward pass
        output = self.model(x)
        
        # Restore original parameters
        for name, param in self.model.named_parameters():
            param.data = original_params[name]
        
        return output
    
    def compute_fitness(self, outputs, targets, method='zscore'):
        """
        Compute fitness values from outputs.
        
        methods:
        - 'raw': Raw negative MSE (higher = better)
        - 'zscore': Z-scored fitness (as in paper's LLM experiments)
        - 'rank': Rank-based fitness normalization
        """
        # MSE per sample
        mse = ((outputs - targets) ** 2).mean(dim=tuple(range(1, outputs.dim())))
        
        # Convert to fitness (higher = better)
        fitness = -mse
        
        if method == 'raw':
            return fitness
        elif method == 'zscore':
            # Z-score normalization
            mean = fitness.mean()
            std = fitness.std() + 1e-8
            return (fitness - mean) / std
        elif method == 'rank':
            # Rank-based normalization to [-1, 1]
            N = fitness.shape[0]
            ranks = fitness.argsort().argsort().float()  # 0 to N-1
            return 2 * ranks / (N - 1) - 1  # Map to [-1, 1]
        else:
            return fitness
    
    def compute_gradient_estimate(self, seed, batch_size, fitnesses):
        """
        Compute gradient estimate: g = (1/N) * Σ E_i * f_i
        
        This is the core EGGROLL update (without learning rate).
        """
        gradients = {}
        
        for layer_id, (name, param) in enumerate(self.model.named_parameters()):
            grad = torch.zeros_like(param, dtype=torch.float32)
            
            for b in range(batch_size):
                E = self.get_rank1_perturbation(param, seed, b, layer_id)
                grad += E * fitnesses[b].item()
            
            grad /= batch_size
            gradients[name] = grad
        
        return gradients
    
    def apply_update(self, gradients):
        """
        Apply the EGGROLL update to parameters.
        
        Without Adam: param += lr * grad
        With Adam: Use Adam optimizer on the gradient estimate
        """
        if self.use_adam:
            self.t += 1
            
            for name, param in self.model.named_parameters():
                grad = gradients[name]
                
                # Update biased moments
                self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
                self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (grad ** 2)
                
                # Bias correction
                m_hat = self.m[name] / (1 - self.beta1 ** self.t)
                v_hat = self.v[name] / (1 - self.beta2 ** self.t)
                
                # Adam update
                update = self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
                param.data = param.data + update.to(param.dtype)
        else:
            # Simple SGD-style update
            for name, param in self.model.named_parameters():
                update = self.lr * gradients[name]
                param.data = param.data + update.to(param.dtype)
    
    def train_step(self, images, targets, seed, fitness_method='zscore'):
        """
        Single EGGROLL training step.
        
        1. Forward pass each batch element with different perturbations
        2. Compute fitness values
        3. Estimate gradient from weighted perturbations
        4. Apply update (optionally with Adam)
        """
        batch_size = images.shape[0]
        
        # Forward pass for each batch element with unique perturbation
        outputs = []
        for b in range(batch_size):
            out = self.forward_with_perturbation(images[b:b+1], seed, b)
            outputs.append(out)
        outputs = torch.cat(outputs, dim=0)
        
        # Compute fitness
        fitnesses = self.compute_fitness(outputs, targets, method=fitness_method)
        
        # Estimate gradient
        gradients = self.compute_gradient_estimate(seed, batch_size, fitnesses)
        
        # Apply update
        self.apply_update(gradients)
        
        # Return info for logging
        mse = ((outputs - targets) ** 2).mean(dim=tuple(range(1, outputs.dim())))
        return {
            'outputs': outputs,
            'mse': mse,
            'fitnesses': fitnesses,
            'best_mse': mse.min().item(),
            'mean_mse': mse.mean().item(),
        }


class EggrollTrainerBatched:
    """
    Batched EGGROLL trainer - more efficient by vectorizing operations.
    
    Instead of looping over batch elements, this version batches the
    perturbation generation and fitness computation.
    """
    
    def __init__(self, model, lr=0.01, sigma=0.1, rank=1,
                 use_adam=False, beta1=0.9, beta2=0.999, eps=1e-8):
        self.model = model
        self.lr = lr
        self.sigma = sigma
        self.rank = rank
        
        self.use_adam = use_adam
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        
        if use_adam:
            self.m = {}
            self.v = {}
            for name, param in model.named_parameters():
                self.m[name] = torch.zeros_like(param, dtype=torch.float32)
                self.v[name] = torch.zeros_like(param, dtype=torch.float32)
        
        # Cache for perturbations during a step
        self._perturbation_cache = {}
    
    def generate_all_perturbations(self, seed, batch_size):
        """Generate all perturbations for all params and batch elements"""
        self._perturbation_cache = {}
        
        for layer_id, (name, param) in enumerate(self.model.named_parameters()):
            # Determine shape for outer product
            if param.dim() == 1:
                m, n = param.shape[0], 1
            elif param.dim() == 2:
                m, n = param.shape
            else:
                m = param.shape[0]
                n = param.numel() // m
            
            # Generate A and B matrices for all batch elements
            # A: [batch_size, m], B: [batch_size, n]
            gen = torch.Generator(device=param.device)
            gen.manual_seed(seed + layer_id * 100000)
            
            A = torch.randn(batch_size, m, generator=gen, device=param.device, dtype=torch.float32)
            B = torch.randn(batch_size, n, generator=gen, device=param.device, dtype=torch.float32)
            
            # Store for later use
            self._perturbation_cache[name] = {
                'A': A,
                'B': B,
                'm': m,
                'n': n,
                'shape': param.shape
            }
    
    def get_perturbation_batch(self, name, batch_indices):
        """Get perturbations for specific batch indices"""
        cache = self._perturbation_cache[name]
        A = cache['A'][batch_indices]  # [len(indices), m]
        B = cache['B'][batch_indices]  # [len(indices), n]
        
        # Batched outer product: [batch, m, n]
        E = torch.bmm(A.unsqueeze(2), B.unsqueeze(1)) / math.sqrt(self.rank)
        E = self.sigma * E
        
        # Reshape to param shape
        E = E.view(-1, *cache['shape'])
        return E
    
    def forward_batched(self, x, seed, batch_size):
        """
        Batched forward pass where each sample gets different perturbations.
        
        This is more complex because we need to apply different perturbations
        to different batch elements. For now, we still loop but cache perturbations.
        """
        self.generate_all_perturbations(seed, batch_size)
        
        outputs = []
        for b in range(batch_size):
            # Apply perturbations for this batch element
            original_params = {}
            for name, param in self.model.named_parameters():
                original_params[name] = param.data.clone()
                E = self.get_perturbation_batch(name, torch.tensor([b]))
                param.data = param.data + E[0].to(param.dtype)
            
            # Forward
            out = self.model(x[b:b+1])
            outputs.append(out)
            
            # Restore
            for name, param in self.model.named_parameters():
                param.data = original_params[name]
        
        return torch.cat(outputs, dim=0)
    
    def compute_gradient_batched(self, fitnesses):
        """Compute gradient using cached perturbations"""
        gradients = {}
        batch_size = fitnesses.shape[0]
        
        for name, param in self.model.named_parameters():
            cache = self._perturbation_cache[name]
            A = cache['A']  # [batch, m]
            B = cache['B']  # [batch, n]
            
            # Weighted sum: Σ f_i * (A_i @ B_i.T)
            # = (A.T @ diag(f)) @ B = (A * f.unsqueeze(1)).T @ B
            f = fitnesses.view(-1, 1)  # [batch, 1]
            
            # [m, batch] @ [batch, n] = [m, n]
            weighted_A = (A * f).T  # [m, batch]
            grad = torch.mm(weighted_A, B) / math.sqrt(self.rank)
            grad = self.sigma * grad / batch_size
            
            # Reshape to param shape
            grad = grad.view(param.shape)
            gradients[name] = grad
        
        return gradients
    
    def apply_update(self, gradients):
        """Apply update with optional Adam"""
        if self.use_adam:
            self.t += 1
            for name, param in self.model.named_parameters():
                grad = gradients[name]
                self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
                self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (grad ** 2)
                
                m_hat = self.m[name] / (1 - self.beta1 ** self.t)
                v_hat = self.v[name] / (1 - self.beta2 ** self.t)
                
                update = self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
                param.data = param.data + update.to(param.dtype)
        else:
            for name, param in self.model.named_parameters():
                update = self.lr * gradients[name]
                param.data = param.data + update.to(param.dtype)
    
    def train_step(self, images, targets, seed, fitness_method='zscore'):
        """Single training step"""
        batch_size = images.shape[0]
        
        # Forward with perturbations
        outputs = self.forward_batched(images, seed, batch_size)
        
        # Compute fitness
        mse = ((outputs - targets) ** 2).mean(dim=tuple(range(1, outputs.dim())))
        fitness = -mse
        
        if fitness_method == 'zscore':
            fitness = (fitness - fitness.mean()) / (fitness.std() + 1e-8)
        elif fitness_method == 'rank':
            N = fitness.shape[0]
            ranks = fitness.argsort().argsort().float()
            fitness = 2 * ranks / (N - 1) - 1
        
        # Compute and apply gradient
        gradients = self.compute_gradient_batched(fitness)
        self.apply_update(gradients)
        
        return {
            'outputs': outputs,
            'mse': mse,
            'fitnesses': fitness,
            'best_mse': mse.min().item(),
            'mean_mse': mse.mean().item(),
        }


class SimpleDataloader:
    def __init__(self, path):
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        if os.path.isdir(path):
            self.paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".png")]
        else:
            self.paths = [path]
    
    def get_image(self):
        img_path = random.choice(self.paths)
        return self.transform(Image.open(img_path))


if __name__ == "__main__":
    print("=" * 60)
    print("EGGROLL PyTorch Implementation")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Config
    batch_size = 256  # Population size
    num_epochs = 10000
    
    # Model
    model = EggrollModel(hidden_dim=32).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params}")
    
    # Trainer with Adam
    trainer = EggrollTrainerBatched(
        model, 
        lr=0.05,      # Learning rate
        sigma=0.6,    # Perturbation scale (σ in paper)
        rank=1,       # Rank-1 perturbations
        use_adam=False,
        beta1=0.9,
        beta2=0.999
    )
    
    # Ground truth - simple gradient pattern

    # Load image
    class SimpleDataloader:
        def __init__(self, path):
            self.transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
            self.paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".png")]
        
        def get_image(self):
            img_path = random.choice(self.paths)
            return self.transform(Image.open(img_path))

    dataset = SimpleDataloader("data\\MNIST\\dataset-part1")
    ground_truth_f = dataset.get_image().to("cuda").permute(1, 2, 0).contiguous()
    ground_truth = ground_truth_f.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [B, H, W, C]


    images = ground_truth.clone()
    
    print(f"\nConfig:")
    print(f"  Batch size (population): {batch_size}")
    print(f"  Learning rate: {trainer.lr}")
    print(f"  Sigma (perturbation): {trainer.sigma}")
    print(f"  Rank: {trainer.rank}")
    print(f"  Using Adam: {trainer.use_adam}")
    print(f"  Ground truth range: [{ground_truth.min():.2f}, {ground_truth.max():.2f}]")
    print()
    
    losses = []
    
    for epoch in range(num_epochs):
        seed = epoch * 1000
        
        # Add small noise to input
        noise_scale = 0.05
        noisy = images + noise_scale * (torch.rand_like(images) * 2 - 1)
        
        result = trainer.train_step(noisy, ground_truth, seed, fitness_method='zscore')
        
        losses.append(result['mean_mse'])
        
        if epoch % 16 == 0 or epoch == num_epochs - 1:
            # Eval without perturbation
            with torch.no_grad():
                eval_out = model(images[:1])
                eval_mse = ((eval_out - ground_truth[:1]) ** 2).mean().item()
            
            try:
                from py.torch_utils import dds_from_tensor
                dds = dds_from_tensor(eval_out[0:1].float().permute(0, 3, 1, 2) / 127.0)
                dds.save(".tmp/epoch.dds")
            except: pass

            print(f"Epoch {epoch}:")
            print(f"  Best MSE: {result['best_mse']:.6f}")
            print(f"  Mean MSE: {result['mean_mse']:.6f}")
            print(f"  Eval MSE: {eval_mse:.6f}")
            print(f"  Fitness std: {result['fitnesses'].std().item():.4f}")
            print()
    
    print("=" * 60)
    print(f"Loss progression: {losses[0]:.6f} -> {losses[-1]:.6f}")
    if losses[0] > 0:
        print(f"Improvement: {(losses[0] - losses[-1]) / losses[0] * 100:.1f}%")
    print("=" * 60)