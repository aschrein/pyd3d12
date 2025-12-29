#!/usr/bin/env python3
# MIT License
# Copyright (c) 2025 Anton Schreiner
"""
Toy Character-Level LLM - Expression Compiler

Learns to compile arithmetic expressions to register-based assembly.
Result is always in r0.

Training data format:
    # 5 + 3
    mov r1 5
    mov r2 3
    add r0 r1 r2
    
    # 53 * (1 + 4)
    mov r1 1
    mov r2 4
    add r3 r1 r2
    mov r1 53
    mul r0 r1 r3

Instruction Set:
    mov rX N      - Load immediate N into rX
    add rX rY rZ  - rX = rY + rZ
    sub rX rY rZ  - rX = rY - rZ
    mul rX rY rZ  - rX = rY * rZ
    div rX rY rZ  - rX = rY // rZ

RL Methods (rl.method=):
    rft           - Rejection Sampling FT (default, safest)
                    Only trains on correct samples via supervised learning
    bon           - Best-of-N: Group sampling + supervised learning on top-k
                    Similar to RFT but uses relative ranking within groups
    eggroll       - EGGROLL: Evolution Strategies with Rank-1 Perturbations
                    E = a ⊗ b^T (outer product), sum gives full-rank updates
                    Reference: arxiv.org/abs/2511.16652
    positive_only - Positive-Only GRPO (safe)
                    Only applies gradient updates when reward > 0
    grpo          - Standard GRPO (can degrade if not tuned)
                    Uses all samples with relative advantages

Usage:
    python toy_llm.py command=train
    python toy_llm.py command=rl                      # Default: EGGROLL
    python toy_llm.py command=rl rl.method=rft        # RFT (safest)
    python toy_llm.py command=rl rl.method=grpo       # Standard GRPO
    python toy_llm.py command=generate generate.prompt="# 7 * 8"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import math
import time
import random
import re
import copy
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from omegaconf import OmegaConf
import hydra
from hydra.core.config_store import ConfigStore


# ============================================================================
# Config
# ============================================================================

@dataclass
class ModelConfig:
    n_layers: int = 4
    n_heads: int = 4
    d_model: int = 256
    d_ff: int = 1024
    dropout: float = 0.1
    max_seq_len: int = 512
    vocab_size: int = 128


@dataclass
class TrainingConfig:
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_iters: int = 5000
    eval_interval: int = 500
    eval_iters: int = 50
    warmup_iters: int = 100
    min_lr: float = 1e-5
    grad_clip: float = 1.0


@dataclass
class DataConfig:
    num_samples: int = 100000
    num_registers: int = 8
    max_num: int = 50  # Keep numbers smaller for easier learning
    min_num: int = 1   # Avoid 0 to prevent div by zero issues
    operations: List[str] = field(default_factory=lambda: ['+', '-', '*'])
    min_program_len: int = 3
    max_program_len: int = 6
    seed: int = 42


@dataclass
class RLConfig:
    batch_size: int = 8
    group_size: int = 32
    max_iters: int = 1000
    learning_rate: float = 5e-6
    min_lr: float = 1e-6
    warmup_iters: int = 50
    grad_clip: float = 0.5
    eval_interval: int = 100
    kl_coef: float = 0.1
    kl_target: float = 0.5
    kl_max: float = 5.0
    adaptive_kl: bool = True
    entropy_coef: float = 0.001
    max_answer_tokens: int = 80
    temperature: float = 0.6
    save_interval: int = 100
    curriculum: bool = True
    method: str = "rft"  # "grpo", "positive_only", "rft", "eggroll", "bon"
    # EGGROLL specific (Evolution Strategies with rank-1 perturbations)
    # Batched forward - all N evaluations run in parallel!
    es_sigma: float = 0.02  # Perturbation scale
    es_population: int = 20  # Pop size (batched, all N evals in parallel)


@dataclass
class GenerateConfig:
    prompt: Optional[str] = None
    max_tokens: int = 100
    temperature: float = 0.8


@dataclass
class CheckpointConfig:
    path: str = ".tmp/toy_llm_vm.pt"
    rl_path: str = ".tmp/toy_llm_vm.pt"
    fresh: bool = False


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    generate: GenerateConfig = field(default_factory=GenerateConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    command: str = "train"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


cs = ConfigStore.instance()
cs.store(name="config", node=Config)


# ============================================================================
# Tokenizer
# ============================================================================

class ASCIITokenizer:
    vocab_size = 128
    
    def encode(self, text: str) -> List[int]:
        return [min(ord(c), 127) for c in text]
    
    def decode(self, tokens: List[int]) -> str:
        return ''.join(chr(t) for t in tokens)


# ============================================================================
# Virtual Machine
# ============================================================================

class VM:
    """
    Simple register-based virtual machine.
    
    Instructions:
        mov rX N      - Load immediate N into rX
        add rX rY rZ  - rX = rY + rZ
        sub rX rY rZ  - rX = rY - rZ
        mul rX rY rZ  - rX = rY * rZ
        div rX rY rZ  - rX = rY // rZ
    """
    
    OPS = {'add', 'sub', 'mul', 'div'}
    
    def __init__(self, num_regs: int = 8):
        self.num_regs = num_regs
        self.regs = {f"r{i}": 0 for i in range(num_regs)}
    
    def reset(self):
        """Reset all registers to 0."""
        for r in self.regs:
            self.regs[r] = 0
    
    def run(self, program: str) -> dict:
        """Execute program and return register state."""
        self.reset()
        
        for line in program.split('\n'):
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if not parts:
                continue
            
            op = parts[0]
            
            if op == 'mov' and len(parts) == 3:
                dst, val = parts[1], parts[2]
                if dst in self.regs:
                    try:
                        self.regs[dst] = int(val)
                    except ValueError:
                        pass
            
            elif op in self.OPS and len(parts) == 4:
                dst, s1, s2 = parts[1], parts[2], parts[3]
                if all(r in self.regs for r in [dst, s1, s2]):
                    a, b = self.regs[s1], self.regs[s2]
                    if op == 'add':
                        self.regs[dst] = a + b
                    elif op == 'sub':
                        self.regs[dst] = a - b
                    elif op == 'mul':
                        self.regs[dst] = a * b
                    elif op == 'div' and b != 0:
                        self.regs[dst] = a // b
        
        return dict(self.regs)
    
    def get(self, reg: str) -> int:
        """Get register value."""
        return self.regs.get(reg, 0)


# ============================================================================
# Expression Compiler
# ============================================================================

class ExprCompiler:
    """
    Compiles arithmetic expressions to assembly.
    Result always ends up in r0.
    
    Example:
        # 5 + 3
        mov r1 5
        mov r2 3
        add r0 r1 r2
        
        # 53 * (1 + 4)
        mov r1 1
        mov r2 4
        add r3 r1 r2
        mov r1 53
        mul r0 r1 r3
    """
    
    OP_MAP = {'+': 'add', '-': 'sub', '*': 'mul', '/': 'div'}
    
    def __init__(self, num_regs: int = 8):
        self.num_regs = num_regs
    
    def compile(self, expr: str) -> Tuple[str, int]:
        """
        Compile expression to assembly.
        Result is always in r0.
        
        Returns:
            asm: Assembly code
            result: Computed result
        """
        self.next_reg = 1  # Start from r1, r0 is for final result
        self.lines = []
        
        # Parse and compile
        result_reg, result_val = self._compile_expr(expr.strip())
        
        # Move result to r0 if not already there
        if result_reg != 'r0':
            # Use add r0 result_reg r1 where r1=0, or just track that result is in result_reg
            # Simpler: just ensure final op writes to r0
            pass
        
        asm = '\n'.join(self.lines)
        return asm, result_val
    
    def _alloc_reg(self) -> str:
        """Allocate next register (r1, r2, ... wraps around but skips r0)."""
        reg = f"r{self.next_reg}"
        self.next_reg += 1
        if self.next_reg >= self.num_regs:
            self.next_reg = 1  # Wrap but skip r0
        return reg
    
    def _compile_expr(self, expr: str, target_reg: str = None) -> Tuple[str, int]:
        """
        Recursively compile expression.
        If target_reg is specified, result goes there.
        Returns (register, value).
        """
        expr = expr.strip()
        
        # Remove outer parens if they wrap the whole expression
        while expr.startswith('(') and expr.endswith(')') and self._matching_parens(expr):
            expr = expr[1:-1].strip()
        
        # Try to parse as number
        try:
            val = int(expr)
            reg = target_reg if target_reg else self._alloc_reg()
            self.lines.append(f"mov {reg} {val}")
            return reg, val
        except ValueError:
            pass
        
        # Find the main operator (lowest precedence, rightmost)
        op_idx, op = self._find_main_op(expr)
        
        if op_idx == -1:
            raise ValueError(f"Cannot parse: {expr}")
        
        # Split and compile left/right
        left = expr[:op_idx].strip()
        right = expr[op_idx + 1:].strip()
        
        left_reg, left_val = self._compile_expr(left)
        right_reg, right_val = self._compile_expr(right)
        
        # Compute result
        if op == '+':
            result_val = left_val + right_val
        elif op == '-':
            result_val = left_val - right_val
        elif op == '*':
            result_val = left_val * right_val
        elif op == '/' and right_val != 0:
            result_val = left_val // right_val
        else:
            result_val = 0
        
        # Generate instruction - use target_reg if specified
        result_reg = target_reg if target_reg else self._alloc_reg()
        self.lines.append(f"{self.OP_MAP[op]} {result_reg} {left_reg} {right_reg}")
        
        return result_reg, result_val
    
    def _matching_parens(self, expr: str) -> bool:
        """Check if outer parens match each other."""
        if not expr.startswith('('):
            return False
        depth = 0
        for i, c in enumerate(expr):
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
            if depth == 0:
                return i == len(expr) - 1
        return False
    
    def _find_main_op(self, expr: str) -> Tuple[int, str]:
        """Find main operator (lowest precedence, rightmost at same level)."""
        depth = 0
        # First pass: look for + or - (lowest precedence)
        for i in range(len(expr) - 1, -1, -1):
            c = expr[i]
            if c == ')':
                depth += 1
            elif c == '(':
                depth -= 1
            elif depth == 0 and c in '+-':
                # Check it's not a unary minus at start
                if i > 0:
                    return i, c
        
        # Second pass: look for * or /
        depth = 0
        for i in range(len(expr) - 1, -1, -1):
            c = expr[i]
            if c == ')':
                depth += 1
            elif c == '(':
                depth -= 1
            elif depth == 0 and c in '*/':
                return i, c
        
        return -1, ''
    
    def compile_to_r0(self, expr: str) -> Tuple[str, int]:
        """Compile expression with result in r0."""
        self.next_reg = 1
        self.lines = []
        
        _, result_val = self._compile_expr(expr.strip(), target_reg='r0')
        
        asm = '\n'.join(self.lines)
        return asm, result_val


# ============================================================================
# Program Generator
# ============================================================================

class ProgramGenerator:
    """Generate training data: expression comment + assembly."""
    
    def __init__(self, cfg: DataConfig):
        self.cfg = cfg
        self.compiler = ExprCompiler(cfg.num_registers)
        self.vm = VM(cfg.num_registers)
        self.rng = random.Random(cfg.seed)
    
    def _random_expr(self, depth: int = 0, max_depth: int = 3) -> str:
        """Generate random expression with varied complexity."""
        # Base case: just a number
        if depth >= max_depth or (depth > 0 and self.rng.random() < 0.3):
            return str(self.rng.randint(self.cfg.min_num, self.cfg.max_num))
        
        # Choose expression type
        expr_type = self.rng.random()
        
        if expr_type < 0.5:
            # Binary expression: left op right
            left = self._random_expr(depth + 1, max_depth)
            right = self._random_expr(depth + 1, max_depth)
            op = self.rng.choice(list(self.cfg.operations))
            
            if op == '/' and right == '0':
                right = str(self.rng.randint(1, self.cfg.max_num))
            
            expr = f"{left} {op} {right}"
            
            # Wrap in parens sometimes (more likely for + - to show precedence)
            if depth > 0 and op in ['+', '-'] and self.rng.random() < 0.6:
                expr = f"({expr})"
            elif depth > 0 and self.rng.random() < 0.3:
                expr = f"({expr})"
                
        elif expr_type < 0.7:
            # Chain: a op b op c (same precedence ops)
            chain_len = self.rng.randint(2, 4)
            chain_ops = ['+', '-'] if self.rng.random() < 0.7 else ['*']
            
            parts = [str(self.rng.randint(self.cfg.min_num, self.cfg.max_num))]
            for _ in range(chain_len):
                op = self.rng.choice(chain_ops)
                num = str(self.rng.randint(self.cfg.min_num, self.cfg.max_num))
                parts.append(f" {op} {num}")
            
            expr = ''.join(parts)
            if depth > 0 and self.rng.random() < 0.5:
                expr = f"({expr})"
                
        elif expr_type < 0.85:
            # Mixed precedence: a + b * c or a * b + c
            a = str(self.rng.randint(self.cfg.min_num, self.cfg.max_num))
            b = str(self.rng.randint(self.cfg.min_num, self.cfg.max_num))
            c = str(self.rng.randint(self.cfg.min_num, self.cfg.max_num))
            
            if self.rng.random() < 0.5:
                # a + b * c
                op1 = self.rng.choice(['+', '-'])
                op2 = self.rng.choice(['*'])
                expr = f"{a} {op1} {b} {op2} {c}"
            else:
                # a * b + c
                op1 = self.rng.choice(['*'])
                op2 = self.rng.choice(['+', '-'])
                expr = f"{a} {op1} {b} {op2} {c}"
                
        else:
            # Nested parens: (a op b) op (c op d)
            a = self._random_expr(depth + 1, max_depth)
            b = self._random_expr(depth + 1, max_depth)
            c = self._random_expr(depth + 1, max_depth)
            d = self._random_expr(depth + 1, max_depth)
            
            op1 = self.rng.choice(list(self.cfg.operations))
            op2 = self.rng.choice(list(self.cfg.operations))
            op3 = self.rng.choice(list(self.cfg.operations))
            
            if op1 == '/' and b == '0':
                b = str(self.rng.randint(1, self.cfg.max_num))
            if op3 == '/' and d == '0':
                d = str(self.rng.randint(1, self.cfg.max_num))
            
            left = f"({a} {op1} {b})"
            right = f"({c} {op3} {d})"
            expr = f"{left} {op2} {right}"
        
        return expr
    
    def generate_one(self) -> Tuple[str, str, int]:
        """
        Generate one example with random difficulty.
        
        Returns:
            full: "# expr\\nasm" 
            expr: The expression
            result: Expected value in r0
        """
        # Random difficulty for each sample
        max_depth = self.rng.randint(1, 4)
        
        expr = self._random_expr(max_depth=max_depth)
        
        try:
            asm, result = self.compiler.compile_to_r0(expr)
            
            # Verify with VM
            state = self.vm.run(asm)
            assert state['r0'] == result, f"VM mismatch: {state['r0']} != {result}"
            
            # Skip if result is too large (overflow-ish)
            if abs(result) > 100000:
                raise ValueError("Result too large")
            
            full = f"# {expr}\n{asm}"
            return full, expr, result
        except Exception as e:
            # Fallback to simple expression
            a = self.rng.randint(1, self.cfg.max_num)
            b = self.rng.randint(1, self.cfg.max_num)
            op = self.rng.choice(['+', '-', '*'])
            expr = f"{a} {op} {b}"
            asm, result = self.compiler.compile_to_r0(expr)
            full = f"# {expr}\n{asm}"
            return full, expr, result
    
    def generate_dataset(self, n: int) -> str:
        """Generate n programs."""
        programs = []
        
        for i in range(n):
            full, expr, result = self.generate_one()
            programs.append(full)
            
            if i % 20000 == 0 and i > 0:
                print(f"  Generated {i:,}...")
        
        print(f"Generated {n:,} programs")
        print("\nExamples:")
        for p in programs[:8]:
            print("-" * 40)
            print(p)
            # Verify
            state = self.vm.run(p)
            print(f"  r0 = {state['r0']}")
        
        return '\n\n'.join(programs)


# ============================================================================
# Dataset
# ============================================================================

class TextDataset(Dataset):
    def __init__(self, data: torch.Tensor, seq_len: int):
        self.data = data
        self.seq_len = seq_len
    
    def __len__(self):
        return max(0, len(self.data) - self.seq_len - 1)
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + 1:idx + self.seq_len + 1]
        return x, y


# ============================================================================
# Model
# ============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_len: int = 2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self._build_cache(max_len)
    
    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos', emb.cos(), persistent=False)
        self.register_buffer('sin', emb.sin(), persistent=False)
    
    def forward(self, seq_len: int):
        if seq_len > self.cos.size(0):
            self._build_cache(seq_len)
        return self.cos[:seq_len], self.sin[:seq_len]


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        self.rope = RotaryEmbedding(self.head_dim, cfg.max_seq_len)
    
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2) for t in qkv]
        
        cos, sin = self.rope(T)
        cos, sin = cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)
        q, k = apply_rope(q, k, cos, sin)
        
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                              dropout_p=self.dropout.p if self.training else 0.0)
        return self.dropout(self.out(out.transpose(1, 2).contiguous().view(B, T, C)))


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.w3 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
    
    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class ToyLLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.tok_emb.weight = self.head.weight
        self.apply(self._init_weights)
        print(f"Parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
    
    def forward(self, idx, targets=None):
        x = self.dropout(self.tok_emb(idx))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_tokens: int, temperature: float = 0.8, stop_at_double_newline: bool = True):
        self.eval()
        B = idx.size(0)
        device = idx.device
        
        # Track double newline for each sequence in batch
        last_was_newline = torch.zeros(B, dtype=torch.bool, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        
        for _ in range(max_tokens):
            idx_cond = idx if idx.size(1) <= self.cfg.max_seq_len else idx[:, -self.cfg.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_tok], dim=1)
            
            # Stop at double newline
            if stop_at_double_newline:
                is_newline = (next_tok.squeeze(-1) == ord('\n'))
                finished = finished | (is_newline & last_was_newline)
                last_was_newline = is_newline
                
                if finished.all():
                    break
        return idx
    
    @torch.no_grad()
    def generate_with_log_probs(self, idx, max_tokens: int, temperature: float = 1.0,
                                 stop_token: int = None):
        """Generate with log probs for RL. Stops at double newline."""
        self.eval()
        B, prompt_len = idx.size(0), idx.size(1)
        device = idx.device
        
        all_lp, all_mask, all_tok = [], [], []
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        last_was_newline = torch.zeros(B, dtype=torch.bool, device=device)
        
        for _ in range(max_tokens):
            idx_cond = idx if idx.size(1) <= self.cfg.max_seq_len else idx[:, -self.cfg.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1)
            
            tok_lp = log_probs.gather(-1, next_tok).squeeze(-1)
            mask = ~finished
            
            all_lp.append(tok_lp * mask.float())
            all_mask.append(mask.float())
            all_tok.append(next_tok.squeeze(-1))
            
            idx = torch.cat([idx, next_tok], dim=1)
            
            # Stop at double newline (empty line = end of assembly)
            is_newline = (next_tok.squeeze(-1) == ord('\n'))
            finished = finished | (is_newline & last_was_newline)
            last_was_newline = is_newline
            
            if finished.all():
                break
        
        # Pad
        gen_len = len(all_lp)
        for _ in range(max_tokens - gen_len):
            all_lp.append(torch.zeros(B, device=device))
            all_mask.append(torch.zeros(B, device=device))
            all_tok.append(torch.zeros(B, dtype=torch.long, device=device))
        
        log_probs = torch.stack(all_lp, dim=1)
        masks = torch.stack(all_mask, dim=1)
        gen_tokens = torch.stack(all_tok, dim=1)
        seqs = torch.cat([idx[:, :prompt_len], gen_tokens], dim=1)
        
        return seqs, log_probs, masks, prompt_len


class GRPO:
    """Standard GRPO (Group Relative Policy Optimization)."""
    
    def __init__(self, model: ToyLLM, tokenizer: ASCIITokenizer, cfg: RLConfig, data_cfg: DataConfig, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.device = device
        
        self.ref_model = copy.deepcopy(model)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False
        
        self.kl_coef = cfg.kl_coef
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=0.01)
        self.problem_gen = RLProblemGen(data_cfg)
        self.vm = VM(data_cfg.num_registers)
        
        self.best_score = -float('inf')
        self.best_state = copy.deepcopy(model.state_dict())
    
    def get_lr(self, it: int) -> float:
        if it < self.cfg.warmup_iters:
            return self.cfg.learning_rate * (it + 1) / self.cfg.warmup_iters
        ratio = (it - self.cfg.warmup_iters) / max(1, self.cfg.max_iters - self.cfg.warmup_iters)
        return self.cfg.min_lr + 0.5 * (1 + math.cos(math.pi * min(1, ratio))) * (self.cfg.learning_rate - self.cfg.min_lr)
    
    def train_step(self, iteration: int) -> Tuple[dict, bool]:
        lr = self.get_lr(iteration)
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        
        if self.cfg.curriculum:
            difficulty = min(1.0, iteration / (self.cfg.max_iters * 0.9))
        else:
            difficulty = 1.0
        self.problem_gen.set_difficulty(difficulty)
        
        problems = self.problem_gen.generate_batch(self.cfg.batch_size)
        prompts = [p[0] for p in problems]
        expected_asms = [p[1] for p in problems]
        
        B, G = self.cfg.batch_size, self.cfg.group_size
        exp_prompts = [p for p in prompts for _ in range(G)]
        exp_asms = [a for a in expected_asms for _ in range(G)]
        
        toks = [self.tokenizer.encode(p) for p in exp_prompts]
        max_len = max(len(t) for t in toks)
        padded = [[ord(' ')] * (max_len - len(t)) + t for t in toks]
        prompt_tensor = torch.tensor(padded, dtype=torch.long, device=self.device)
        
        seqs, old_lp, masks, prompt_len = self.model.generate_with_log_probs(
            prompt_tensor, self.cfg.max_answer_tokens, self.cfg.temperature
        )
        
        rewards, infos = [], []
        for i in range(B * G):
            text = self.tokenizer.decode(seqs[i].tolist())
            r, info = compute_reward(text, exp_asms[i], exp_prompts[i], self.vm)
            rewards.append(r)
            infos.append(info)
        rewards = torch.tensor(rewards, device=self.device)
        rewards = torch.clamp(rewards, -1.0, 1.0)
        
        rewards_g = rewards.view(B, G)
        mean_r = rewards_g.mean(1, keepdim=True)
        std_r = rewards_g.std(1, keepdim=True)
        std_r = torch.where(std_r < 1e-4, torch.ones_like(std_r), std_r)
        adv = ((rewards_g - mean_r) / std_r).view(B * G)
        adv = torch.clamp(adv, -2.0, 2.0)
        
        self.model.eval()
        with torch.enable_grad():
            logits, _ = self.model(seqs)
            gen_len = old_lp.size(1)
            pred_logits = logits[:, prompt_len - 1:prompt_len - 1 + gen_len, :]
            gen_tokens = seqs[:, prompt_len:prompt_len + gen_len]
            cur_lp = F.log_softmax(pred_logits, dim=-1).gather(-1, gen_tokens.unsqueeze(-1)).squeeze(-1)
            
            with torch.no_grad():
                ref_logits, _ = self.ref_model(seqs)
                ref_lp = F.log_softmax(ref_logits[:, prompt_len - 1:prompt_len - 1 + gen_len, :], dim=-1)
                ref_lp = ref_lp.gather(-1, gen_tokens.unsqueeze(-1)).squeeze(-1)
            
            policy_loss = -(adv.unsqueeze(1) * cur_lp)
            kl = cur_lp - ref_lp
            total = (policy_loss + self.kl_coef * kl - self.cfg.entropy_coef * (-cur_lp)) * masks
            loss = total.sum() / (masks.sum() + 1e-8)
            
            self.optimizer.zero_grad()
            loss.backward()
        
        if self.cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        self.model.train()
        
        mean_kl = (kl * masks).sum().item() / (masks.sum().item() + 1e-8)
        if self.cfg.adaptive_kl:
            if mean_kl > self.cfg.kl_target * 1.5:
                self.kl_coef = min(self.kl_coef * 1.5, 1.0)
            elif mean_kl < self.cfg.kl_target * 0.5:
                self.kl_coef = max(self.kl_coef * 0.8, 0.001)
        
        acc = sum(1 for i in infos if i['correct']) / len(infos)
        clean = sum(1 for i in infos if i['correct'] and i['garbage'] == 0) / len(infos)
        if acc > self.best_score:
            self.best_score = acc
            self.best_state = copy.deepcopy(self.model.state_dict())
        
        return {'loss': loss.item(), 'kl': mean_kl, 'kl_coef': self.kl_coef,
                'reward': rewards.mean().item(), 'accuracy': acc, 'clean': clean, 
                'difficulty': difficulty, 'lr': lr}, mean_kl > self.cfg.kl_max
    
    def restore_best(self):
        self.model.load_state_dict(self.best_state)


# ============================================================================
# Safe RL Methods
# ============================================================================

class RejectionSamplingFT:
    """
    Rejection Sampling Fine-Tuning (RFT).
    
    1. Generate samples from current model
    2. Filter to keep only correct ones
    3. Fine-tune with supervised learning on correct samples
    
    This CANNOT degrade the model because we only learn from successes.
    """
    
    def __init__(self, model: ToyLLM, tokenizer: ASCIITokenizer, cfg: RLConfig, 
                 data_cfg: DataConfig, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.device = device
        
        self.problem_gen = RLProblemGen(data_cfg)
        self.vm = VM(data_cfg.num_registers)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=0.01)
        
        self.best_acc = 0.0
        self.best_state = copy.deepcopy(model.state_dict())
        self.difficulty = 0.0
    
    def get_lr(self, it: int) -> float:
        if it < self.cfg.warmup_iters:
            return self.cfg.learning_rate * (it + 1) / self.cfg.warmup_iters
        ratio = (it - self.cfg.warmup_iters) / max(1, self.cfg.max_iters - self.cfg.warmup_iters)
        return self.cfg.min_lr + 0.5 * (1 + math.cos(math.pi * min(1, ratio))) * (self.cfg.learning_rate - self.cfg.min_lr)
    
    def train_step(self, iteration: int) -> Tuple[dict, bool]:
        lr = self.get_lr(iteration)
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        
        # Curriculum
        if self.cfg.curriculum:
            self.difficulty = min(1.0, iteration / (self.cfg.max_iters * 0.9))
        else:
            self.difficulty = 1.0
        self.problem_gen.set_difficulty(self.difficulty)
        
        # Generate more samples to find enough correct ones
        num_attempts = self.cfg.batch_size * 4
        problems = self.problem_gen.generate_batch(num_attempts)
        
        # Generate completions
        correct_pairs = []  # (prompt_tokens, completion_tokens)
        
        self.model.eval()
        for prompt, expected_asm, info in problems:
            prompt_toks = self.tokenizer.encode(prompt)
            idx = torch.tensor([prompt_toks], dtype=torch.long, device=self.device)
            
            with torch.no_grad():
                out = self.model.generate(idx, self.cfg.max_answer_tokens, temperature=self.cfg.temperature)
            
            gen_text = self.tokenizer.decode(out[0].tolist())
            gen_asm = gen_text[len(prompt):]
            
            # Check if correct
            try:
                state = self.vm.run(gen_asm)
                if state['r0'] == info['result']:
                    # Success! Save for training
                    full_text = gen_text.split('\n\n')[0]  # Up to first empty line
                    full_toks = self.tokenizer.encode(full_text)
                    correct_pairs.append((prompt_toks, full_toks))
            except:
                pass
            
            if len(correct_pairs) >= self.cfg.batch_size:
                break
        
        metrics = {
            'correct_found': len(correct_pairs),
            'attempts': min(len(problems), num_attempts),
            'difficulty': self.difficulty,
            'lr': lr
        }
        
        if len(correct_pairs) < 4:
            # Not enough correct samples, skip update
            metrics['loss'] = 0.0
            metrics['accuracy'] = len(correct_pairs) / num_attempts
            return metrics, False
        
        # Supervised learning on correct samples
        self.model.train()
        
        # Pad sequences
        max_len = max(len(t) for _, t in correct_pairs)
        padded = []
        for _, toks in correct_pairs:
            padded.append([0] * (max_len - len(toks)) + toks)
        
        batch = torch.tensor(padded, dtype=torch.long, device=self.device)
        x, y = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
        
        _, loss = self.model(x, y)
        
        self.optimizer.zero_grad()
        loss.backward()
        if self.cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        
        metrics['loss'] = loss.item()
        metrics['accuracy'] = len(correct_pairs) / num_attempts
        
        if metrics['accuracy'] > self.best_acc:
            self.best_acc = metrics['accuracy']
            self.best_state = copy.deepcopy(self.model.state_dict())
        
        return metrics, False
    
    def restore_best(self):
        self.model.load_state_dict(self.best_state)


class PositiveOnlyGRPO(GRPO):
    """
    GRPO but only updates on positive rewards.
    Skips gradient updates for failed samples.
    """
    
    def train_step(self, iteration: int) -> Tuple[dict, bool]:
        lr = self.get_lr(iteration)
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        
        if self.cfg.curriculum:
            difficulty = min(1.0, iteration / (self.cfg.max_iters * 0.9))
        else:
            difficulty = 1.0
        self.problem_gen.set_difficulty(difficulty)
        
        problems = self.problem_gen.generate_batch(self.cfg.batch_size)
        prompts = [p[0] for p in problems]
        expected_asms = [p[1] for p in problems]
        
        B, G = self.cfg.batch_size, self.cfg.group_size
        exp_prompts = [p for p in prompts for _ in range(G)]
        exp_asms = [a for a in expected_asms for _ in range(G)]
        
        toks = [self.tokenizer.encode(p) for p in exp_prompts]
        max_len = max(len(t) for t in toks)
        padded = [[ord(' ')] * (max_len - len(t)) + t for t in toks]
        prompt_tensor = torch.tensor(padded, dtype=torch.long, device=self.device)
        
        seqs, old_lp, masks, prompt_len = self.model.generate_with_log_probs(
            prompt_tensor, self.cfg.max_answer_tokens, self.cfg.temperature
        )
        
        rewards, infos = [], []
        for i in range(B * G):
            text = self.tokenizer.decode(seqs[i].tolist())
            r, info = compute_reward(text, exp_asms[i], exp_prompts[i], self.vm)
            rewards.append(r)
            infos.append(info)
        rewards = torch.tensor(rewards, device=self.device)
        
        # KEY DIFFERENCE: Only positive rewards contribute
        positive_mask = (rewards > 0).float()
        
        if positive_mask.sum() < 1:
            # No positive samples, skip update entirely
            return {
                'loss': 0.0, 'kl': 0.0, 'kl_coef': self.kl_coef,
                'reward': rewards.mean().item(), 
                'accuracy': sum(1 for i in infos if i['correct']) / len(infos),
                'clean': 0.0, 'difficulty': difficulty, 'lr': lr,
                'positive_samples': 0
            }, False
        
        # Advantages only for positive samples
        # Use reward directly as advantage (no relative normalization needed)
        adv = rewards * positive_mask
        
        # Current log probs
        self.model.eval()
        with torch.enable_grad():
            logits, _ = self.model(seqs)
            gen_len = old_lp.size(1)
            pred_logits = logits[:, prompt_len - 1:prompt_len - 1 + gen_len, :]
            gen_tokens = seqs[:, prompt_len:prompt_len + gen_len]
            cur_lp = F.log_softmax(pred_logits, dim=-1).gather(-1, gen_tokens.unsqueeze(-1)).squeeze(-1)
            
            with torch.no_grad():
                ref_logits, _ = self.ref_model(seqs)
                ref_lp = F.log_softmax(ref_logits[:, prompt_len - 1:prompt_len - 1 + gen_len, :], dim=-1)
                ref_lp = ref_lp.gather(-1, gen_tokens.unsqueeze(-1)).squeeze(-1)
            
            # Loss only on positive samples
            policy_loss = -(adv.unsqueeze(1) * cur_lp)
            kl = cur_lp - ref_lp
            
            # Mask: only positive samples AND valid tokens
            combined_mask = masks * positive_mask.unsqueeze(1)
            
            total = (policy_loss + self.kl_coef * kl) * combined_mask
            loss = total.sum() / (combined_mask.sum() + 1e-8)
            
            self.optimizer.zero_grad()
            loss.backward()
        
        if self.cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        self.model.train()
        
        mean_kl = (kl * combined_mask).sum().item() / (combined_mask.sum().item() + 1e-8)
        
        acc = sum(1 for i in infos if i['correct']) / len(infos)
        clean = sum(1 for i in infos if i['correct'] and i['garbage'] == 0) / len(infos)
        
        if acc > self.best_score:
            self.best_score = acc
            self.best_state = copy.deepcopy(self.model.state_dict())
        
        return {
            'loss': loss.item(), 'kl': mean_kl, 'kl_coef': self.kl_coef,
            'reward': rewards.mean().item(), 'accuracy': acc, 'clean': clean,
            'difficulty': difficulty, 'lr': lr,
            'positive_samples': int(positive_mask.sum().item())
        }, mean_kl > self.cfg.kl_max


class EGGROLL:
    """
    EGGROLL: Evolution Guided General Optimization via Low-rank Learning
    
    Reference: "Evolution Strategies at the Hyperscale" (2024)
    https://arxiv.org/abs/2511.16652
    
    Efficient batched forward (no weight modification needed!):
        y = x @ W.T + σ * sign * (x @ b) * a
        
    where (x @ b) is a dot product giving a scalar, then scaled by a.
    All N population members computed in parallel!
    """
    
    def __init__(self, model: ToyLLM, tokenizer: ASCIITokenizer, cfg: RLConfig,
                 data_cfg: DataConfig, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.device = device
        
        self.problem_gen = RLProblemGen(data_cfg)
        self.vm = VM(data_cfg.num_registers)
        
        self.sigma = cfg.es_sigma
        self.population_size = cfg.es_population
        
        self.best_acc = 0.0
        self.best_state = copy.deepcopy(model.state_dict())
        self.difficulty = 0.0
        
        # Build list of weights in the EXACT ORDER we access them in _batched_forward
        self._weights = []  # List of (weight_tensor, d_out, d_in)
        for block in model.blocks:
            self._weights.append((block.attn.qkv.weight, block.attn.qkv.weight.shape[0], block.attn.qkv.weight.shape[1]))
            self._weights.append((block.attn.out.weight, block.attn.out.weight.shape[0], block.attn.out.weight.shape[1]))
            self._weights.append((block.mlp.w1.weight, block.mlp.w1.weight.shape[0], block.mlp.w1.weight.shape[1]))
            self._weights.append((block.mlp.w3.weight, block.mlp.w3.weight.shape[0], block.mlp.w3.weight.shape[1]))
            self._weights.append((block.mlp.w2.weight, block.mlp.w2.weight.shape[0], block.mlp.w2.weight.shape[1]))
        self._weights.append((model.head.weight, model.head.weight.shape[0], model.head.weight.shape[1]))
        
        print(f"  EGGROLL tracking {len(self._weights)} weight matrices")
    
    def get_lr(self, it: int) -> float:
        if it < self.cfg.warmup_iters:
            return self.cfg.learning_rate * (it + 1) / self.cfg.warmup_iters
        ratio = (it - self.cfg.warmup_iters) / max(1, self.cfg.max_iters - self.cfg.warmup_iters)
        return self.cfg.min_lr + 0.5 * (1 + math.cos(math.pi * min(1, ratio))) * (self.cfg.learning_rate - self.cfg.min_lr)
    
    def _sample_perturbations(self, N: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Sample rank-1 perturbations for all weights in forward order."""
        a_list, b_list = [], []
        for w, d_out, d_in in self._weights:
            a = torch.randn(N, d_out, device=self.device)
            b = torch.randn(N, d_in, device=self.device)
            a_list.append(a)
            b_list.append(b)
        return a_list, b_list
    
    def _perturbed_linear(self, x: torch.Tensor, weight: torch.Tensor, 
                          a: torch.Tensor, b: torch.Tensor, 
                          signs: torch.Tensor, bias=None) -> torch.Tensor:
        """
        Batched linear with rank-1 perturbation.
        
        Args:
            x: (N, ..., d_in) input
            weight: (d_out, d_in) base weight
            a: (N, d_out) perturbation vector
            b: (N, d_in) perturbation vector
            signs: (N,) +1 or -1 for antithetic
            bias: optional (d_out,)
        
        Returns:
            y: (N, ..., d_out)
        """
        # Base: x @ W.T + bias
        y = F.linear(x, weight, bias)  # (N, ..., d_out)
        
        # Perturbation: σ * sign * (x @ b) * a
        # x @ b: (N, ..., d_in) @ (N, d_in) -> need einsum
        # For x shape (N, B, T, d_in), b shape (N, d_in):
        # Result should be (N, B, T) scalar per position
        
        orig_shape = x.shape[:-1]  # (N, ...) without d_in
        N = x.shape[0]
        d_in = x.shape[-1]
        
        # Flatten middle dims: (N, -1, d_in)
        x_flat = x.reshape(N, -1, d_in)
        
        # (N, L, d_in) @ (N, d_in, 1) -> (N, L, 1)
        xb = torch.bmm(x_flat, b.unsqueeze(-1)).squeeze(-1)  # (N, L)
        
        # Scale by sigma * sign: (N, L) * (N, 1)
        xb_scaled = self.sigma * signs.view(N, 1) * xb  # (N, L)
        
        # Outer with a: (N, L, 1) * (N, 1, d_out) -> (N, L, d_out)
        perturbation = xb_scaled.unsqueeze(-1) * a.unsqueeze(1)  # (N, L, d_out)
        
        # Reshape back and add
        perturbation = perturbation.view(*orig_shape, -1)
        return y + perturbation
    
    def _batched_forward(self, idx: torch.Tensor, a_list: List[torch.Tensor],
                         b_list: List[torch.Tensor], signs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for all N population members in parallel.
        
        Args:
            idx: (N, B, T) token indices
            a_list, b_list: perturbation vectors per layer (in forward order)
            signs: (N,) +1 or -1
            
        Returns:
            logits: (N, B, T, vocab_size)
        """
        N, B, seq_len = idx.shape
        cfg = self.model.cfg
        
        # Token embedding: (N, B, T, d_model)
        x = self.model.tok_emb(idx)
        x = self.model.dropout(x)
        
        layer_idx = 0
        
        # Forward through blocks
        for block in self.model.blocks:
            # Attention
            norm_x = block.norm1(x)
            
            # QKV with perturbation (layer_idx = 0, 5, 10, ...)
            qkv = self._perturbed_linear(norm_x, block.attn.qkv.weight, 
                                         a_list[layer_idx], b_list[layer_idx], signs)
            layer_idx += 1
            q, k, v = qkv.chunk(3, dim=-1)
            
            # Reshape for attention
            head_dim = cfg.d_model // cfg.n_heads
            T = x.shape[2]
            q = q.view(N * B, T, cfg.n_heads, head_dim).transpose(1, 2)
            k = k.view(N * B, T, cfg.n_heads, head_dim).transpose(1, 2)
            v = v.view(N * B, T, cfg.n_heads, head_dim).transpose(1, 2)
            
            # RoPE
            cos, sin = block.attn.rope(T)
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
            q, k = apply_rope(q, k, cos, sin)
            
            # Attention
            attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            attn_out = attn_out.transpose(1, 2).contiguous().view(N, B, T, cfg.d_model)
            
            # Output projection (layer_idx = 1, 6, 11, ...)
            attn_out = self._perturbed_linear(attn_out, block.attn.out.weight,
                                              a_list[layer_idx], b_list[layer_idx], signs)
            layer_idx += 1
            x = x + attn_out
            
            # MLP
            norm_x = block.norm2(x)
            
            # w1 (layer_idx = 2, 7, 12, ...)
            gate = self._perturbed_linear(norm_x, block.mlp.w1.weight,
                                          a_list[layer_idx], b_list[layer_idx], signs)
            layer_idx += 1
            
            # w3 (layer_idx = 3, 8, 13, ...)
            up = self._perturbed_linear(norm_x, block.mlp.w3.weight,
                                        a_list[layer_idx], b_list[layer_idx], signs)
            layer_idx += 1
            
            h = F.silu(gate) * up
            
            # w2 (layer_idx = 4, 9, 14, ...)
            mlp_out = self._perturbed_linear(h, block.mlp.w2.weight,
                                             a_list[layer_idx], b_list[layer_idx], signs)
            layer_idx += 1
            x = x + mlp_out
        
        # Final norm and head
        x = self.model.norm(x)
        logits = self._perturbed_linear(x, self.model.head.weight,
                                        a_list[layer_idx], b_list[layer_idx], signs)
        
        return logits
    
    def _batched_generate(self, idx: torch.Tensor, a_list: List[torch.Tensor],
                          b_list: List[torch.Tensor], signs: torch.Tensor,
                          max_tokens: int, temperature: float) -> torch.Tensor:
        """
        Generate tokens for all N population members in parallel.
        
        Args:
            idx: (B, T) initial tokens (same prompt for all pop members)
            
        Returns:
            output: (N, B, T + gen_len) full sequences
        """
        N = signs.shape[0]
        B, T = idx.shape
        
        # Current sequence for each pop member: (N, B, T)
        curr = idx.unsqueeze(0).expand(N, -1, -1).clone()
        
        # Track stopping
        finished = torch.zeros(N, B, dtype=torch.bool, device=self.device)
        last_was_newline = torch.zeros(N, B, dtype=torch.bool, device=self.device)
        
        for _ in range(max_tokens):
            with torch.no_grad():
                # Truncate if too long
                curr_len = curr.shape[2]
                if curr_len > self.model.cfg.max_seq_len:
                    curr_in = curr[:, :, -self.model.cfg.max_seq_len:]
                else:
                    curr_in = curr
                
                # Forward all N population members in parallel
                # curr_in: (N, B, T)
                logits = self._batched_forward(curr_in, a_list, b_list, signs)
                # Shape: (N, B, T, vocab)
                
                # Get last token logits
                last_logits = logits[:, :, -1, :] / temperature  # (N, B, vocab)
                probs = F.softmax(last_logits, dim=-1)
                
                # Sample: (N*B, vocab) -> (N*B, 1) -> (N, B, 1)
                next_tok = torch.multinomial(probs.view(N * B, -1), 1).view(N, B, 1)
                
                # Append
                curr = torch.cat([curr, next_tok], dim=2)
                
                # Check for double newline
                is_newline = (next_tok.squeeze(-1) == ord('\n'))
                finished = finished | (is_newline & last_was_newline)
                last_was_newline = is_newline
                
                if finished.all():
                    break
        
        return curr
    
    def train_step(self, iteration: int) -> Tuple[dict, bool]:
        lr = self.get_lr(iteration)
        
        if self.cfg.curriculum:
            self.difficulty = min(1.0, iteration / (self.cfg.max_iters * 0.9))
        else:
            self.difficulty = 1.0
        self.problem_gen.set_difficulty(self.difficulty)
        
        # Generate problems
        problems = self.problem_gen.generate_batch(min(4, self.cfg.batch_size))
        prompts = [p[0] for p in problems]
        
        # Sample perturbations (doubled for antithetic)
        N_half = self.population_size
        N = N_half * 2
        
        a_list, b_list = self._sample_perturbations(N_half)
        # Double for antithetic
        a_list = [torch.cat([a, a], dim=0) for a in a_list]
        b_list = [torch.cat([b, b], dim=0) for b in b_list]
        signs = torch.cat([torch.ones(N_half, device=self.device),
                          -torch.ones(N_half, device=self.device)])
        
        # Encode prompts
        toks = [self.tokenizer.encode(p) for p in prompts]
        max_len = max(len(t) for t in toks)
        padded = [[ord(' ')] * (max_len - len(t)) + t for t in toks]
        idx = torch.tensor(padded, dtype=torch.long, device=self.device)  # (B, T)
        
        # Generate for all population members
        self.model.eval()
        with torch.no_grad():
            outputs = self._batched_generate(idx, a_list, b_list, signs,
                                            self.cfg.max_answer_tokens, self.cfg.temperature)
        # outputs: (N, B, T + gen_len)
        
        # Evaluate fitness for each population member
        B = len(problems)
        all_fitness = []
        
        for pop_i in range(N):
            total_reward = 0.0
            for b_i, (prompt, expected_asm, info) in enumerate(problems):
                text = self.tokenizer.decode(outputs[pop_i, b_i].tolist())
                reward, _ = compute_reward(text, expected_asm, prompt, self.vm)
                total_reward += reward
            all_fitness.append(total_reward / B)
        
        # Fitness shaping
        fitness_tensor = torch.tensor(all_fitness, device=self.device)
        fitness_mean = fitness_tensor.mean()
        fitness_std = fitness_tensor.std() + 1e-8
        normalized = (fitness_tensor - fitness_mean) / fitness_std
        
        # Update weights: Δθ = (lr/σN) * Σ norm_fitness_i * sign_i * a_i ⊗ b_i^T
        with torch.no_grad():
            weights = normalized * signs  # (N,)
            
            for (w, d_out, d_in), a, b in zip(self._weights, a_list, b_list):
                # a: (N, d_out), b: (N, d_in)
                # Update = Σ weight_i * a_i ⊗ b_i^T = (weights * a).T @ b
                weighted_a = weights.view(N, 1) * a  # (N, d_out)
                update = weighted_a.T @ b  # (d_out, d_in)
                w.add_((lr / (self.sigma * N)) * update)
        
        # Metrics
        accuracy = sum(1 for f in all_fitness if f > 0.5) / N
        
        if accuracy > self.best_acc:
            self.best_acc = accuracy
            self.best_state = copy.deepcopy(self.model.state_dict())
        
        return {
            'reward': fitness_mean.item(),
            'accuracy': accuracy,
            'difficulty': self.difficulty,
            'lr': lr,
            'fitness_mean': fitness_mean.item(),
            'fitness_std': fitness_std.item(),
            'population': N,
            'loss': 0.0,
        }, False
    
    def restore_best(self):
        self.model.load_state_dict(self.best_state)


class BestOfN:
    """
    Best-of-N Training: Group sampling with supervised learning on top-k samples.
    
    Algorithm:
    1. Generate G samples per prompt (group sampling)
    2. Rank samples by reward within each group
    3. Select top-k samples from each group (with reward >= threshold)
    4. Supervised fine-tuning on selected samples only
    
    This is simpler than GRPO and cannot degrade the model
    since we only learn from the best samples.
    """
    
    def __init__(self, model: ToyLLM, tokenizer: ASCIITokenizer, cfg: RLConfig,
                 data_cfg: DataConfig, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.device = device
        
        self.problem_gen = RLProblemGen(data_cfg)
        self.vm = VM(data_cfg.num_registers)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=0.01)
        
        self.best_acc = 0.0
        self.best_state = copy.deepcopy(model.state_dict())
        self.difficulty = 0.0
        
        # Take top-k from each group
        self.top_k = max(1, cfg.group_size // 4)  # Top 25%
        self.min_reward = 0.0  # Only take samples with reward >= this
    
    def get_lr(self, it: int) -> float:
        if it < self.cfg.warmup_iters:
            return self.cfg.learning_rate * (it + 1) / self.cfg.warmup_iters
        ratio = (it - self.cfg.warmup_iters) / max(1, self.cfg.max_iters - self.cfg.warmup_iters)
        return self.cfg.min_lr + 0.5 * (1 + math.cos(math.pi * min(1, ratio))) * (self.cfg.learning_rate - self.cfg.min_lr)
    
    def train_step(self, iteration: int) -> Tuple[dict, bool]:
        lr = self.get_lr(iteration)
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        
        # Curriculum
        if self.cfg.curriculum:
            self.difficulty = min(1.0, iteration / (self.cfg.max_iters * 0.9))
        else:
            self.difficulty = 1.0
        self.problem_gen.set_difficulty(self.difficulty)
        
        B, G = self.cfg.batch_size, self.cfg.group_size
        problems = self.problem_gen.generate_batch(B)
        
        # Generate G samples per problem
        all_samples = []  # [(group_idx, reward, tokens), ...]
        all_rewards = []
        all_infos = []
        
        self.model.eval()
        
        for group_idx, (prompt, expected_asm, info) in enumerate(problems):
            prompt_toks = self.tokenizer.encode(prompt)
            
            # Generate G samples for this prompt
            idx = torch.tensor([prompt_toks] * G, dtype=torch.long, device=self.device)
            
            with torch.no_grad():
                out = self.model.generate(idx, self.cfg.max_answer_tokens, 
                                          temperature=self.cfg.temperature,
                                          stop_at_double_newline=True)
            
            group_samples = []
            for i in range(G):
                text = self.tokenizer.decode(out[i].tolist())
                
                # Compute reward
                reward, reward_info = compute_reward(text, expected_asm, prompt, self.vm)
                
                # Store full sequence tokens for training
                full_text = text.split('\n\n')[0]  # Up to first empty line
                full_toks = self.tokenizer.encode(full_text)
                
                group_samples.append((group_idx, reward, full_toks, reward_info))
                all_rewards.append(reward)
                all_infos.append(reward_info)
            
            # Rank by reward and take top-k
            group_samples.sort(key=lambda x: x[1], reverse=True)
            for sample in group_samples[:self.top_k]:
                if sample[1] >= self.min_reward:  # Only if reward is good enough
                    all_samples.append(sample)
        
        metrics = {
            'difficulty': self.difficulty,
            'lr': lr,
            'total_samples': B * G,
            'selected_samples': len(all_samples),
            'reward': sum(all_rewards) / len(all_rewards) if all_rewards else 0,
            'accuracy': sum(1 for i in all_infos if i['correct']) / len(all_infos) if all_infos else 0,
        }
        
        if len(all_samples) < 2:
            # Not enough good samples
            metrics['loss'] = 0.0
            return metrics, False
        
        # Supervised learning on selected samples
        self.model.train()
        
        # Pad sequences
        max_len = max(len(s[2]) for s in all_samples)
        padded = []
        for _, _, toks, _ in all_samples:
            padded.append([0] * (max_len - len(toks)) + toks)
        
        batch = torch.tensor(padded, dtype=torch.long, device=self.device)
        x, y = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
        
        _, loss = self.model(x, y)
        
        self.optimizer.zero_grad()
        loss.backward()
        if self.cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        
        metrics['loss'] = loss.item()
        
        # Track best
        if metrics['accuracy'] > self.best_acc:
            self.best_acc = metrics['accuracy']
            self.best_state = copy.deepcopy(self.model.state_dict())
        
        return metrics, False
    
    def restore_best(self):
        self.model.load_state_dict(self.best_state)


def best_of_n_generate(model: ToyLLM, tokenizer: ASCIITokenizer, prompt: str,
                       vm: VM, n: int = 8, max_tokens: int = 60, 
                       temperature: float = 0.8, device: str = 'cpu') -> str:
    """
    Generate N samples and return the one with highest reward.
    This cannot degrade the model since we don't update weights.
    """
    model.eval()
    prompt_toks = tokenizer.encode(prompt)
    idx = torch.tensor([prompt_toks] * n, dtype=torch.long, device=device)
    
    with torch.no_grad():
        out = model.generate(idx, max_tokens, temperature, stop_at_double_newline=True)
    
    best_text = None
    best_reward = float('-inf')
    
    for i in range(n):
        text = tokenizer.decode(out[i].tolist())
        gen_asm = text[len(prompt):]
        
        try:
            # Simple reward: correct = 1, wrong = 0
            state = vm.run(gen_asm)
            # Count garbage lines as penalty
            garbage = sum(1 for line in gen_asm.split('\n') 
                         if line.strip() and line.split()[0] not in ('mov', 'add', 'sub', 'mul', 'div', '#'))
            reward = 1.0 - garbage * 0.1
        except:
            reward = -1.0
        
        if reward > best_reward:
            best_reward = reward
            best_text = text
    
    return best_text if best_text else tokenizer.decode(out[0].tolist())


class RLProblemGen:
    """Generate RL problems: model compiles expressions to assembly."""
    
    def __init__(self, cfg: DataConfig):
        self.cfg = cfg
        self.compiler = ExprCompiler(cfg.num_registers)
        self.vm = VM(cfg.num_registers)
        self.rng = random.Random()
        self.difficulty = 0.0  # 0.0 = easy, 1.0 = hard
    
    def set_difficulty(self, level: float):
        """Set difficulty from 0.0 (simple) to 1.0 (complex)."""
        self.difficulty = max(0.0, min(1.0, level))
    
    def _random_expr(self, depth: int = 0, max_depth: int = 1) -> str:
        """Generate random expression based on difficulty."""
        if depth >= max_depth or (depth > 0 and self.rng.random() < 0.4):
            return str(self.rng.randint(self.cfg.min_num, self.cfg.max_num))
        
        # Choose expression type based on depth
        if depth == 0 and max_depth >= 2 and self.rng.random() < 0.3:
            # At top level, sometimes make (a op b) op (c op d)
            a = str(self.rng.randint(self.cfg.min_num, self.cfg.max_num))
            b = str(self.rng.randint(self.cfg.min_num, self.cfg.max_num))
            c = str(self.rng.randint(self.cfg.min_num, self.cfg.max_num))
            d = str(self.rng.randint(self.cfg.min_num, self.cfg.max_num))
            
            op1 = self.rng.choice(list(self.cfg.operations))
            op2 = self.rng.choice(list(self.cfg.operations))
            op3 = self.rng.choice(list(self.cfg.operations))
            
            if op1 == '/' and b == '0': b = '1'
            if op3 == '/' and d == '0': d = '1'
            
            return f"({a} {op1} {b}) {op2} ({c} {op3} {d})"
        
        left = self._random_expr(depth + 1, max_depth)
        right = self._random_expr(depth + 1, max_depth)
        op = self.rng.choice(list(self.cfg.operations))
        
        if op == '/' and right == '0':
            right = str(self.rng.randint(1, self.cfg.max_num))
        
        expr = f"{left} {op} {right}"
        
        if depth > 0 and self.rng.random() < 0.5:
            expr = f"({expr})"
        
        return expr
    
    def generate_one(self) -> Tuple[str, str, dict]:
        """
        Generate a problem based on current difficulty.
        Always mixes in some easier problems for stability.
        
        Returns:
            prompt: "# expr\\n" - model should complete with assembly
            expected_asm: The correct assembly
            info: {expr, result, asm}
        """
        # Mix difficulties even at high difficulty level
        # This prevents catastrophic forgetting of simple cases
        r = self.rng.random()
        if r < 0.3:
            # 30% always simple
            max_depth = 1
        elif r < 0.5:
            # 20% always medium  
            max_depth = 2
        else:
            # 50% based on difficulty setting
            if self.difficulty < 0.3:
                max_depth = 1
            elif self.difficulty < 0.6:
                max_depth = 2
            else:
                max_depth = 3
        
        expr = self._random_expr(max_depth=max_depth)
        
        try:
            asm, result = self.compiler.compile_to_r0(expr)
            # Skip if too complex
            if abs(result) > 50000:
                raise ValueError("too large")
        except:
            # Fallback
            a = self.rng.randint(1, self.cfg.max_num)
            b = self.rng.randint(1, self.cfg.max_num)
            op = self.rng.choice(['+', '-', '*'])
            expr = f"{a} {op} {b}"
            asm, result = self.compiler.compile_to_r0(expr)
        
        prompt = f"# {expr}\n"
        
        return prompt, asm, {'expr': expr, 'result': result, 'asm': asm}
    
    def generate_batch(self, n: int):
        return [self.generate_one() for _ in range(n)]


def compute_reward(generated: str, expected_asm: str, prompt: str, vm: VM) -> Tuple[float, dict]:
    """
    Compute reward by running the generated assembly in VM.
    Correct if r0 matches expected result.
    Penalize garbage tokens.
    """
    # Extract the assembly part (after the comment line)
    if prompt in generated:
        asm = generated[len(prompt):]
    else:
        asm = generated
    
    # Get expected result by running expected asm
    expected_state = vm.run(expected_asm)
    expected_r0 = expected_state['r0']
    expected_lines = len(expected_asm.strip().split('\n'))
    
    # Run generated asm
    try:
        gen_state = vm.run(asm)
        gen_r0 = gen_state['r0']
    except:
        gen_r0 = None
    
    # Count valid instruction lines vs garbage
    valid_instructions = 0
    garbage_chars = 0
    for line in asm.split('\n'):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if parts and parts[0] in ('mov', 'add', 'sub', 'mul', 'div'):
            valid_instructions += 1
        else:
            garbage_chars += len(line)
    
    info = {
        'generated_asm': asm.strip()[:80],
        'expected_r0': expected_r0,
        'got_r0': gen_r0,
        'correct': gen_r0 == expected_r0,
        'valid_instr': valid_instructions,
        'garbage': garbage_chars
    }
    
    if gen_r0 == expected_r0:
        # Correct! But penalize garbage
        if garbage_chars == 0:
            return 1.0, info  # Perfect
        else:
            # Penalize proportionally to garbage
            penalty = min(0.5, garbage_chars * 0.02)
            return 1.0 - penalty, info
    elif gen_r0 is not None:
        # Wrong result
        if garbage_chars == 0:
            return -0.3, info  # Clean but wrong
        else:
            return -0.5, info  # Wrong and garbage
    else:
        # Couldn't even run it
        return -1.0, info



# ============================================================================
# Training
# ============================================================================

def get_lr(it: int, cfg: TrainingConfig) -> float:
    if it < cfg.warmup_iters:
        return cfg.learning_rate * it / cfg.warmup_iters
    ratio = (it - cfg.warmup_iters) / (cfg.max_iters - cfg.warmup_iters)
    return cfg.min_lr + 0.5 * (1 + math.cos(math.pi * ratio)) * (cfg.learning_rate - cfg.min_lr)


@torch.no_grad()
def estimate_loss(model, train_loader, val_loader, cfg, device):
    model.eval()
    out = {}
    for name, loader in [('train', train_loader), ('val', val_loader)]:
        losses = []
        for i, (x, y) in enumerate(loader):
            if i >= cfg.eval_iters:
                break
            _, loss = model(x.to(device), y.to(device))
            losses.append(loss.item())
        out[name] = sum(losses) / len(losses) if losses else 0
    model.train()
    return out


def train(cfg: Config):
    device = cfg.device
    ckpt_path = Path(cfg.checkpoint.path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    
    if cfg.checkpoint.fresh and ckpt_path.exists():
        ckpt_path.unlink()
    
    data_cfg = DataConfig(**OmegaConf.to_container(cfg.data))
    gen = ProgramGenerator(data_cfg)
    text = gen.generate_dataset(data_cfg.num_samples)
    
    tokenizer = ASCIITokenizer()
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    
    train_cfg = TrainingConfig(**OmegaConf.to_container(cfg.training))
    model_cfg = ModelConfig(**OmegaConf.to_container(cfg.model))
    
    split = int(len(data) * 0.9)
    train_set = TextDataset(data[:split], model_cfg.max_seq_len)
    val_set = TextDataset(data[split:], model_cfg.max_seq_len)
    
    train_loader = DataLoader(train_set, batch_size=train_cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=train_cfg.batch_size)
    
    print(f"Train: {len(train_set):,}, Val: {len(val_set):,}")
    
    model = ToyLLM(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)
    
    start_iter, best_loss = 0, float('inf')
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        start_iter = ckpt.get('iteration', 0)
        best_loss = ckpt.get('val_loss', float('inf'))
        print(f"Resumed from iter {start_iter}")
    
    it = start_iter
    while it < train_cfg.max_iters:
        for x, y in train_loader:
            if it >= train_cfg.max_iters:
                break
            
            lr = get_lr(it, train_cfg)
            for pg in optimizer.param_groups:
                pg['lr'] = lr
            
            _, loss = model(x.to(device), y.to(device))
            optimizer.zero_grad()
            loss.backward()
            if train_cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()
            
            if it % 100 == 0:
                print(f"iter {it:5d} | loss {loss.item():.4f} | lr {lr:.2e}")
            
            if it > 0 and it % train_cfg.eval_interval == 0:
                losses = estimate_loss(model, train_loader, val_loader, train_cfg, device)
                print(f">>> eval | train {losses['train']:.4f} | val {losses['val']:.4f}")
                
                if losses['val'] < best_loss:
                    best_loss = losses['val']
                    torch.save({'model': model.state_dict(), 'iteration': it, 'val_loss': best_loss,
                                'config': OmegaConf.to_container(cfg)}, ckpt_path)
                    print(f">>> saved")
                
                # Sample
                prompt = "# 7 + 3\n"
                idx = torch.tensor([tokenizer.encode(prompt)], device=device)
                out = model.generate(idx, 50, temperature=0.5)
                gen = tokenizer.decode(out[0].tolist())
                print(f"Sample:\n{gen}")
                
                # Verify with VM
                try:
                    r0 = VM(data_cfg.num_registers).run(gen[len(prompt):])['r0']
                    print(f"  VM r0 = {r0} (expected 10)")
                except Exception as e:
                    print(f"  VM error: {e}")
                print()
            
            it += 1
    
    print(f"Done. Best val loss: {best_loss:.4f}")


def rl_train(cfg: Config):
    device = cfg.device
    ckpt_path = Path(cfg.checkpoint.path)
    rl_path = Path(cfg.checkpoint.rl_path)
    rl_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not ckpt_path.exists():
        raise ValueError(f"No checkpoint at {ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_cfg = ModelConfig(**ckpt['config']['model'])
    model = ToyLLM(model_cfg).to(device)
    model.load_state_dict(ckpt['model'])
    
    tokenizer = ASCIITokenizer()
    rl_cfg = RLConfig(**OmegaConf.to_container(cfg.rl))
    data_cfg = DataConfig(**OmegaConf.to_container(cfg.data))
    
    # Choose RL method
    method = rl_cfg.method.lower()
    if method == "rft":
        trainer = RejectionSamplingFT(model, tokenizer, rl_cfg, data_cfg, device)
        print(f"Using Rejection Sampling Fine-Tuning (safest)")
    elif method == "positive_only":
        trainer = PositiveOnlyGRPO(model, tokenizer, rl_cfg, data_cfg, device)
        print(f"Using Positive-Only GRPO (safe)")
    elif method == "eggroll":
        trainer = EGGROLL(model, tokenizer, rl_cfg, data_cfg, device)
        print(f"Using EGGROLL - Batched Rank-1 Evolution Strategies")
        print(f"  σ={rl_cfg.es_sigma}, pop={rl_cfg.es_population} (N={rl_cfg.es_population*2} parallel evals)")
    elif method == "bon":
        trainer = BestOfN(model, tokenizer, rl_cfg, data_cfg, device)
        print(f"Using Best-of-N (group sampling + supervised learning)")
    else:  # grpo
        trainer = GRPO(model, tokenizer, rl_cfg, data_cfg, device)
        print(f"Using standard GRPO")
    
    vm = VM(data_cfg.num_registers)
    
    print(f"RL: {rl_cfg.max_iters} iters, curriculum={rl_cfg.curriculum}, method={method}")
    
    # Show initial model capability
    print("\nInitial model test:")
    model.eval()
    for expr in ["5 + 3", "10 * 2", "(4 + 3) * 2"]:
        prompt = f"# {expr}\n"
        idx = torch.tensor([tokenizer.encode(prompt)], device=device)
        out = model.generate(idx, 40, temperature=0.1)
        text = tokenizer.decode(out[0].tolist())
        gen = text[len(prompt):].split('\n\n')[0]
        try:
            r0 = vm.run(gen)['r0']
        except:
            r0 = "err"
        print(f"  # {expr} -> r0={r0}")
        print(f"    {gen.replace(chr(10), ' | ')[:60]}")
    model.train()
    print()
    
    for it in range(rl_cfg.max_iters):
        metrics, early_stop = trainer.train_step(it)
        
        if it % 10 == 0:
            diff_str = f"d={metrics.get('difficulty', 0):.1f}"
            if method == "rft":
                print(f"iter {it:4d} | found {metrics.get('correct_found', 0):2d}/{metrics.get('attempts', 0)} | "
                      f"acc {metrics['accuracy']:.0%} | loss {metrics['loss']:.4f} | {diff_str}")
            elif method == "positive_only":
                print(f"iter {it:4d} | reward {metrics['reward']:+.2f} | acc {metrics['accuracy']:.0%} | "
                      f"pos {metrics.get('positive_samples', 0)} | kl {metrics['kl']:.4f} | {diff_str}")
            elif method == "eggroll":
                n_evals = metrics.get('population', 0)
                print(f"iter {it:4d} | reward {metrics['reward']:+.2f} | acc {metrics['accuracy']:.0%} | "
                      f"evals {n_evals} | f_μ {metrics.get('fitness_mean', 0):+.2f} | {diff_str}")
            elif method == "bon":
                print(f"iter {it:4d} | reward {metrics['reward']:+.2f} | acc {metrics['accuracy']:.0%} | "
                      f"sel {metrics.get('selected_samples', 0)}/{metrics.get('total_samples', 0)} | loss {metrics['loss']:.4f} | {diff_str}")
            else:
                print(f"iter {it:4d} | reward {metrics['reward']:+.2f} | acc {metrics['accuracy']:.0%} clean {metrics.get('clean', 0):.0%} | "
                      f"kl {metrics['kl']:.4f} | {diff_str}")
        
        # Show sample every 50 iters
        if it % 50 == 0 and it > 0:
            model.eval()
            prompt = "# 5 + 3\n"
            idx = torch.tensor([tokenizer.encode(prompt)], device=device)
            out = model.generate(idx, 30, temperature=0.1)
            gen = tokenizer.decode(out[0].tolist())
            asm = gen[len(prompt):].split('\n\n')[0]
            try:
                r0 = vm.run(asm)['r0']
            except:
                r0 = "err"
            print(f"  Quick check: # 5 + 3 -> r0={r0} (expected 8) | {asm.replace(chr(10), ' | ')[:40]}")
            model.train()
        
        if early_stop:
            print(f"Early stop: KL {metrics.get('kl', 0):.4f} > {rl_cfg.kl_max}")
            trainer.restore_best()
            break
        
        if it > 0 and it % rl_cfg.eval_interval == 0:
            print("\n" + "=" * 50)
            print("Eval:")
            model.eval()
            
            # Test at different difficulties
            for diff_name, diff_val in [("easy", 0.0), ("med", 0.5), ("hard", 1.0)]:
                trainer.problem_gen.set_difficulty(diff_val)
                problems = trainer.problem_gen.generate_batch(2)
                
                for prompt, expected_asm, info in problems:
                    idx = torch.tensor([tokenizer.encode(prompt)], device=device)
                    out = model.generate(idx, 50, temperature=0.1)
                    text = tokenizer.decode(out[0].tolist())
                    
                    # Run generated asm in VM
                    gen_asm = text[len(prompt):]
                    
                    # Count valid vs garbage lines
                    valid_lines = []
                    garbage = []
                    for line in gen_asm.split('\n'):
                        line_s = line.strip()
                        if not line_s:
                            continue
                        parts = line_s.split()
                        if parts and parts[0] in ('mov', 'add', 'sub', 'mul', 'div'):
                            valid_lines.append(line_s)
                        else:
                            garbage.append(line_s[:20])
                    
                    try:
                        state = vm.run(gen_asm)
                        got_r0 = state['r0']
                    except:
                        got_r0 = "err"
                    
                    mark = "✓" if got_r0 == info['result'] else "✗"
                    clean = "(clean)" if not garbage else f"(+{len(garbage)} garbage)"
                    print(f"  [{diff_name}] {mark} {info['expr']} => r0={got_r0} (expected {info['result']}) {clean}")
            
            model.train()
            print("=" * 50 + "\n")
        
        if it > 0 and it % rl_cfg.save_interval == 0:
            torch.save({'model': trainer.best_state, 'config': OmegaConf.to_container(cfg)}, rl_path)
    
    trainer.restore_best()
    torch.save({'model': model.state_dict(), 'config': OmegaConf.to_container(cfg)}, rl_path)
    print(f"Done. Best accuracy: {trainer.best_acc if hasattr(trainer, 'best_acc') else trainer.best_score:.1%}")


def generate(cfg: Config):
    device = cfg.device
    ckpt_path = Path(cfg.checkpoint.rl_path)
    if not ckpt_path.exists():
        ckpt_path = Path(cfg.checkpoint.path)
    if not ckpt_path.exists():
        raise ValueError("No checkpoint")
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_cfg = ModelConfig(**ckpt['config']['model'])
    model = ToyLLM(model_cfg).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    tokenizer = ASCIITokenizer()
    gen_cfg = GenerateConfig(**OmegaConf.to_container(cfg.generate))
    
    prompt = (gen_cfg.prompt or "# 5 * (2 + 3)\n").replace('\\n', '\n')
    idx = torch.tensor([tokenizer.encode(prompt)], device=device)
    out = model.generate(idx, gen_cfg.max_tokens, gen_cfg.temperature)
    print(tokenizer.decode(out[0].tolist()))


@hydra.main(version_base=None, config_name="config")
def main(cfg: Config):
    if cfg.command == "train":
        train(cfg)
    elif cfg.command == "rl":
        rl_train(cfg)
    elif cfg.command == "generate":
        generate(cfg)
    else:
        print(f"Unknown: {cfg.command}")


if __name__ == '__main__':
    main()