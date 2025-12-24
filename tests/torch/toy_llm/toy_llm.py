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

# ============================================================================
# Configuration (Single Source of Truth - no YAML needed)
# ============================================================================

@dataclass
class ModelConfig:
    n_layers     : int      = 1
    n_recurrence : int      = 6
    n_heads      : int      = 6
    d_model      : int      = 384
    d_ff         : int      = 1536
    dropout      : float    = 0.1
    max_seq_len  : int      = 512
    vocab_size   : int      = 128


@dataclass
class TrainingConfig:
    start_iteration     : int       = 0
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
class DataConfig:
    data_dir            : Optional[str] = None
    extensions          : Optional[List[str]] = None
    wiki                : bool = False
    max_wiki_chars      : int = 100_000_000
    brackets            : bool = False
    bracket_samples     : int = 100000


@dataclass
class CheckpointConfig:
    path                : str       = ".tmp/toy_llm.pt"
    fresh               : bool      = False


@dataclass
class GenerateConfig:
    prompt              : Optional[str] = None
    max_tokens          : int = 500
    temperature         : float = 0.8
    top_k               : int = 50


@dataclass 
class Config:
    """Full config combining all sub-configs."""
    model               : ModelConfig = field(default_factory=ModelConfig)
    training            : TrainingConfig = field(default_factory=TrainingConfig)
    data                : DataConfig = field(default_factory=DataConfig)
    checkpoint          : CheckpointConfig = field(default_factory=CheckpointConfig)
    generate            : GenerateConfig = field(default_factory=GenerateConfig)
    command             : str = "train"
    device              : str = "cuda" if torch.cuda.is_available() else "cpu"


# Register with Hydra's ConfigStore - this replaces the YAML file
cs = ConfigStore.instance()
cs.store(name="config", node=Config)


# ============================================================================
# ASCII Tokenizer (trivial)
# ============================================================================

class ASCIITokenizer:
    """Dead simple: char -> ord, clamp to [0, 127]"""
    vocab_size = 128
    
    def encode(self, text: str) -> list[int]:
        return [min(ord(c), 127) for c in text]
    
    def decode(self, tokens: list[int]) -> str:
        return ''.join(chr(t) for t in tokens)


# ============================================================================
# Dataset
# ============================================================================

class CodeDataset(Dataset):
    def __init__(self, data: torch.Tensor, seq_len: int):
        self.data = data
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.data) - self.seq_len - 1
    
    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return x, y


def filter_ascii(text: str) -> str:
    """Keep only ASCII characters (0-127), drop everything else."""
    return ''.join(c for c in text if ord(c) < 128)


def generate_balanced_brackets(num_samples: int = 100000, min_depth: int = 1, max_depth: int = 20, seed: int = 42) -> str:
    """
    Generate a dataset of balanced bracket sequences.
    
    This is a classic test for whether a model can learn:
    - Matching/counting (each open must have a close)
    - Hierarchical structure (proper nesting)
    - Long-range dependencies
    
    Bracket types: () [] {} <>
    """
    import random
    rng = random.Random(seed)
    
    BRACKETS = [('(', ')'), ('[', ']'), ('{', '}'), ('<', '>')]
    
    def generate_one(max_d: int) -> str:
        """Generate a single balanced bracket sequence."""
        if max_d <= 0:
            return ''
        
        # Choose how many top-level groups
        n_groups = rng.randint(1, 3)
        parts = []
        
        for _ in range(n_groups):
            open_b, close_b = rng.choice(BRACKETS)
            # Recursively generate inside content
            inner = generate_one(max_d - 1) if rng.random() > 0.3 else ''
            parts.append(f"{open_b}{inner}{close_b}")
        
        return ''.join(parts)
    
    sequences = []
    total_chars = 0
    
    for i in range(num_samples):
        depth = rng.randint(min_depth, max_depth)
        seq = generate_one(depth)
        
        # Skip empty or trivial sequences
        if len(seq) < 2:
            continue
            
        sequences.append(seq)
        total_chars += len(seq) + 1  # +1 for newline
        
        if i % 10000 == 0:
            print(f"  Generated {i:,} sequences, {total_chars:,} chars...")
    
    # Join with newlines so the model learns sequence boundaries
    text = '\n'.join(sequences)
    print(f"Generated {len(sequences):,} bracket sequences, {len(text):,} characters")
    
    # Print some examples
    print("Examples:")
    for s in sequences[:5]:
        print(f"  {s}")
    
    return text


def mix_texts(texts: list[str], chunk_size: int = 10000) -> str:
    """
    Interleave multiple text sources by chunks.
    This prevents the model from training on all of source A, then all of B
    (which causes catastrophic forgetting).
    """
    import random
    
    # Split each text into chunks
    all_chunks = []
    for text in texts:
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        all_chunks.extend(chunks)
    
    # Shuffle chunks
    random.shuffle(all_chunks)
    
    combined = ''.join(all_chunks)
    print(f"Mixed {len(all_chunks)} chunks from {len(texts)} sources -> {len(combined):,} chars")
    return combined


def load_wikipedia(cache_dir: str = ".tmp/wikipedia", max_articles: int = None, max_chars: int = 100_000_000) -> str:
    """
    Load English Wikipedia from HuggingFace, with local caching.
    
    Args:
        cache_dir: Directory to cache the processed text
        max_articles: Max number of articles to load (None = all)
        max_chars: Stop after this many characters (default 100M)
    """
    from datasets import load_dataset
    
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    processed_file = cache_path / "wikipedia_ascii.txt"
    
    # Use cached version if available
    if processed_file.exists():
        print(f"Loading cached Wikipedia from {processed_file}...")
        text = processed_file.read_text(encoding='utf-8')
        print(f"Loaded {len(text):,} characters from cache")
        return text
    
    print("Downloading English Wikipedia (this may take a while)...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", cache_dir=str(cache_path / "hf_cache"))
    
    print(f"Processing {len(ds):,} articles...")
    texts = []
    total_chars = 0
    
    for i, article in enumerate(ds):
        if max_articles and i >= max_articles:
            break
        
        title = filter_ascii(article['title'])
        content = filter_ascii(article['text'])
        
        article_text = f"# {title}\n\n{content}\n\n"
        texts.append(article_text)
        total_chars += len(article_text)
        
        if i % 10000 == 0:
            print(f"  Processed {i:,} articles, {total_chars:,} chars...")
        
        if max_chars and total_chars >= max_chars:
            print(f"  Reached {max_chars:,} char limit")
            break
    
    combined = ''.join(texts)
    
    # Cache processed text
    print(f"Caching processed text to {processed_file}...")
    processed_file.write_text(combined, encoding='utf-8')
    
    print(f"Loaded {len(texts):,} articles, {len(combined):,} characters")
    return combined


def load_codebase(data_dir: str, extensions: list[str] = None, exclude_dirs: set[str] = None) -> str:
    TMP_DIR = Path('.tmp/toy_llm_codebase')
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    if (TMP_DIR / 'cache.txt').exists():
        print(f"Loading cached codebase from {(TMP_DIR / 'cache.txt')}...")
        text = (TMP_DIR / 'cache.txt').read_text(encoding='utf-8')
        print(f"Loaded {len(text):,} characters from cache")
        return text 

    """Recursively load all code files from a directory."""
    if extensions is None:
        extensions = ['.py', '.c', '.cpp', '.h', '.hpp', '.js', '.ts', '.rs', '.go', 
                      '.java', '.cs', '.rb', '.lua', '.sh', '.md', '.txt', '.json']
    
    if exclude_dirs is None:
        exclude_dirs = {
            '.git', '.hg', '.svn',
            '.tmp',
            'wikipedia',
            'build', 'dist', 'target', 'out', 'bin',
            '.tox', '.nox', '.eggs', '*.egg-info',
            'vendor', 'third_party', 'external',
            'uv_cache', '.uv', '.cache', 'cache',
        }
    
    data_dir = Path(data_dir).resolve()
    texts = []
    
    def should_skip(path: Path) -> bool:
        return any(part in exclude_dirs or part.endswith('.egg-info') for part in path.parts)
    
    def walk_safe(directory: Path):
        """Walk directory tree, handling Windows path length errors."""
        try:
            entries = list(directory.iterdir())
        except (OSError, PermissionError) as e:
            return
        
        for entry in entries:
            try:
                if entry.is_dir():
                    if entry.name not in exclude_dirs and not entry.name.endswith('.egg-info'):
                        yield from walk_safe(entry)
                elif entry.is_file():
                    yield entry
            except (OSError, PermissionError):
                continue
    
    total_chars = 0

    for file_path in walk_safe(data_dir):
        if should_skip(file_path):
            continue
        
        if file_path.suffix not in extensions:
            continue
        
        try:
            text = file_path.read_text(encoding='utf-8', errors='ignore')
            rel_path = file_path.relative_to(data_dir)
            texts.append(f"# FILE: {rel_path}\n{text}\n\n")

            total_chars += len(text)
            if total_chars > 1_000_000_000:
                print(f"Reached 1,000,000,000 character limit, stopping load.")
                break

        except Exception as e:
            print(f"Skipping {file_path}: {e}")
    
    combined = ''.join(texts)
    print(f"Loaded {len(texts)} files, {len(combined):,} characters")

    # Cache processed text
    print(f"Caching processed text to {(TMP_DIR / 'cache.txt')}...")
    (TMP_DIR / 'cache.txt').write_text(combined, encoding='utf-8')

    return combined


# ============================================================================
# Model Components
# ============================================================================

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


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        
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
        
        # Apply RoPE
        cos, sin = self.rope(T)
        cos, sin = cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)
        q, k = apply_rope(q, k, cos, sin)
        
        # Scaled dot-product attention with causal mask
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout.p if self.training else 0.0)
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.out(out))


class MLP(nn.Module):
    """SwiGLU-style MLP"""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.w3 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
    
    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================================
# The Model
# ============================================================================

class ToyLLM(nn.Module):
    def __init__(self, model_cfg: ModelConfig):
        super().__init__()
        self.model_cfg = model_cfg
        
        self.token_emb = nn.Embedding(model_cfg.vocab_size, model_cfg.d_model)
        self.dropout = nn.Dropout(model_cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(model_cfg) for _ in range(model_cfg.n_layers)])
        self.norm = RMSNorm(model_cfg.d_model)
        self.lm_head = nn.Linear(model_cfg.d_model, model_cfg.vocab_size, bias=False)
        
        # Weight tying
        self.token_emb.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Count parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Model parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None, n_recurrence: int = 6):
        x = self.dropout(self.token_emb(idx))
        
        for recurrent_step in range(n_recurrence):
            for block in self.blocks:
                x = block(x)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 0.8, top_k: int = 50, n_recurrence: int = 6):
        """Autoregressive generation with sampling."""
        self.eval()
        
        for _ in range(max_new_tokens):
            # Crop context if too long
            idx_cond = idx if idx.size(1) <= self.model_cfg.max_seq_len else idx[:, -self.model_cfg.max_seq_len:]
            
            logits, _ = self(idx_cond, n_recurrence=n_recurrence)
            logits = logits[:, -1, :] / temperature
            
            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        
        return idx


# ============================================================================
# Training
# ============================================================================

def get_lr(it: int, cfg: TrainingConfig) -> float:
    """Cosine decay with warmup."""
    if it < cfg.warmup_iters:
        return cfg.learning_rate * it / cfg.warmup_iters
    if it > cfg.max_iters:
        return cfg.min_lr
    
    decay_ratio = (it - cfg.warmup_iters) / (cfg.max_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


@torch.no_grad()
def estimate_loss(model, train_loader, val_loader, cfg: TrainingConfig, device: str, n_recurrence: int):
    model.eval()
    out = {}
    
    for name, loader in [('train', train_loader), ('val', val_loader)]:
        losses = []
        for i, (x, y) in enumerate(loader):
            if i >= cfg.eval_iters:
                break
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y, n_recurrence=n_recurrence)
            losses.append(loss.item())
        out[name] = sum(losses) / len(losses)
    
    model.train()
    return out


def train(cfg: Config):
    """Main training function."""
    model_cfg = cfg.model
    train_cfg = cfg.training
    data_cfg = cfg.data
    ckpt_cfg = cfg.checkpoint
    device = cfg.device
    
    # Auto-checkpoint path - use absolute path to avoid Hydra cwd issues
    auto_checkpoint = Path(ckpt_cfg.path)
    if not auto_checkpoint.is_absolute():
        auto_checkpoint = Path(hydra.utils.get_original_cwd()) / ckpt_cfg.path
    auto_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    
    # Fresh start - delete existing checkpoint
    if ckpt_cfg.fresh and auto_checkpoint.exists():
        print(f"fresh=true specified, removing existing checkpoint {auto_checkpoint}")
        auto_checkpoint.unlink()
    
    # Load data
    tokenizer = ASCIITokenizer()
    
    texts = []
    sample_prompt = "The "
    
    if data_cfg.brackets:
        bracket_text = generate_balanced_brackets(num_samples=data_cfg.bracket_samples)
        texts.append(bracket_text)
        sample_prompt = "[("
    
    if data_cfg.wiki:
        wiki_text = load_wikipedia(cache_dir=".tmp/wikipedia", max_chars=data_cfg.max_wiki_chars)
        texts.append(wiki_text)
        sample_prompt = "The "
    
    if data_cfg.data_dir:
        # Resolve data_dir relative to original cwd
        data_dir = Path(data_cfg.data_dir)
        if not data_dir.is_absolute():
            data_dir = Path(hydra.utils.get_original_cwd()) / data_dir
        extensions = list(data_cfg.extensions) if data_cfg.extensions else None
        code_text = load_codebase(str(data_dir), extensions)
        texts.append(code_text)
        sample_prompt = "def "
    
    if not texts:
        raise ValueError("Must specify at least one of: data.data_dir, data.wiki, or data.brackets")
    
    # Combine all data sources
    if len(texts) > 1:
        print(f"Mixing {len(texts)} data sources...")
        text = mix_texts(texts, chunk_size=10000)
        sample_prompt = "The "
    else:
        text = texts[0]
    
    if len(text) < model_cfg.max_seq_len * 10:
        raise ValueError(f"Not enough data! Got {len(text)} chars, need at least {model_cfg.max_seq_len * 10}")
    
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    
    # Split train/val
    split_idx = int(len(data) * train_cfg.train_split)
    train_data, val_data = data[:split_idx], data[split_idx:]
    
    train_dataset = CodeDataset(train_data, model_cfg.max_seq_len)
    val_dataset = CodeDataset(val_data, model_cfg.max_seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=train_cfg.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=train_cfg.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"Train: {len(train_dataset):,} samples, Val: {len(val_dataset):,} samples")
    
    # Create model - convert OmegaConf to dataclass for model
    model_config = ModelConfig(**OmegaConf.to_container(model_cfg))
    model = ToyLLM(model_config).to(device)
    
    n_recurrence = model_cfg.n_recurrence
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=train_cfg.weight_decay
    )
    
    # Load checkpoint if exists
    start_iteration  = 0
    start_epoch      = 0
    best_val_loss    = float('inf')
    
    if auto_checkpoint.exists():
        print(f"Loading checkpoint from {auto_checkpoint}...")
        ckpt = torch.load(auto_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'], strict=False)
        start_iteration  = ckpt.get('iteration', 0)
        start_epoch      = ckpt.get('epoch', 0)
        best_val_loss    = ckpt.get('val_loss', float('inf'))
        print(f"Resumed from iteration {start_iteration}, epoch {start_epoch}, best_val_loss {best_val_loss:.4f}")
    
    if train_cfg.compile_model and hasattr(torch, 'compile'):
        print("Compiling model...")
        model = torch.compile(model)
    
    # Training loop
    iteration = start_iteration
    if train_cfg.start_iteration >= 0:
        iteration = train_cfg.start_iteration
    epoch = start_epoch
    iters_per_epoch = len(train_loader)
    
    # Convert to plain TrainingConfig for get_lr
    train_config = TrainingConfig(**OmegaConf.to_container(train_cfg))
    
    print(f"Iterations per epoch: {iters_per_epoch}")
    
    while iteration < train_cfg.max_iters:
        epoch_start_iter = iteration
        
        for batch_idx, (x, y) in enumerate(train_loader):
            if iteration >= train_cfg.max_iters:
                break
            
            t0 = time.time()
            x, y = x.to(device), y.to(device)
            
            # Update learning rate
            lr = get_lr(iteration, train_config)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            # Forward + backward
            _, loss = model(x, y, n_recurrence=n_recurrence)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            
            if train_cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            
            optimizer.step()
            
            dt = time.time() - t0
            
            # Logging
            if iteration % 100 == 0:
                print(f"epoch {epoch:3d} | iter {iteration:5d} | loss {loss.item():.4f} | lr {lr:.2e} | {dt*1000:.1f}ms")
            
            # Evaluation
            if iteration > 0 and iteration % train_cfg.eval_interval == 0:
                losses = estimate_loss(model, train_loader, val_loader, train_config, device, n_recurrence)
                print(f">>> eval | train {losses['train']:.4f} | val {losses['val']:.4f}")
                
                # Save best model
                if losses['val'] < best_val_loss:
                    best_val_loss = losses['val']
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'config': OmegaConf.to_container(cfg),
                        'iteration': iteration,
                        'epoch': epoch,
                        'val_loss': best_val_loss,
                    }, auto_checkpoint)
                    print(f">>> saved best checkpoint to {auto_checkpoint}")
                
                # Generate sample
                idx = torch.tensor([tokenizer.encode(sample_prompt)], device=device)
                generated = model.generate(idx, max_new_tokens=200, temperature=0.8, n_recurrence=n_recurrence)
                print("-" * 60)
                print(tokenizer.decode(generated[0].tolist()))
                print("-" * 60)
            
            iteration += 1
        
        # End of epoch - save checkpoint
        epoch += 1
        print(f">>> Epoch {epoch} complete ({iteration - epoch_start_iter} iters)")
        
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': OmegaConf.to_container(cfg),
            'iteration': iteration,
            'epoch': epoch,
            'val_loss': best_val_loss,
        }, auto_checkpoint)
        print(f">>> saved epoch checkpoint to {auto_checkpoint}")
    
    return model, tokenizer


# ============================================================================
# Inference
# ============================================================================

def load_model(checkpoint_path: str, device: str = None):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    cfg = checkpoint['config']
    
    # Handle both old-style Config and new-style dict configs
    if isinstance(cfg, dict):
        model_dict = cfg.get('model', cfg)
        model_cfg = ModelConfig(**{k: v for k, v in model_dict.items() if k in ModelConfig.__dataclass_fields__})
        n_recurrence = model_dict.get('n_recurrence', 6)
    else:
        # Old dataclass-style config
        model_cfg = ModelConfig(
            n_layers=getattr(cfg, 'n_layers', 1),
            n_recurrence=getattr(cfg, 'n_recurrence', 6),
            n_heads=getattr(cfg, 'n_heads', 6),
            d_model=getattr(cfg, 'd_model', 384),
            d_ff=getattr(cfg, 'd_ff', 1536),
            dropout=getattr(cfg, 'dropout', 0.1),
            max_seq_len=getattr(cfg, 'max_seq_len', 512),
            vocab_size=getattr(cfg, 'vocab_size', 128),
        )
        n_recurrence = getattr(cfg, 'n_recurrence', 6)
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = ToyLLM(model_cfg).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    return model, ASCIITokenizer(), device, n_recurrence


def generate_cmd(cfg: Config):
    """Generation command."""
    gen_cfg = cfg.generate
    ckpt_path = cfg.checkpoint.path
    
    # Resolve checkpoint path relative to original cwd
    if not Path(ckpt_path).is_absolute():
        ckpt_path = str(Path(hydra.utils.get_original_cwd()) / ckpt_path)
    
    model, tokenizer, device, n_recurrence = load_model(ckpt_path)
    
    if gen_cfg.prompt:
        idx = torch.tensor([tokenizer.encode(gen_cfg.prompt)], device=device)
        generated = model.generate(idx, gen_cfg.max_tokens, gen_cfg.temperature, gen_cfg.top_k, n_recurrence=n_recurrence)
        print(tokenizer.decode(generated[0].tolist()))
    else:
        # Interactive mode
        print("\nInteractive mode. Type a prompt and press Enter. Ctrl+C to exit.")
        while True:
            try:
                prompt = input("\n>>> ")
                if not prompt:
                    continue
                
                idx = torch.tensor([tokenizer.encode(prompt)], device=device)
                generated = model.generate(idx, gen_cfg.max_tokens, gen_cfg.temperature, gen_cfg.top_k, n_recurrence=n_recurrence)
                print(tokenizer.decode(generated[0].tolist()))
                
            except KeyboardInterrupt:
                print("\nBye!")
                break


# ============================================================================
# Hydra Entry Point (no YAML file needed - uses ConfigStore)
# ============================================================================

@hydra.main(version_base=None, config_name="config")
def main(cfg: Config):
    # Print config
    print(OmegaConf.to_yaml(cfg))
    
    if cfg.command == "train":
        train(cfg)
    elif cfg.command == "generate":
        generate_cmd(cfg)
    else:
        print(f"Unknown command: {cfg.command}")
        print("Use command=train or command=generate")


if __name__ == '__main__':
    main()