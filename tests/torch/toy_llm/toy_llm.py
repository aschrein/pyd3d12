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
    
    # Pretrain on synthetic math expressions (do this before RL!)
    python toy_llm.py command=math
    python toy_llm.py command=math math.num_samples=1000000 math.max_num=200
    
    # RL fine-tune on math with GRPO (requires pretrained checkpoint)
    python toy_llm.py command=rl
    python toy_llm.py command=rl rl.operations="['+','-','*','/']" rl.max_num=50
    
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
import random
import re
import copy
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.config_store import ConfigStore

# ============================================================================
# Configuration (Single Source of Truth - no YAML needed)
# ============================================================================

@dataclass
class ModelConfig:
    n_layers     : int      = 8
    n_recurrence : int      = 1
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
class MathDataConfig:
    """Configuration for synthetic math pretraining data."""
    num_samples         : int       = 500000    # Number of math expressions to generate
    min_num             : int       = 0
    max_num             : int       = 100
    operations          : Optional[List[str]] = None  # Default: ['+', '-', '*', '/']
    include_negative    : bool      = False
    include_chains      : bool      = True      # Multi-step expressions like "3 + 4 + 5 = 12"
    max_chain_length    : int       = 3         # Max operations in a chain
    include_parens      : bool      = True      # Parenthetical expressions like "(3 + 4) * 2 = 14"
    difficulty_mix      : bool      = True      # Mix of easy and hard problems
    seed                : int       = 42


@dataclass
class RLConfig:
    """GRPO (Group Relative Policy Optimization) configuration for math RL."""
    # Training
    batch_size          : int       = 32      # Number of math problems per batch
    group_size          : int       = 30       # Completions per problem (G in GRPO)
    max_iters           : int       = 2000     # RL training iterations (reduced - RL is unstable)
    learning_rate       : float     = 3e-5    # Lower LR for stability
    min_lr              : float     = 1e-6    # Minimum LR
    warmup_iters        : int       = 20
    grad_clip           : float     = 0.5     # Tighter gradient clipping
    eval_interval       : int       = 50
    
    # GRPO hyperparameters  
    kl_coef             : float     = 0.01     # KL penalty coefficient (initial) - higher for stability
    kl_target           : float     = 0.05   # Target KL divergence per token
    kl_max              : float     = 1.5    # Max KL before early stopping
    adaptive_kl         : bool      = True    # Adapt KL coefficient dynamically
    entropy_coef        : float     = 0.001   # Entropy bonus for exploration
    advantage_eps       : float     = 1e-8    # Numerical stability for advantage norm
    
    # Generation during RL
    max_answer_tokens   : int       = 16      # Max tokens for math answer (answers are short)
    temperature         : float     = 0.8     # Sampling temperature during RL
    
    # Math problem configuration
    max_num             : int       = 100     # Maximum number in problems
    min_num             : int       = 0       # Minimum number in problems
    operations          : Optional[List[str]] = None  # Default: ['+', '-', '*']
    include_negative    : bool      = False   # Allow negative numbers
    include_decimals    : bool      = False   # Include decimal problems
    difficulty_curriculum: bool     = True    # Start easy, increase difficulty
    
    # Checkpointing
    save_interval       : int       = 50
    rl_checkpoint_path  : str       = ".tmp/toy_llm.pt"


@dataclass 
class Config:
    """Full config combining all sub-configs."""
    model               : ModelConfig = field(default_factory=ModelConfig)
    training            : TrainingConfig = field(default_factory=TrainingConfig)
    data                : DataConfig = field(default_factory=DataConfig)
    checkpoint          : CheckpointConfig = field(default_factory=CheckpointConfig)
    generate            : GenerateConfig = field(default_factory=GenerateConfig)
    math                : MathDataConfig = field(default_factory=MathDataConfig)
    rl                  : RLConfig = field(default_factory=RLConfig)
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


def generate_math_expressions(cfg: 'MathDataConfig') -> str:
    """
    Generate a dataset of math expressions for pretraining.
    
    This teaches the model:
    - Number representation and manipulation
    - Operator semantics (+, -, *, /)
    - Equation format (a op b = result)
    - Order of operations (with parentheses)
    - Multi-step reasoning (chains)
    
    Examples:
        5 + 3 = 8
        12 - 7 = 5
        6 * 4 = 24
        20 / 4 = 5
        3 + 4 + 5 = 12
        (3 + 4) * 2 = 14
        2 * (10 - 3) = 14
    """
    rng = random.Random(cfg.seed)
    operations = cfg.operations or ['+', '-', '*', '/']
    
    expressions = []
    
    def safe_eval(expr: str) -> Optional[int]:
        """Safely evaluate a math expression, return None if result is not a clean integer."""
        try:
            result = eval(expr)
            # Only keep clean integer results
            if isinstance(result, float):
                if result != int(result) or abs(result) > 1e9:
                    return None
                return int(result)
            return result
        except:
            return None
    
    def random_num(max_n: int = None) -> int:
        """Generate a random number within range."""
        max_n = max_n or cfg.max_num
        n = rng.randint(cfg.min_num, max_n)
        if cfg.include_negative and rng.random() < 0.2:
            n = -n
        return n
    
    def generate_simple(difficulty: float = 0.5) -> Optional[Tuple[str, int]]:
        """Generate a simple two-operand expression."""
        max_n = int(cfg.min_num + (cfg.max_num - cfg.min_num) * difficulty)
        max_n = max(max_n, 1)
        
        op = rng.choice(operations)
        
        if op == '/':
            # Ensure clean division
            b = rng.randint(1, max(1, max_n // 3))
            result = rng.randint(0, max(1, max_n // 3))
            a = b * result
        elif op == '*':
            # Keep multiplication reasonable
            a = rng.randint(cfg.min_num, max(1, int(max_n ** 0.5) + 5))
            b = rng.randint(cfg.min_num, max(1, int(max_n ** 0.5) + 5))
            result = a * b
        elif op == '-':
            a = rng.randint(cfg.min_num, max_n)
            b = rng.randint(cfg.min_num, a)  # Keep result non-negative by default
            result = a - b
        else:  # +
            a = rng.randint(cfg.min_num, max_n)
            b = rng.randint(cfg.min_num, max_n)
            result = a + b
        
        expr = f"{a} {op} {b}"
        return expr, result
    
    def generate_chain(difficulty: float = 0.5) -> Optional[Tuple[str, int]]:
        """Generate a chain expression like '3 + 4 + 5' or '2 * 3 + 4'."""
        max_n = int(cfg.min_num + (cfg.max_num - cfg.min_num) * difficulty * 0.5)
        max_n = max(max_n, 5)
        
        # Limit chain ops to + and - for simpler evaluation, occasionally *
        chain_ops = ['+', '-']
        if '*' in operations and rng.random() < 0.3:
            chain_ops.append('*')
        
        chain_len = rng.randint(2, cfg.max_chain_length)
        
        nums = [rng.randint(cfg.min_num, max_n) for _ in range(chain_len + 1)]
        ops = [rng.choice(chain_ops) for _ in range(chain_len)]
        
        # Build expression string
        parts = [str(nums[0])]
        for i, op in enumerate(ops):
            parts.append(f" {op} {nums[i+1]}")
        expr = ''.join(parts)
        
        result = safe_eval(expr)
        if result is None or abs(result) > 10000:
            return None
        
        return expr, result
    
    def generate_paren(difficulty: float = 0.5) -> Optional[Tuple[str, int]]:
        """Generate a parenthetical expression like '(3 + 4) * 2'."""
        max_n = int(cfg.min_num + (cfg.max_num - cfg.min_num) * difficulty * 0.3)
        max_n = max(max_n, 5)
        
        # Inner operation (inside parens)
        inner_op = rng.choice(['+', '-'])
        a = rng.randint(cfg.min_num, max_n)
        b = rng.randint(cfg.min_num, max_n)
        
        # Outer operation
        outer_op = rng.choice(['*', '+', '-'])
        c = rng.randint(1, max(1, max_n // 2))
        
        # Randomly put parens on left or right
        if rng.random() < 0.5:
            expr = f"({a} {inner_op} {b}) {outer_op} {c}"
        else:
            expr = f"{c} {outer_op} ({a} {inner_op} {b})"
        
        result = safe_eval(expr)
        if result is None or abs(result) > 10000:
            return None
        
        return expr, result
    
    def generate_comparison() -> str:
        """Generate comparison/inequality expressions (no solving, just format exposure)."""
        a = random_num()
        b = random_num()
        
        if a > b:
            return f"{a} > {b}"
        elif a < b:
            return f"{a} < {b}"
        else:
            return f"{a} = {b}"
    
    def generate_word_problem() -> Optional[str]:
        """Generate simple word problem format."""
        templates = [
            ("If you have {a} apples and get {b} more, you have {a} + {b} = {r} apples.", '+'),
            ("Starting with {a} coins and spending {b} leaves {a} - {b} = {r} coins.", '-'),
            ("{a} groups of {b} items equals {a} * {b} = {r} items.", '*'),
            ("Dividing {a} cookies among {b} people gives {a} / {b} = {r} each.", '/'),
            ("The sum of {a} and {b} is {a} + {b} = {r}.", '+'),
            ("The difference between {a} and {b} is {a} - {b} = {r}.", '-'),
            ("The product of {a} and {b} is {a} * {b} = {r}.", '*'),
        ]
        
        template, op = rng.choice(templates)
        
        if op == '/':
            b = rng.randint(1, 10)
            r = rng.randint(1, 10)
            a = b * r
        elif op == '*':
            a = rng.randint(1, 12)
            b = rng.randint(1, 12)
            r = a * b
        elif op == '-':
            a = rng.randint(10, 50)
            b = rng.randint(1, a)
            r = a - b
        else:
            a = rng.randint(1, 50)
            b = rng.randint(1, 50)
            r = a + b
        
        return template.format(a=a, b=b, r=r)
    
    # Generate expressions with different types and difficulties
    num_simple = int(cfg.num_samples * 0.5)  # 50% simple
    num_chain = int(cfg.num_samples * 0.2) if cfg.include_chains else 0  # 20% chains
    num_paren = int(cfg.num_samples * 0.15) if cfg.include_parens else 0  # 15% parens
    num_word = int(cfg.num_samples * 0.1)  # 10% word problems
    num_compare = int(cfg.num_samples * 0.05)  # 5% comparisons
    
    print(f"Generating {cfg.num_samples:,} math expressions...")
    print(f"  Simple: {num_simple:,}, Chains: {num_chain:,}, Parens: {num_paren:,}")
    print(f"  Word problems: {num_word:,}, Comparisons: {num_compare:,}")
    
    # Simple expressions
    for i in range(num_simple):
        difficulty = rng.random() if cfg.difficulty_mix else 0.5
        result = generate_simple(difficulty)
        if result:
            expr, answer = result
            expressions.append(f"{expr} = {answer}")
        
        if i % 50000 == 0 and i > 0:
            print(f"  Generated {i:,} simple expressions...")
    
    # Chain expressions
    for i in range(num_chain):
        difficulty = rng.random() if cfg.difficulty_mix else 0.5
        result = generate_chain(difficulty)
        if result:
            expr, answer = result
            expressions.append(f"{expr} = {answer}")
    
    # Parenthetical expressions
    for i in range(num_paren):
        difficulty = rng.random() if cfg.difficulty_mix else 0.5
        result = generate_paren(difficulty)
        if result:
            expr, answer = result
            expressions.append(f"{expr} = {answer}")
    
    # Word problems
    for i in range(num_word):
        problem = generate_word_problem()
        if problem:
            expressions.append(problem)
    
    # Comparisons
    for i in range(num_compare):
        expressions.append(generate_comparison())
    
    # Shuffle everything
    rng.shuffle(expressions)
    
    # Join with newlines
    text = '\n'.join(expressions)
    
    print(f"Generated {len(expressions):,} math expressions, {len(text):,} characters")
    
    # Print examples
    print("\nExamples:")
    for expr in expressions[:15]:
        print(f"  {expr}")
    
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
    
    def forward_with_log_probs(self, idx: torch.Tensor, n_recurrence: int = 6) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that returns logits and per-token log probabilities.
        Used for GRPO to compute policy probabilities.
        
        Returns:
            logits: (B, T, V) raw logits
            log_probs: (B, T) log probability of each token given previous tokens
        """
        logits, _ = self.forward(idx, n_recurrence=n_recurrence)
        
        # Compute log probs for each token (shifted by 1 for autoregressive)
        # logits[:, :-1] predicts idx[:, 1:]
        log_probs_all = F.log_softmax(logits[:, :-1, :], dim=-1)  # (B, T-1, V)
        
        # Gather the log prob of the actual next tokens
        next_tokens = idx[:, 1:]  # (B, T-1)
        log_probs = log_probs_all.gather(dim=-1, index=next_tokens.unsqueeze(-1)).squeeze(-1)  # (B, T-1)
        
        return logits, log_probs
    
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
    
    @torch.no_grad()
    def generate_with_log_probs(
        self, 
        idx: torch.Tensor, 
        max_new_tokens: int, 
        temperature: float = 1.0,
        n_recurrence: int = 6,
        stop_token: int = ord('\n')
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Generate tokens and track log probabilities for RL.
        Uses eval mode for consistent log probs.
        
        Args:
            idx: Starting tokens (B, T)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            n_recurrence: Recurrence depth
            stop_token: Token id that stops generation (e.g., newline)
        
        Returns:
            generated: Full sequence including prompt, padded to prompt_len + max_new_tokens
            log_probs: Log prob of each generated token (B, max_new_tokens)
            masks: Binary mask indicating valid (non-padding) tokens (B, max_new_tokens)
            prompt_len: Length of the prompt (for alignment)
        """
        was_training = self.training
        self.eval()  # Eval mode for consistent log probs (no dropout variation)
        
        B = idx.size(0)
        device = idx.device
        prompt_len = idx.size(1)
        
        all_log_probs = []
        all_masks = []
        all_tokens = []
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        
        for step in range(max_new_tokens):
            # Crop context if too long
            idx_cond = idx if idx.size(1) <= self.model_cfg.max_seq_len else idx[:, -self.model_cfg.max_seq_len:]
            
            logits, _ = self(idx_cond, n_recurrence=n_recurrence)
            logits = logits[:, -1, :] / temperature
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            log_probs_all = F.log_softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            
            # Get log prob of sampled token
            token_log_prob = log_probs_all.gather(dim=-1, index=next_token).squeeze(-1)  # (B,)
            
            # Track which sequences are still active (before updating finished)
            mask = ~finished
            all_log_probs.append(token_log_prob * mask.float())  # Zero out log probs for finished sequences
            all_masks.append(mask.float())
            all_tokens.append(next_token.squeeze(-1))  # (B,)
            
            # Append token to sequence
            idx = torch.cat([idx, next_token], dim=1)
            
            # Update finished status (after appending, so we include the stop token)
            finished = finished | (next_token.squeeze(-1) == stop_token)
            
            if finished.all():
                break
        
        # Pad to max_new_tokens for consistent tensor shapes
        generated_len = len(all_log_probs)
        pad_len = max_new_tokens - generated_len
        
        if pad_len > 0:
            # Pad with zeros
            pad_token = torch.zeros(B, dtype=torch.long, device=device)  # Use 0 (null) as pad
            for _ in range(pad_len):
                all_log_probs.append(torch.zeros(B, device=device))
                all_masks.append(torch.zeros(B, device=device))
                all_tokens.append(pad_token)
        
        # Stack results
        log_probs = torch.stack(all_log_probs, dim=1)  # (B, max_new_tokens)
        masks = torch.stack(all_masks, dim=1)  # (B, max_new_tokens)
        gen_tokens = torch.stack(all_tokens, dim=1)  # (B, max_new_tokens)
        
        # Build padded sequences: prompt + generated tokens (padded)
        # This ensures all sequences have the same length: prompt_len + max_new_tokens
        sequences = torch.cat([idx[:, :prompt_len], gen_tokens], dim=1)  # (B, prompt_len + max_new_tokens)
        
        if was_training:
            self.train()
        
        return sequences, log_probs, masks, prompt_len


# ============================================================================
# Math Problem Generation and Reward
# ============================================================================

class MathProblemGenerator:
    """
    Generates arithmetic problems for RL training.
    
    Format: "a op b = " -> model should complete with the answer
    """
    
    def __init__(
        self,
        min_num: int = 0,
        max_num: int = 100,
        operations: List[str] = None,
        include_negative: bool = False,
        include_decimals: bool = False,
        seed: int = None
    ):
        self.min_num = min_num
        self.max_num = max_num
        self.operations = operations or ['+', '-', '*']
        self.include_negative = include_negative
        self.include_decimals = include_decimals
        self.rng = random.Random(seed)
    
    def set_difficulty(self, level: float):
        """
        Adjust difficulty from 0.0 (easy) to 1.0 (hard).
        Affects number range and operation complexity.
        """
        # Scale max_num based on difficulty
        base_max = 10
        full_max = self.max_num
        self.current_max = int(base_max + (full_max - base_max) * level)
        
        # Add harder operations at higher difficulty
        if level < 0.3:
            self.current_ops = ['+']
        elif level < 0.5:
            self.current_ops = ['+', '-']
        elif level < 0.7:
            self.current_ops = ['+', '-', '*']
        else:
            self.current_ops = self.operations
    
    def generate_one(self) -> Tuple[str, str, float]:
        """
        Generate a single math problem.
        
        Returns:
            prompt: The problem string (e.g., "12 + 34 = ")
            answer: The correct answer as string
            difficulty: Relative difficulty score
        """
        max_n = getattr(self, 'current_max', self.max_num)
        ops = getattr(self, 'current_ops', self.operations)
        
        op = self.rng.choice(ops)
        
        if op == '/':
            # For division, ensure clean integer result
            b = self.rng.randint(1, max(1, max_n // 5))
            result = self.rng.randint(self.min_num, max(1, max_n // 5))
            a = b * result
        else:
            a = self.rng.randint(self.min_num, max_n)
            b = self.rng.randint(self.min_num, max_n)
            
            if op == '+':
                result = a + b
            elif op == '-':
                result = a - b
            elif op == '*':
                result = a * b
        
        if self.include_negative:
            if self.rng.random() < 0.3:
                a = -a
            if self.rng.random() < 0.3:
                b = -b
        
        prompt = f"{a} {op} {b} = "
        answer = str(result)
        
        # Difficulty based on magnitude
        difficulty = (abs(a) + abs(b)) / (2 * max_n)
        
        return prompt, answer, difficulty
    
    def generate_batch(self, n: int) -> List[Tuple[str, str, float]]:
        """Generate a batch of math problems."""
        return [self.generate_one() for _ in range(n)]


def compute_math_reward(
    generated_text: str,
    correct_answer: str,
    prompt: str
) -> Tuple[float, dict]:
    """
    Compute reward for a math completion.
    
    Args:
        generated_text: Full generated sequence including prompt
        correct_answer: The expected answer
        prompt: The original prompt
    
    Returns:
        reward: Float reward value
        info: Dict with debugging info
    """
    # Extract the answer part (after the prompt)
    if prompt in generated_text:
        response = generated_text[len(prompt):]
    else:
        response = generated_text
    
    # Clean up: take everything before newline
    response = response.split('\n')[0]
    
    # Try to extract a number from the response
    # Match optional negative sign followed by digits
    match = re.match(r'^(-?\d+)', response.strip())
    
    info = {
        'response': response,
        'predicted': None,
        'expected': int(correct_answer),
        'correct': False,
        'clean': False
    }
    
    if match is None:
        # Couldn't parse any number
        return -1.0, info
    
    try:
        predicted = int(match.group(1))
        expected = int(correct_answer)
    except (ValueError, AttributeError):
        return -1.0, info
    
    info['predicted'] = predicted
    
    # Check if the response is "clean" (just the number, optionally followed by whitespace)
    # Clean examples: "42", "42 ", "42\n"
    # Dirty examples: "42FFXXX", "42abc", "42 garbage"
    response_after_number = response.strip()[len(match.group(1)):]
    is_clean = len(response_after_number.strip()) == 0
    info['clean'] = is_clean
    
    if predicted == expected:
        info['correct'] = True
        if is_clean:
            # Correct and clean - full reward
            return 1.0, info
        else:
            # Correct but has garbage after - partial reward
            # Penalize based on amount of garbage
            garbage_len = len(response_after_number.strip())
            penalty = min(0.5, garbage_len * 0.1)  # Up to 0.5 penalty
            return 1.0 - penalty, info
    
    # Wrong answer
    if is_clean:
        # Wrong but clean format - small penalty
        return -0.5, info
    else:
        # Wrong and garbage - larger penalty
        return -1.0, info


# ============================================================================
# GRPO (Group Relative Policy Optimization)
# ============================================================================

class GRPO:
    """
    Group Relative Policy Optimization for RL fine-tuning.
    
    Key idea: Sample G completions per prompt, compute rewards,
    then use relative advantages within each group for policy update.
    No critic needed - uses Monte Carlo returns.
    """
    
    def __init__(
        self,
        model: ToyLLM,
        tokenizer: ASCIITokenizer,
        cfg: RLConfig,
        device: str = "cuda"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.device = device
        
        # Reference model for KL penalty (frozen copy)
        self.ref_model = copy.deepcopy(model)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        # Adaptive KL coefficient
        self.kl_coef = cfg.kl_coef
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.01
        )
        
        # Problem generator
        operations = list(cfg.operations) if cfg.operations else ['+', '-', '*']
        self.problem_gen = MathProblemGenerator(
            min_num=cfg.min_num,
            max_num=cfg.max_num,
            operations=operations,
            include_negative=cfg.include_negative,
            include_decimals=cfg.include_decimals
        )
        
        # Best model tracking
        self.best_score = -float('inf')
        self.best_state_dict = copy.deepcopy(model.state_dict())
        
        # Metrics tracking
        self.metrics = {
            'rewards': [],
            'accuracy': [],
            'kl': [],
            'loss': []
        }
    
    def get_lr(self, iteration: int) -> float:
        """Cosine decay with warmup."""
        if iteration < self.cfg.warmup_iters:
            return self.cfg.learning_rate * (iteration + 1) / self.cfg.warmup_iters
        
        decay_ratio = (iteration - self.cfg.warmup_iters) / max(1, self.cfg.max_iters - self.cfg.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * min(1.0, decay_ratio)))
        return self.cfg.min_lr + coeff * (self.cfg.learning_rate - self.cfg.min_lr)
    
    def collect_rollouts(
        self,
        prompts: List[str],
        answers: List[str],
        n_recurrence: int
    ) -> dict:
        """
        Collect G completions per prompt and compute rewards.
        
        Returns dict with:
            - sequences: (B*G, max_len) token sequences
            - old_log_probs: (B*G, gen_len) per-token log probs from rollout
            - rewards: (B*G,) scalar rewards
            - masks: (B*G, gen_len) valid token masks
            - prompt_len: int, length of (padded) prompts
        """
        B = len(prompts)
        G = self.cfg.group_size
        
        # Expand prompts G times each
        expanded_prompts = []
        expanded_answers = []
        for p, a in zip(prompts, answers):
            for _ in range(G):
                expanded_prompts.append(p)
                expanded_answers.append(a)
        
        # Tokenize prompts
        prompt_tokens = [self.tokenizer.encode(p) for p in expanded_prompts]
        max_prompt_len = max(len(t) for t in prompt_tokens)
        
        # Pad prompts (left pad with space)
        padded_prompts = []
        for tokens in prompt_tokens:
            padding = [ord(' ')] * (max_prompt_len - len(tokens))
            padded_prompts.append(padding + tokens)
        
        prompt_tensor = torch.tensor(padded_prompts, dtype=torch.long, device=self.device)
        
        # Generate completions with log probs (in eval mode for consistency)
        sequences, old_log_probs, masks, prompt_len = self.model.generate_with_log_probs(
            prompt_tensor,
            max_new_tokens=self.cfg.max_answer_tokens,
            temperature=self.cfg.temperature,
            n_recurrence=n_recurrence,
            stop_token=ord('\n')
        )
        
        # Compute rewards
        rewards = []
        infos = []
        for i in range(B * G):
            seq_tokens = sequences[i].tolist()
            generated_text = self.tokenizer.decode(seq_tokens)
            reward, info = compute_math_reward(
                generated_text,
                expanded_answers[i],
                expanded_prompts[i]
            )
            rewards.append(reward)
            infos.append(info)
        
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        
        return {
            'sequences': sequences,
            'old_log_probs': old_log_probs,
            'rewards': rewards,
            'masks': masks,
            'prompt_len': prompt_len,
            'prompts': expanded_prompts,
            'answers': expanded_answers,
            'infos': infos,
            'B': B,
            'G': G
        }
    
    def compute_grpo_loss(self, rollouts: dict, n_recurrence: int) -> Tuple[torch.Tensor, dict]:
        """
        Compute GRPO loss with group-normalized advantages.
        
        GRPO key idea: Use group-relative advantages instead of a learned baseline.
        For each prompt, we sample G completions and normalize rewards within the group.
        
        Loss = -E[A * log π(a|s)] + kl_coef * KL(π || π_ref)
        """
        B, G = rollouts['B'], rollouts['G']
        rewards = rollouts['rewards']  # (B*G,)
        old_log_probs = rollouts['old_log_probs']  # (B*G, T) - from generation, no grad
        masks = rollouts['masks']  # (B*G, T)
        sequences = rollouts['sequences']  # (B*G, seq_len)
        prompt_len = rollouts['prompt_len']
        
        # ===== 1. Compute group-relative advantages =====
        # Reshape rewards to (B, G) for group normalization
        rewards_grouped = rewards.view(B, G)
        
        # Normalize within each group: A_i = (r_i - mean) / (std + eps)
        group_mean = rewards_grouped.mean(dim=1, keepdim=True)
        group_std = rewards_grouped.std(dim=1, keepdim=True)
        advantages = (rewards_grouped - group_mean) / (group_std + self.cfg.advantage_eps)
        advantages = advantages.view(B * G)  # Flatten back to (B*G,)
        
        # ===== 2. Compute current policy log probs (with gradients) =====
        self.model.train()
        
        # Forward pass on full sequences
        logits, _ = self.model(sequences, n_recurrence=n_recurrence)  # (B*G, seq_len, vocab)
        
        # Get log probs for the generated tokens (after prompt)
        # logits[:, t, :] predicts token at position t+1
        # So for generated tokens starting at prompt_len, we use logits[:, prompt_len-1:-1, :]
        gen_len = old_log_probs.size(1)
        
        # Slice logits for positions that predict generated tokens
        # Position prompt_len-1 predicts token at prompt_len (first generated token)
        pred_logits = logits[:, prompt_len-1:prompt_len-1+gen_len, :]  # (B*G, gen_len, vocab)
        
        # Get the actual generated tokens
        gen_tokens = sequences[:, prompt_len:prompt_len+gen_len]  # (B*G, gen_len)
        
        # Compute log probs
        log_probs_all = F.log_softmax(pred_logits, dim=-1)  # (B*G, gen_len, vocab)
        current_log_probs = log_probs_all.gather(dim=-1, index=gen_tokens.unsqueeze(-1)).squeeze(-1)  # (B*G, gen_len)
        
        # ===== 3. Compute reference model log probs (for KL penalty) =====
        with torch.no_grad():
            self.ref_model.eval()
            ref_logits, _ = self.ref_model(sequences, n_recurrence=n_recurrence)
            ref_pred_logits = ref_logits[:, prompt_len-1:prompt_len-1+gen_len, :]
            ref_log_probs_all = F.log_softmax(ref_pred_logits, dim=-1)
            ref_log_probs = ref_log_probs_all.gather(dim=-1, index=gen_tokens.unsqueeze(-1)).squeeze(-1)
        
        # ===== 4. Compute policy gradient loss =====
        # REINFORCE: -A * log π(a|s), summed over tokens
        # Expand advantages to match token dimension
        advantages_expanded = advantages.unsqueeze(1).expand_as(current_log_probs)  # (B*G, gen_len)
        
        # Policy loss (negative because we maximize)
        policy_loss = -advantages_expanded * current_log_probs  # (B*G, gen_len)
        
        # ===== 5. KL penalty =====
        # KL(π || π_ref) ≈ log π - log π_ref (approximation)
        kl_div = current_log_probs - ref_log_probs  # (B*G, gen_len)
        kl_penalty = self.kl_coef * kl_div  # Use adaptive kl_coef
        
        # ===== 6. Entropy bonus (encourage exploration) =====
        # Entropy = -sum(p * log p), approximate with -log p for sampled actions
        entropy = -current_log_probs
        entropy_bonus = -self.cfg.entropy_coef * entropy  # Negative because we add it to loss
        
        # ===== 7. Combine with masking =====
        total_loss = (policy_loss + kl_penalty + entropy_bonus) * masks
        
        # Mean over valid tokens
        n_valid = masks.sum() + 1e-8
        loss = total_loss.sum() / n_valid
        
        # ===== 8. Compute metrics =====
        with torch.no_grad():
            n_correct = sum(1 for info in rollouts['infos'] if info['correct'])
            n_clean = sum(1 for info in rollouts['infos'] if info.get('clean', False))
            n_total = len(rollouts['infos'])
            accuracy = n_correct / n_total
            clean_rate = n_clean / n_total
            mean_reward = rewards.mean().item()
            mean_kl = (kl_div * masks).sum().item() / n_valid.item()
            mean_entropy = (entropy * masks).sum().item() / n_valid.item()
            
            # Log prob ratio (for debugging)
            log_ratio = current_log_probs - old_log_probs
            mean_ratio = torch.exp(log_ratio * masks).sum().item() / n_valid.item()
        
        metrics = {
            'loss': loss.item(),
            'policy_loss': (policy_loss * masks).sum().item() / n_valid.item(),
            'kl': mean_kl,
            'kl_coef': self.kl_coef,
            'entropy': mean_entropy,
            'reward': mean_reward,
            'accuracy': accuracy,
            'clean_rate': clean_rate,
            'advantage_std': advantages.std().item(),
            'ratio': mean_ratio,
        }
        
        return loss, metrics
    
    def train_step(self, iteration: int, n_recurrence: int) -> Tuple[dict, bool]:
        """
        Single GRPO training step.
        
        Returns:
            metrics: Dict of training metrics
            early_stop: True if KL divergence exceeded maximum threshold
        """
        # Update learning rate
        lr = self.get_lr(iteration)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        # Curriculum: increase difficulty over time
        if self.cfg.difficulty_curriculum:
            difficulty = min(1.0, iteration / (self.cfg.max_iters * 0.7))
            self.problem_gen.set_difficulty(difficulty)
        
        # Generate problems
        problems = self.problem_gen.generate_batch(self.cfg.batch_size)
        prompts = [p[0] for p in problems]
        answers = [p[1] for p in problems]
        
        # Collect rollouts
        rollouts = self.collect_rollouts(prompts, answers, n_recurrence)
        
        # Compute loss and update
        loss, metrics = self.compute_grpo_loss(rollouts, n_recurrence)
        
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        if self.cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        
        self.optimizer.step()
        
        # Adaptive KL coefficient adjustment
        if self.cfg.adaptive_kl:
            kl = metrics['kl']
            if kl > self.cfg.kl_target * 1.5:
                # KL too high - increase penalty
                self.kl_coef = min(self.kl_coef * 1.5, 1.0)
            elif kl < self.cfg.kl_target * 0.5:
                # KL too low - decrease penalty (allow more exploration)
                self.kl_coef = max(self.kl_coef * 0.8, 0.001)
        
        # Track best model
        score = metrics['accuracy'] * (0.5 + 0.5 * metrics['clean_rate'])
        if score > self.best_score:
            self.best_score = score
            self.best_state_dict = copy.deepcopy(self.model.state_dict())
        
        # Check for early stopping
        early_stop = metrics['kl'] > self.cfg.kl_max
        
        metrics['lr'] = lr
        return metrics, early_stop
    
    def restore_best(self):
        """Restore the best model found during training."""
        self.model.load_state_dict(self.best_state_dict)
        print(f"Restored best model (score={self.best_score:.3f})")


def math_pretrain(cfg: Config):
    """
    Pretrain model on synthetic math expressions.
    
    This creates a foundation for RL fine-tuning by teaching:
    - Number representation
    - Operator semantics  
    - Equation format
    - Basic arithmetic patterns
    """
    model_cfg = cfg.model
    train_cfg = cfg.training
    math_cfg = cfg.math
    ckpt_cfg = cfg.checkpoint
    device = cfg.device
    
    print("=" * 60)
    print("Math Pretraining")
    print("=" * 60)
    
    # Auto-checkpoint path
    auto_checkpoint = Path(ckpt_cfg.path)
    if not auto_checkpoint.is_absolute():
        auto_checkpoint = Path(hydra.utils.get_original_cwd()) / ckpt_cfg.path
    auto_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    
    # Fresh start - delete existing checkpoint
    if ckpt_cfg.fresh and auto_checkpoint.exists():
        print(f"fresh=true specified, removing existing checkpoint {auto_checkpoint}")
        auto_checkpoint.unlink()
    
    # Generate math data
    # Convert OmegaConf to proper MathDataConfig
    math_config = MathDataConfig(**OmegaConf.to_container(math_cfg))
    math_text = generate_math_expressions(math_config)
    
    # Tokenize
    tokenizer = ASCIITokenizer()
    data = torch.tensor(tokenizer.encode(math_text), dtype=torch.long)
    
    # Split train/val
    split_idx = int(len(data) * train_cfg.train_split)
    train_data, val_data = data[:split_idx], data[split_idx:]
    
    train_dataset = CodeDataset(train_data, model_cfg.max_seq_len)
    val_dataset = CodeDataset(val_data, model_cfg.max_seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=train_cfg.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=train_cfg.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"\nData: {len(math_text):,} chars, Train: {len(train_dataset):,} samples, Val: {len(val_dataset):,} samples")
    
    # Create or load model
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
    start_iteration = 0
    start_epoch = 0
    best_val_loss = float('inf')
    
    if auto_checkpoint.exists():
        print(f"Loading checkpoint from {auto_checkpoint}...")
        ckpt = torch.load(auto_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'], strict=False)
        # start_iteration = ckpt.get('iteration', 0)
        # start_epoch = ckpt.get('epoch', 0)
        best_val_loss = ckpt.get('val_loss', float('inf'))
        print(f"Resumed from iteration {start_iteration}, epoch {start_epoch}, best_val_loss {best_val_loss:.4f}")
    
    if train_cfg.compile_model and hasattr(torch, 'compile'):
        print("Compiling model...")
        model = torch.compile(model)
    
    # Training loop
    iteration = start_iteration
    epoch = start_epoch
    train_config = TrainingConfig(**OmegaConf.to_container(train_cfg))
    
    sample_prompts = ["5 + 3 = ", "12 - 7 = ", "6 * 4 = ", "15 + 27 = ", "(3 + 4) * 2 = "]
    
    print(f"\nStarting math pretraining...")
    print(f"Iterations per epoch: {len(train_loader)}")
    
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
                
                # Generate samples
                print("-" * 60)
                model.eval()
                for prompt in sample_prompts:
                    idx = torch.tensor([tokenizer.encode(prompt)], device=device)
                    generated = model.generate(idx, max_new_tokens=10, temperature=0.5, n_recurrence=n_recurrence)
                    output = tokenizer.decode(generated[0].tolist()).split('\n')[0]  # First line only
                    print(f"  {output}")
                model.train()
                print("-" * 60)
            
            iteration += 1
        
        # End of epoch
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
    
    print(f"\n>>> Math pretraining complete! Best val loss: {best_val_loss:.4f}")
    print(f">>> Checkpoint saved to {auto_checkpoint}")
    print(f">>> Now run: python toy_llm.py command=rl")
    
    return model, tokenizer


def rl_train(cfg: Config):
    """RL fine-tuning with GRPO on math problems."""
    rl_cfg = cfg.rl
    model_cfg = cfg.model
    ckpt_cfg = cfg.checkpoint
    device = cfg.device
    
    print("=" * 60)
    print("GRPO RL Fine-tuning on Math")
    print("=" * 60)
    
    # Load pretrained model
    ckpt_path = Path(ckpt_cfg.path)
    if not ckpt_path.is_absolute():
        ckpt_path = Path(hydra.utils.get_original_cwd()) / ckpt_path
    
    if not ckpt_path.exists():
        raise ValueError(f"No checkpoint found at {ckpt_path}. Train a model first with command=train")
    
    print(f"Loading pretrained model from {ckpt_path}...")
    model, tokenizer, device, n_recurrence = load_model(str(ckpt_path), device)
    model.train()
    
    # Setup GRPO trainer
    # Convert OmegaConf to proper RLConfig
    rl_config = RLConfig(**OmegaConf.to_container(rl_cfg))
    grpo = GRPO(model, tokenizer, rl_config, device)
    
    # RL checkpoint path
    rl_ckpt_path = Path(rl_config.rl_checkpoint_path)
    if not rl_ckpt_path.is_absolute():
        rl_ckpt_path = Path(hydra.utils.get_original_cwd()) / rl_ckpt_path
    rl_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nRL Config:")
    print(f"  Batch size: {rl_config.batch_size}")
    print(f"  Group size: {rl_config.group_size}")
    print(f"  Max iterations: {rl_config.max_iters}")
    print(f"  Operations: {rl_config.operations or ['+', '-', '*']}")
    print(f"  Number range: {rl_config.min_num} to {rl_config.max_num}")
    print()
    
    # Training loop
    best_accuracy = 0.0
    running_reward = 0.0
    running_accuracy = 0.0
    running_clean = 0.0
    
    for iteration in range(rl_config.max_iters):
        t0 = time.time()
        
        metrics, early_stop = grpo.train_step(iteration, n_recurrence)
        
        # Update running stats
        alpha = 0.05  # Slower smoothing for more stable estimates
        running_reward = alpha * metrics['reward'] + (1 - alpha) * running_reward
        running_accuracy = alpha * metrics['accuracy'] + (1 - alpha) * running_accuracy
        running_clean = alpha * metrics['clean_rate'] + (1 - alpha) * running_clean
        
        dt = time.time() - t0
        
        # Logging
        if iteration % 10 == 0:
            print(f"iter {iteration:5d} | reward {metrics['reward']:+.3f} ({running_reward:+.3f}) | "
                  f"acc {metrics['accuracy']:.0%} clean {metrics['clean_rate']:.0%} | "
                  f"kl {metrics['kl']:.4f} (β={metrics['kl_coef']:.3f}) | "
                  f"lr {metrics['lr']:.2e} | {dt*1000:.0f}ms")
        
        # Early stopping check
        if early_stop:
            print(f"\n!!! KL divergence {metrics['kl']:.4f} exceeded max {rl_config.kl_max:.4f} - stopping early !!!")
            print("Restoring best model...")
            grpo.restore_best()
            break
        
        # Evaluation and examples
        if iteration > 0 and iteration % rl_config.eval_interval == 0:
            print("\n" + "=" * 60)
            print(f"Evaluation at iteration {iteration}")
            print("=" * 60)
            
            # Show metrics
            print(f"Running: reward={running_reward:+.3f}, accuracy={running_accuracy:.1%}, clean={running_clean:.1%}")
            print(f"Batch:   kl={metrics['kl']:.4f}, kl_coef={metrics['kl_coef']:.3f}, entropy={metrics['entropy']:.3f}")
            
            # Generate random test problems using current difficulty
            n_eval = 10
            eval_problems = grpo.problem_gen.generate_batch(n_eval)
            
            print("\nSample generations (greedy):")
            model.eval()
            correct = 0
            clean = 0
            for prompt, expected_str, _ in eval_problems:
                idx = torch.tensor([tokenizer.encode(prompt)], device=device)
                generated = model.generate(idx, max_new_tokens=10, temperature=0.1, n_recurrence=n_recurrence)
                output = tokenizer.decode(generated[0].tolist()).split('\n')[0]
                
                # Check correctness
                try:
                    expected = int(expected_str)
                    response = output[len(prompt):]
                    match = re.match(r'^(-?\d+)', response.strip())
                    predicted = int(match.group(1)) if match else None
                    is_correct = predicted == expected
                    
                    # Check if clean (no garbage after number)
                    if match:
                        after_num = response.strip()[len(match.group(1)):]
                        is_clean = len(after_num.strip()) == 0
                    else:
                        is_clean = False
                    
                    correct += is_correct
                    clean += is_clean
                    
                    if is_correct and is_clean:
                        mark = "✓"
                    elif is_correct:
                        mark = "~"  # Correct but dirty
                    else:
                        mark = "✗"
                    
                    # Show clean output or truncated dirty output
                    display = output if is_clean else output[:40] + "..." if len(output) > 40 else output
                    print(f"  {mark} {display} (expected: {expected})")
                except:
                    print(f"  ? {output[:40]}...")
            
            eval_acc = correct / n_eval
            clean_rate = clean / n_eval
            print(f"\nEval accuracy: {eval_acc:.1%} ({correct}/{n_eval}), Clean: {clean_rate:.1%} ({clean}/{n_eval})")
            model.train()
            print("=" * 60 + "\n")
        
        # Save checkpoint
        if iteration > 0 and iteration % rl_config.save_interval == 0:
            # Use combined score: accuracy weighted by clean rate
            current_score = running_accuracy * (0.5 + 0.5 * running_clean)
            if current_score > best_accuracy:
                best_accuracy = current_score
                torch.save({
                    'model': grpo.best_state_dict,  # Save the best model, not current
                    'config': OmegaConf.to_container(cfg),
                    'iteration': iteration,
                    'accuracy': running_accuracy,
                    'clean_rate': running_clean,
                    'reward': running_reward
                }, rl_ckpt_path)
                print(f">>> Saved best RL checkpoint (acc={running_accuracy:.1%}, clean={running_clean:.1%}) to {rl_ckpt_path}")
    
    # Restore and save the best model
    grpo.restore_best()
    
    torch.save({
        'model': grpo.best_state_dict,
        'config': OmegaConf.to_container(cfg),
        'iteration': iteration,
        'accuracy': running_accuracy,
        'clean_rate': running_clean,
        'reward': running_reward,
        'best_score': grpo.best_score
    }, rl_ckpt_path)
    print(f"\n>>> RL training complete! Best score: {grpo.best_score:.3f}")
    print(f">>> Saved to {rl_ckpt_path}")
    
    return model, tokenizer


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
    elif cfg.command == "math":
        math_pretrain(cfg)
    elif cfg.command == "rl":
        rl_train(cfg)
    else:
        print(f"Unknown command: {cfg.command}")
        print("Use command=train, command=generate, command=math, or command=rl")


if __name__ == '__main__':
    main()