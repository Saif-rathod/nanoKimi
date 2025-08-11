# nanoKimi/config.py
import yaml
from dataclasses import dataclass, field
from typing import Any, Dict

@dataclass
class KimiConfig:
    vocab_size: int = 50257
    block_size: int = 1024
    n_layers: int = 12
    n_heads: int = 12
    d_model: int = 768
    dropout: float = 0.1
    latent_dim: int = 128
    moe_experts: int = 0
    moe_top_k: int = 2
    ff_mult: int = 4
    device: str = "cuda"
    use_fp16: bool = False

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return yaml.safe_load(f)
