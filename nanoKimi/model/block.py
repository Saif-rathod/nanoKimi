# nanoKimi/model/block.py
import torch.nn as nn
from .attention import LatentSelfAttention
from .feedforward import FeedForward
from .moe import MoELayer

class KimiBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = LatentSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.use_moe = (cfg.moe_experts > 0)
        if self.use_moe:
            self.moe = MoELayer(cfg)
            self.ffn = None
        else:
            self.moe = None
            self.ffn = FeedForward(cfg)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, attn_mask=None):
        residual = x
        x = self.ln1(x)
        x = self.attn(x, attn_mask=attn_mask)
        x = residual + self.dropout(x)

        residual = x
        x = self.ln2(x)
        if self.use_moe:
            x = self.moe(x)
        else:
            x = self.ffn(x)
        x = residual + self.dropout(x)
        return x
