import torch
import torch.nn as nn
import math

class LatentSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        l = cfg.latent_dim
        self.to_q = nn.Linear(d, l, bias=False)
        self.to_k = nn.Linear(d, l, bias=False)
        self.to_v = nn.Linear(d, l, bias=False)
        self.to_out = nn.Linear(l, d, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        self.l = l

    def forward(self, x, attn_mask=None):
        B, T, D = x.shape
        q = self.to_q(x)  # B,T,L
        k = self.to_k(x)
        v = self.to_v(x)

        scores = torch.einsum("btl,bsl->bts", q, k) / math.sqrt(self.l)
        if attn_mask is None:
            mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
        else:
            mask = attn_mask.bool()
        scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out_latent = torch.einsum("bts,bsl->btl", attn, v)
        out = self.to_out(out_latent)
        return out
