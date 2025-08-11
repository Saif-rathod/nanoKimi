# nanoKimi/model/moe.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, d_model, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x):
        return self.net(x)

class MoELayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.E = cfg.moe_experts
        self.k = max(1, cfg.moe_top_k)
        hidden = cfg.d_model * cfg.ff_mult
        self.experts = nn.ModuleList([Expert(cfg.d_model, hidden) for _ in range(self.E)])
        self.gate = nn.Linear(cfg.d_model, self.E)

    def forward(self, x):
        # x: (B,T,D)
        B, T, D = x.shape
        logits = self.gate(x)  # B,T,E
        topk_vals, topk_idx = torch.topk(logits, self.k, dim=-1)  # B,T,k
        topk_w = F.softmax(topk_vals, dim=-1)  # B,T,k
        out = x.new_zeros(B, T, D)
        for e in range(self.E):
            mask = (topk_idx == e)  # B,T,k
            if not mask.any():
                continue
            sel_weights = (mask.float() * topk_w.unsqueeze(-1)).sum(dim=-1)  # B,T
            sel_weights = sel_weights.unsqueeze(-1)
            expert_out = self.experts[e](x)  # B,T,D
            out = out + sel_weights * expert_out
        return out
