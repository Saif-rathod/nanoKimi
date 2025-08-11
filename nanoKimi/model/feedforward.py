import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        hid = cfg.ff_mult * d
        self.w1 = nn.Linear(d, hid * 2)
        self.w2 = nn.Linear(hid, d)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x_proj = self.w1(x)
        a, b = x_proj.chunk(2, dim=-1)
        a = F.silu(a)
        out = self.w2(a * b)
        return self.dropout(out)
