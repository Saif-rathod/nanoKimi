import torch.nn as nn
import torch
from .embedding import EmbeddingLayer
from .block import KimiBlock

class KimiModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = EmbeddingLayer(cfg)
        self.blocks = nn.ModuleList([KimiBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, idx, attn_mask=None):
        b, t = idx.shape
        x = self.embed(idx)
        if attn_mask is None:
            attn_mask = torch.tril(torch.ones(t, t, device=idx.device)).bool()
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100, temperature=1.0, top_k=40):
        device = next(self.parameters()).device
        cur = input_ids.to(device)
        for _ in range(max_new_tokens):
            if cur.shape[1] > self.cfg.block_size:
                cur = cur[:, -self.cfg.block_size:]
            logits = self.forward(cur)
            logits = logits[:, -1, :] / max(1e-8, temperature)
            if top_k is not None:
                vals, idxs = logits.topk(top_k, dim=-1)
                probs = torch.zeros_like(logits).scatter_(-1, idxs, torch.softmax(vals, dim=-1))
            else:
                probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            cur = torch.cat([cur, next_id], dim=1)
        return cur
