# nanoKimi/benchmark.py
import time
import torch
from .config import load_yaml
from .model.kimi import KimiModel
from .dataset import load_text
from .dataset import TextDataset
from torch.utils.data import DataLoader

def measure_train_speed(model_cfg, train_cfg, dataset_path, steps=50):
    import math
    cfg = model_cfg
    model = KimiModel(cfg).to(train_cfg.get("device", "cuda"))
    text = load_text(dataset_path)
    from .tokenizer import Tokenizer
    tokenizer = Tokenizer()
    tokens = tokenizer.encode(text)
    ds = TextDataset(tokens, cfg.block_size)
    dl = DataLoader(ds, batch_size=train_cfg.get("batch_size", 8), shuffle=True, drop_last=True)
    it = iter(dl)
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.get("lr", 1e-3))
    t0 = time.time()
    tokens_processed = 0
    for i in range(steps):
        try:
            xb, yb = next(it)
        except StopIteration:
            it = iter(dl)
            xb, yb = next(it)
        xb = xb.to(train_cfg.get("device", "cuda"))
        yb = yb.to(train_cfg.get("device", "cuda"))
        logits = model(xb)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        tokens_processed += xb.numel()
    t1 = time.time()
    duration = t1 - t0
    tps = tokens_processed / duration
    print(f"Tokens processed: {tokens_processed}, time: {duration:.2f}s, tokens/sec: {tps:.2f}")
    if torch.cuda.is_available():
        print("Peak GPU mem (GB):", torch.cuda.max_memory_allocated()/(1024**3))
