# nanoKimi/trainer.py
import time
import os
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .config import KimiConfig, load_yaml
from .dataset import TextDataset, load_text
from .tokenizer import Tokenizer
from .model.kimi import KimiModel
from .model.optimizer import Muon

def set_seed(seed=42):
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class Trainer:
    def __init__(self, model_cfg, train_cfg):
        self.cfg = train_cfg
        self.model_cfg = model_cfg
        self.device = torch.device(train_cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
        self.model = KimiModel(model_cfg).to(self.device)
        self.out_dir = train_cfg.get("save_dir", "out")
        os.makedirs(self.out_dir, exist_ok=True)
        self.optimizer = Muon(self.model.parameters(), lr=train_cfg.get("lr", 1e-3))

    def train(self, dataset_path):
        set_seed(self.cfg.get("seed", 42))
        text = load_text(dataset_path)
        tokenizer = Tokenizer()  # char-level default
        tokens = tokenizer.encode(text)
        ds = TextDataset(tokens, self.model_cfg.block_size)
        dl = DataLoader(ds, batch_size=self.cfg.get("batch_size", 8), shuffle=True, drop_last=True)

        epochs = self.cfg.get("epochs", 1)
        for epoch in range(epochs):
            t0 = time.time()
            total_loss = 0.0
            it = 0
            for xb, yb in dl:
                xb = xb.to(self.device); yb = yb.to(self.device)
                logits = self.model(xb)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                it += 1
                if it % self.cfg.get("log_interval", 50) == 0:
                    print(f"Epoch {epoch+1}, iter {it}, avg_loss {total_loss/it:.4f}")
            t1 = time.time()
            print(f"Epoch {epoch+1} done — avg loss {total_loss/it:.4f} time {t1-t0:.2f}s")
            torch.save(self.model.state_dict(), os.path.join(self.out_dir, f"ckpt_epoch{epoch+1}.pt"))
        torch.save(self.model.state_dict(), os.path.join(self.out_dir, "ckpt_last.pt"))
