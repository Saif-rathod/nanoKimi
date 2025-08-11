# nanoKimi/dataset.py
import os
from torch.utils.data import Dataset
from typing import List

def load_text(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

class TextDataset(Dataset):
    def __init__(self, tokens: List[int], block_size: int):
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.tokens) - self.block_size - 1)

    def __getitem__(self, idx):
        x = self.tokens[idx: idx + self.block_size]
        y = self.tokens[idx + 1: idx + 1 + self.block_size]
        import torch
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
