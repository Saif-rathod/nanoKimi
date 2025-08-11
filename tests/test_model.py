# tests/test_model.py
import torch
from nanoKimi.config import KimiConfig
from nanoKimi.model.kimi import KimiModel

def test_forward_shapes():
    cfg = KimiConfig(vocab_size=256, block_size=64, n_layers=2, d_model=128, latent_dim=32)
    model = KimiModel(cfg)
    x = torch.randint(0, 256, (2, 32), dtype=torch.long)
    logits = model(x)
    assert logits.shape == (2, 32, 256)
    print("test_forward_shapes passed")

if __name__ == "__main__":
    test_forward_shapes()
