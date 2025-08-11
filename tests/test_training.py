import torch
from nanoKimi.model.kimi import NanoKimiModel
from nanoKimi.trainer import train_loop
from nanoKimi.dataset import get_dummy_dataloader
from nanoKimi.optimizer import get_optimizer

def test_training_loop():
    cfg = {
        "vocab_size": 100,
        "dim": 64,
        "n_layers": 2,
        "n_heads": 2,
        "ff_dim": 128,
        "dropout": 0.1
    }

    model = NanoKimiModel(cfg)
    train_loader, val_loader = get_dummy_dataloader(vocab_size=100, seq_len=16, batch_size=2)
    optimizer = get_optimizer({"lr": 1e-3, "optimizer": "adamw"}, model.parameters())

    try:
        train_loop(model, train_loader, val_loader, optimizer, {"epochs": 1, "device": "cpu"}, None, "tmp/")
    except Exception as e:
        assert False, f"Training loop failed with error: {e}"
