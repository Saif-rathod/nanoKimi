#!/usr/bin/env python3
import torch
import argparse
from nanoKimi.config import load_config
from nanoKimi.model.kimi import NanoKimiModel
from nanoKimi.dataset import get_dataloader
from nanoKimi.tokenizer import build_tokenizer
from nanoKimi.optimizer import get_optimizer
from nanoKimi.trainer import train_loop

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune nanoKimi on a dataset")
    parser.add_argument("--config", type=str, required=True, help="Path to training config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to pretrained model checkpoint")
    parser.add_argument("--output_dir", type=str, default="checkpoints/finetuned")
    parser.add_argument("--use_lora", action="store_true", help="Enable LoRA fine-tuning")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)

    # Tokenizer
    tokenizer = build_tokenizer(cfg["tokenizer"])

    # Dataset loader
    train_loader, val_loader = get_dataloader(cfg["dataset"], tokenizer)

    # Model
    model = NanoKimiModel(cfg["model"])
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))

    if args.use_lora:
        from peft import get_peft_model, LoraConfig
        peft_cfg = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, peft_cfg)

    model = model.to(cfg["training"]["device"])

    # Optimizer
    optimizer = get_optimizer(cfg["training"], model.parameters())

    # Fine-tuning
    train_loop(model, train_loader, val_loader, optimizer, cfg["training"], tokenizer, args.output_dir)
