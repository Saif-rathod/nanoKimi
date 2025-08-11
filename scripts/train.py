# scripts/train.py
import argparse
import yaml
from nanoKimi.config import load_yaml, KimiConfig
from nanoKimi.trainer import Trainer

def main(config_path="configs/train.yaml", model_config_path="configs/model.yaml"):
    train_cfg = load_yaml(config_path)
    model_cfg_dict = load_yaml(model_config_path)
    model_cfg = KimiConfig(**model_cfg_dict)
    trainer = Trainer(model_cfg, train_cfg)
    trainer.train(train_cfg.get("dataset"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--model_config", default="configs/model.yaml")
    args = parser.parse_args()
    main(args.config, args.model_config)
