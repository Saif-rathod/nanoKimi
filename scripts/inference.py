# scripts/inference.py
import argparse
import torch
from nanoKimi.config import KimiConfig, load_yaml
from nanoKimi.tokenizer import Tokenizer
from nanoKimi.model.kimi import KimiModel

def main(ckpt, prompt="Hello", max_new_tokens=100, model_config="configs/model.yaml"):
    model_cfg = KimiConfig(**load_yaml(model_config))
    tokenizer = Tokenizer()
    model = KimiModel(model_cfg)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
    out = model.generate(input_ids, max_new_tokens=max_new_tokens)
    print(tokenizer.decode(out[0].tolist()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    args = parser.parse_args()
    main(args.ckpt, args.prompt, args.max_new_tokens)
