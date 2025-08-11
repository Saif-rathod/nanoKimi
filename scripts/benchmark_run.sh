#!/usr/bin/env bash
# scripts/benchmark_run.sh
set -e
python scripts/train.py --config configs/train.yaml --model_config configs/model.yaml
python scripts/inference.py --ckpt out/ckpt_last.pt --prompt "Hello world" --max_new_tokens 50
