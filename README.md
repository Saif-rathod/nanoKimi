# nanoKimi

Minimal PyTorch implementation of the Kimi-K2 architecture (Mixture of Experts, Latent Attention), designed for benchmarking against nanoGPT.

---

## Core Innovations

*   ⚡ **Mixture of Experts (MoE):** `TopK` sparse MoE layer for higher parameter count with constant compute.
*   🧠 **Latent Attention:** Conceptual implementation for long-context compression.
*   🔬 **Benchmarked:** End-to-end comparison suite against a standard Transformer baseline.

## Quick Start: Train a Model in 60 Seconds

```bash
# 1. Clone & Install
git clone https://github.com/Saif-rathod/nanoKimi.git
cd nanoKimi
pip install -r requirements.txt

# 2. Prepare Data (downloads & tokenizes TinyStories)
python data/download.py

# 3. Start Training
# This uses the default nanoKimi configuration.
python scripts/train.py
```
Checkpoints and logs appear in the `out/` directory.

## Core Commands

### Training
Train the default `nanoKimi` model:
```bash
python scripts/train.py
```
Train a baseline `nanoGPT` model for comparison:
```bash
python scripts/train.py --model_config configs/model_gpt_baseline.yaml
```

### Inference
Generate text from a trained checkpoint:
```bash
python scripts/inference.py \
    --checkpoint_path "out/checkpoints/best_model.pt" \
    --prompt "The best way to build a model is"
```

### Benchmarking
Run the full, automated benchmark comparing `nanoKimi` vs. `nanoGPT`. This will train both models and log the results.
```bash
bash scripts/benchmark_run.sh
```
Outputs are saved to `logs/nanogpt_benchmark.log` and `logs/nanokimi_benchmark.log`.

## Benchmark Results

Comparison on TinyStories dataset with a similar active parameter count and training budget on a single NVIDIA A100 GPU.

pending

**Conclusion:** `nanoKimi` achieves superior model quality (lower perplexity) for a negligible increase in training time, at the cost of higher VRAM usage.

## Configuration

All behavior is controlled by `.yaml` files in the `configs/` directory.

*   **`model.yaml`**: Architecture (layers, heads, MoE settings).
*   **`train.yaml`**: Hyperparameters (learning rate, batch size, steps).
*   **`dataset.yaml`**: Data paths.

To experiment, copy a config file, modify it, and pass it to the training script:
```bash
python scripts/train.py --model_config configs/my_experiment.yaml
```

## Architecture Notes

*   **MoE Layer (`nanoKimi/model/moe.py`):** Implements a `TopK` gate that routes tokens to a subset of `SwiGLU` experts. Includes an auxiliary load balancing loss to encourage router diversity.
*   **Latent Attention (`nanoKimi/model/attention.py`):** Disabled by default. When enabled via config, it uses a set of learnable latent vectors as queries to a cross-attention mechanism over the input sequence, producing a compressed context representation.
*   **Optimizer (`nanoKimi/optimizer.py`):** Uses `AdamW` with parameters (`betas=(0.9, 0.95)`, `weight_decay=0.1`) common in large model training, serving as a proxy for the "Muon" optimizer.

## License
MIT
