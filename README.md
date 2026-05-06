# Transformer (2017)

PyTorch implementation of the Transformer architecture from ["Attention is All You Need"](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017), trained on German→English translation using the [Multi30k](https://huggingface.co/datasets/bentrevett/multi30k) dataset.

## Setup

```bash
uv pip install -r requirements.txt
```

## Usage

```bash
# Train with defaults
python run.py

# Override hyperparameters from the CLI
python run.py --num-epochs 10 --batch-size 64 --label-smoothing 0.0

# Resume from latest checkpoint (glob-matched by prefix)
python run.py --resume-from small_transformer_multi30k_de_en_

# Evaluate BLEU on the test set
python run.py --mode eval --resume-from small_transformer_multi30k_de_en_

# Show all options
python run.py --help
```

Defaults come from `TrainConfig` in `config.py`. Any field not passed on the CLI falls back to those defaults. Training resumes automatically when `--resume-from` is set to the checkpoint prefix — the latest `epoch*.pt` file is picked up automatically.

## Model

A small Transformer (2 layers, d_model=256, 4 heads) trained end-to-end with:
- Shared German+English vocabulary (10k tokens)
- Noam learning rate schedule with warmup
- Label smoothing (ε=0.1)
- Gradient accumulation

## Project Structure

| File | Purpose |
|------|---------|
| `config.py` | Hyperparameters and `TrainConfig` dataclass |
| `model.py` | Transformer encoder-decoder architecture |
| `utils.py` | Data pipeline, training loop, checkpointing |
| `run.py` | Entry point: train and eval modes |
