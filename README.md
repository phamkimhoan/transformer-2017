# Transformer (2017)

PyTorch implementation of the Transformer architecture from ["Attention is All You Need"](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017), trained on German→English translation using the [Multi30k](https://huggingface.co/datasets/bentrevett/multi30k) dataset.

## Setup

```bash
uv pip install -r requirements.txt
```

## Usage

Edit `TrainConfig` in `config.py`, then:

```bash
# Train
python train.py

# Evaluate BLEU on test set (set mode="eval" and resume_from in TrainConfig)
python train.py
```

Training resumes automatically from the latest checkpoint when `resume_from` is set to the file prefix (e.g. `"small_transformer_multi30k_de_en_"`).

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
| `train.py` | Entry point: train and eval modes |
