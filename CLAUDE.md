# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A PyTorch implementation of the Transformer architecture from "Attention is All You Need" (Vaswani et al., 2017), applied to German-to-English neural machine translation using the Multi30k dataset.

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Download required Spacy language models (also done automatically by load_tokenizers())
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

There is no training entrypoint script — the modules are imported and called directly (e.g., from a notebook or REPL). Key entry points:

```python
from config import *       # hyperparameters and dataset config
from model import make_model
from utils import load_tokenizers, load_vocab, run_epoch, greedy_decode
```

## Architecture

Three files comprise the entire codebase:

**`config.py`** — All hyperparameters and dataset settings as module-level constants (`N`, `D_MODEL`, `D_FF`, `H`, `DROPOUT`, `DATASET`, `SRC_LANGUAGE`, `TGT_LANGUAGE`, `MAX_VOCAB`).

**`model.py`** — Transformer encoder-decoder built from composable layers:
- `make_model(src_vocab, tgt_vocab, N, d_model, d_ff, h, dropout)` — assembles the full model
- `EncoderDecoder` → `Encoder`/`Decoder` → `EncoderLayer`/`DecoderLayer` → `MultiHeadedAttention` + `PositionwiseFeedForward`
- `SublayerConnection` wraps each sublayer with residual connection + layer norm
- `subsequent_mask()` generates causal masks for autoregressive decoding
- `clones(module, N)` replicates layers (creates independent copies via `deepcopy`)

**`utils.py`** — Data pipeline and training loop:
- `load_tokenizers()` — lazily downloads Spacy models and returns `(src_tokenizer, tgt_tokenizer)`
- `load_vocab(spacy_de, spacy_en)` — builds or loads cached `vocab.pt` (shared German+English vocabulary with special tokens `<s>`, `</s>`, `<blank>`, `<unk>`)
- `Batch` — wraps source/target tensors and computes masks; `ntokens` attribute used for loss normalization
- `run_epoch(data_iter, model, loss_compute, optimizer, scheduler, mode, accum_iter)` — training loop with gradient accumulation
- `rate(step, model_size, factor, warmup)` — Noam learning rate schedule (warmup then inverse-sqrt decay)
- `LabelSmoothing` — KL-divergence loss with label smoothing
- `greedy_decode(model, src, src_mask, max_len, start_symbol)` — token-by-token greedy inference

## Key Design Decisions

- **Shared vocabulary**: German and English tokens share a single vocabulary built from both languages together.
- **Vocabulary caching**: `load_vocab()` saves/loads `vocab.pt` to avoid rebuilding on each run.
- **Gradient accumulation**: `run_epoch` accumulates gradients over `accum_iter` batches before stepping — adjust for memory constraints.
- **No beam search**: `greedy_decode` only; beam search is not implemented.
- **No checkpointing**: Model saving/loading is not built into the training utilities.
