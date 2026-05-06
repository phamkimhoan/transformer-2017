# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A PyTorch implementation of the Transformer architecture from "Attention is All You Need" (Vaswani et al., 2017), applied to German-to-English neural machine translation using the Multi30k dataset (`bentrevett/multi30k` on HuggingFace).

## Setup

```bash
# Install dependencies (use uv)
uv pip install -r requirements.txt
```

Spacy models are downloaded automatically by `load_tokenizers()` on first run.

## Running

```bash
# Train (resumes from latest checkpoint if resume_from is set in TrainConfig)
python train.py

# Evaluate BLEU on test set
# Set mode="eval" in TrainConfig, then:
python train.py
```

All training configuration lives in `TrainConfig` in `config.py` — edit it directly before running.

## Architecture

Four files comprise the entire codebase:

**`config.py`** — Two concerns:
- Module-level constants used as defaults: `NUMBER_OF_LAYERS`, `DIM_MODEL`, `DIM_FEEDFORWARD`, `NUMBER_OF_ATTENTION_HEAD`, `DROP_OUT`, `DATASET_NAME`, `SRC_LANGUAGE`, `TGT_LANGUAGE`, `MAX_VOCABULARY_SIZE`, `MIN_VOCAB_FREQ`
- `TrainConfig` dataclass: runtime training settings (`batch_size`, `num_epochs`, `warmup`, `accum_iter`, `resume_from`, `mode`, etc.)

**`model.py`** — Transformer encoder-decoder:
- `make_model(src_vocab, tgt_vocab, N, d_model, d_ff, h, dropout)` — assembles the full model; all params have defaults from config constants
- `EncoderDecoder` → `Encoder`/`Decoder` → `EncoderLayer`/`DecoderLayer` → `MultiHeadedAttention` + `PositionwiseFeedForward`
- `SublayerConnection` implements pre-norm residual connection
- `subsequent_mask(size)` — causal mask for autoregressive decoding
- `clones(module, N)` — deep-copies a layer N times into a `ModuleList`

**`utils.py`** — Data pipeline and training utilities:
- `load_tokenizers(src_lang, tgt_lang)` — loads Spacy models, downloads if missing
- `load_vocab(spacy_src, spacy_tgt, directory, ...)` — builds or loads cached `vocab.pt`; invalidates cache if config changes
- `load_dataset(dataset_name, src_lang, tgt_lang)` — loads `bentrevett/multi30k` from HuggingFace; fields are `row[src_lang]` / `row[tgt_lang]` directly (no `"translation"` nesting)
- `Batch` — wraps src/tgt tensors, builds causal + padding masks; `ntokens` used for loss normalisation
- `run_epoch(data_iter, model, loss_compute, optimizer, scheduler, mode, accum_iter, train_state, total, desc)` — training/eval loop with tqdm progress bar; shows live loss, lr, tok/s per batch
- `TrainState` dataclass — tracks `step`, `accum_step`, `samples`, `tokens` across epochs
- `rate(step, model_size, factor, warmup)` — Noam LR schedule
- `LabelSmoothing` — KLDivLoss with label smoothing
- `save_checkpoint(ckpt_dict, directory, prefix)` — saves as `{prefix}epoch{N:03d}.pt`
- `find_checkpoint(path)` — accepts exact path or prefix; globs for `{prefix}epoch*.pt` and returns the latest
- `greedy_decode(model, src, src_mask, max_len, start_symbol)` — token-by-token greedy inference

**`train.py`** — Entry point:
- `train_model` / `train_worker` — handles device setup (CUDA/MPS/CPU), DDP for multi-GPU CUDA, checkpoint resume, epoch loop
- `eval_model` — loads checkpoint, runs greedy decode on test set, reports sacrebleu BLEU score
- `__main__` dispatches on `config.mode` (`"train"` or `"eval"`)

## Key Design Decisions

- **Shared vocabulary**: German and English tokens share one vocabulary built from train+val only (test set excluded to avoid leakage).
- **Vocabulary caching**: `load_vocab()` saves `vocab.pt` alongside a config fingerprint; rebuilds automatically if any vocab-relevant setting changes.
- **Gradient accumulation**: gradients accumulate over `accum_iter` batches before each optimizer step; loss is scaled by `1/accum_iter` before `.backward()`.
- **Epoch-stamped checkpoints**: saved as `{prefix}epoch000.pt`, `epoch001.pt`, etc. Set `resume_from` to the prefix and `find_checkpoint` picks the latest automatically.
- **DDP only on CUDA**: distributed training via `mp.spawn` + NCCL is gated on `device_type == "cuda"`. MPS and CPU always use a single process.
- **No beam search**: `greedy_decode` only.

## Code Conventions

- **Explicit parameters with defaults**: functions never read config constants directly from the module scope inside their body. All config dependencies are explicit parameters with the constant as the default value (e.g. `def load_vocab(..., max_vocab=MAX_VOCABULARY_SIZE)`).
- **`@dataclass` for all data containers**: plain classes with class-level annotations are forbidden — they cause shared state across instances. Every data container uses `@dataclass`.
- **Subprocesses**: always `subprocess.run([sys.executable, ...], check=True)`, never `os.system`.
- **Logging vs print**: use `logging.info` inside library/model code; `print` is acceptable only in `train.py`.
- **Package management**: use `uv pip install`, not `pip install`.
