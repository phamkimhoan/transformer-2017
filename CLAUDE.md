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
# Step 1 — preprocess locally (builds vocab.pt + dataset_cache.pt, CPU only)
python run.py --mode preprocess

# Step 2 — transfer to training machine
scp vocab.pt dataset_cache.pt root@<host>:/workspace/transformer-2017/

# Step 3 — train on GPU machine (loads cache, no tokenization)
python run.py --mode train --num-epochs 10 --distributed

# Evaluate BLEU on test set
python run.py --mode eval --resume-from wmt14_full_

python run.py --help                   # list all flags
```

`TrainConfig` in `config.py` holds the defaults; CLI args override only what is explicitly passed. All `TrainConfig` fields are exposed as `--kebab-case` flags.

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
- `detokenize(tokens)` — reverses spacy tokenization (reattaches punctuation, fixes contractions) for human-readable output and correct sacrebleu scoring
- `greedy_decode(model, src, src_mask, max_len, start_symbol, eos_symbol)` — batched greedy inference; supports batch > 1 and exits early once all sequences emit `eos_symbol`

**`run.py`** — Entry point:
- `parse_args()` — builds argparse from `TrainConfig` defaults; every field is a `--kebab-case` flag
- `train_model` / `train_worker` — handles device setup (CUDA/MPS/CPU), DDP for multi-GPU CUDA, checkpoint resume, epoch loop
- `eval_model` — loads checkpoint, runs batched greedy decode (batch size from `config.batch_size`) on test set, detokenizes hypotheses, reports sacrebleu BLEU score
- `__main__` calls `parse_args()` then dispatches on `config.mode`

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
- **Logging vs print**: use `logging.info` inside library/model code; `print` is acceptable only in `run.py`.
- **Package management**: use `uv pip install`, not `pip install`.

## Remote Training (vast.ai)

Typical workflow for cloud GPU training:

```bash
# On the remote machine — start a persistent tmux session before launching training
tmux new -s train
cd /workspace/transformer-2017
/workspace/.venv/bin/python run.py --num-epochs 20 --file-prefix my_run_ 2>&1 | tee /workspace/train.log

# Detach without killing: Ctrl-B then D
# Reattach later: tmux attach -t train
# Kill session when done: tmux kill-session -t train
```

Monitor from a second terminal:

```bash
# GPU + CPU utilisation
watch -n 2 "nvidia-smi && echo '---' && free -h"

# Training log
tail -f /workspace/train.log
```

**Known compatibility issues:**

- **Python 3.14**: `datasets` library's `dill` pickling is broken on Python 3.14. Use Python 3.11 or 3.12.
- **NVIDIA Blackwell GPUs (RTX 5060 Ti, sm_120)**: PyTorch stable (≤2.5) does not support sm_120. Install PyTorch nightly with CUDA 12.8:
  ```bash
  uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
  ```
- **PyTorch 2.6+ `torch.load`**: default changed to `weights_only=True`, which blocks unpickling `TrainState` and `TrainConfig`. All `torch.load` calls in `run.py` use `weights_only=False` explicitly.

**Checkpoint resume on remote:** `--resume-from` defaults to `--file-prefix` in `parse_args()`. When restarting a run, pass the same `--file-prefix` and omit `--resume-from` — `find_checkpoint` will glob for the latest `epoch*.pt` automatically.

**`.venv` location:** The virtual environment should be created inside the repo directory (`/workspace/transformer-2017/.venv`). Activate with `source .venv/bin/activate` or invoke directly as `.venv/bin/python run.py ...`.
