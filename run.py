import argparse
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
import GPUtil

from model import make_model
from config import TrainConfig, DIM_MODEL
import sacrebleu

from utils import (
    Batch, TrainState, run_epoch, rate,
    LabelSmoothing, SimpleLossCompute,
    DummyOptimizer, DummyScheduler,
    find_checkpoint, save_checkpoint,
    create_dataloaders, pretokenize_and_cache, load_dataset,
    load_tokenizers, load_vocab,
    tokenize, detokenize, greedy_decode,
)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def empty_cache(device_type):
    if device_type == "cuda":
        torch.cuda.empty_cache()
    elif device_type == "mps":
        torch.mps.empty_cache()


def train_worker(
    gpu, vocab_src, vocab_tgt,
    spacy_src, spacy_tgt, config: TrainConfig, is_distributed=False,
):
    """Run training on a single device.

    Args:
        gpu: GPU index for CUDA (passed automatically by mp.spawn for each
            process). Ignored on MPS/CPU — always 0 in those cases.
        vocab_src: Source language vocabulary.
        vocab_tgt: Target language vocabulary.
        spacy_src: Spacy tokenizer for the source language.
        spacy_tgt: Spacy tokenizer for the target language.
        config: Training configuration (see TrainConfig in config.py).
        is_distributed: If True, initialises a NCCL process group and wraps
            the model in DDP. Only meaningful when device is CUDA and multiple
            GPUs are available; ignored on MPS/CPU.
    """
    device_type = get_device()

    if device_type == "cuda":
        torch.cuda.set_device(gpu)
        device = torch.device(f'cuda:{gpu}')
        ngpus_per_node = torch.cuda.device_count() if is_distributed else 1
    else:
        device = torch.device(device_type)
        ngpus_per_node = 1

    print(f'[{device}] train_worker starting', flush=True)

    pad_idx = vocab_tgt['<blank>']
    model = make_model(len(vocab_src), len(vocab_tgt))
    model.to(device)
    module = model
    is_main_process = True

    if is_distributed and device_type == "cuda":
        dist.init_process_group(
            'nccl', init_method='env://', rank=gpu, world_size=ngpus_per_node
        )
        model = DDP(model, device_ids=[gpu])
        module = model.module
        is_main_process = gpu == 0

    criterion = LabelSmoothing(
        size=len(vocab_tgt), padding_idx=pad_idx, smoothing=config.label_smoothing
    )
    criterion.to(device)

    train_dataloader, valid_dataloader = create_dataloaders(
        device, vocab_src, spacy_src, spacy_tgt,
        batch_size=config.batch_size // ngpus_per_node,
        max_padding=config.max_padding,
        is_distributed=is_distributed and device_type == "cuda",
        directory=config.directory,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.base_lr, betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(step, DIM_MODEL, factor=1, warmup=config.warmup),
    )
    train_state = TrainState()
    start_epoch = 0
    loss_history = []

    ckpt_path = find_checkpoint(config.resume_from)
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        module.load_state_dict(ckpt.get('model', ckpt))
        if 'epoch' in ckpt:
            start_epoch = ckpt['epoch'] + 1
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
            lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
            train_state = ckpt['train_state']
            print(f'[{device}] Resumed at epoch {start_epoch} (steps so far: {train_state.step})', flush=True)
        else:
            print(f'[{device}] Weights-only checkpoint loaded — starting at epoch {start_epoch}', flush=True)
        loss_history = ckpt.get('loss_history', [])
    else:
        print(f'[{device}] Starting fresh.', flush=True)

    total_epochs = start_epoch + config.num_epochs

    for epoch in range(start_epoch, total_epochs):
        if is_distributed and device_type == "cuda":
            train_dataloader.sampler.set_epoch(epoch)
            valid_dataloader.sampler.set_epoch(epoch)

        model.train()
        train_loss, train_state = run_epoch(
            (Batch(b[0].to(device), b[1].to(device), pad_idx) for b in train_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            optimizer, lr_scheduler,
            mode='train+log',
            accum_iter=config.accum_iter,
            train_state=train_state,
            total=len(train_dataloader),
            desc=f'Epoch {epoch} train',
        )

        if device_type == "cuda":
            GPUtil.showUtilization()
        empty_cache(device_type)

        model.eval()
        with torch.no_grad():
            val_loss, _ = run_epoch(
                (Batch(b[0].to(device), b[1].to(device), pad_idx) for b in valid_dataloader),
                model,
                SimpleLossCompute(module.generator, criterion),
                DummyOptimizer(), DummyScheduler(),
                mode='eval',
                total=len(valid_dataloader),
                desc=f'Epoch {epoch} val  ',
            )
        print(f'Epoch {epoch} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f}', flush=True)
        empty_cache(device_type)

        if is_main_process:
            loss_history.append({
                'epoch': epoch,
                'train_loss': float(train_loss),
                'val_loss': float(val_loss),
            })

            if (epoch + 1) % config.checkpoint_every == 0:
                save_checkpoint(
                    {
                        'epoch': epoch,
                        'model': module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'train_state': train_state,
                        'config': config,
                        'loss_history': loss_history,
                    },
                    directory=config.directory,
                    prefix=config.file_prefix,
                )


def preprocess_model(vocab_src, spacy_src, spacy_tgt, config: TrainConfig):
    """Build vocab and pre-tokenize dataset. Run locally; transfer outputs to training machine."""
    directory = config.directory or os.path.dirname(os.path.abspath(__file__))
    pretokenize_and_cache(
        vocab_src, spacy_src, spacy_tgt,
        directory=directory, max_padding=config.max_padding,
    )
    vocab_path = os.path.join(directory, "vocab.pt")
    cache_path = os.path.join(directory, "dataset_cache.pt")
    print(f"\nPreprocessing complete. Transfer these two files to the training machine:")
    print(f"  scp {vocab_path} {cache_path} root@<host>:/workspace/transformer-2017/")


def train_model(vocab_src, vocab_tgt, spacy_src, spacy_tgt, config: TrainConfig):
    device_type = get_device()
    print(f'Using device: {device_type}')

    ckpt_path = find_checkpoint(config.resume_from)
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        epoch = ckpt.get('epoch', '?')
        has_full = 'optimizer' in ckpt
        print(f'Resuming from epoch {epoch} ({"full" if has_full else "weights-only"} checkpoint) — '
              f'will run {config.num_epochs} more epoch(s).')
    else:
        print('No checkpoint found — starting fresh.')

    # Build cache if not already present (fallback for machines without preprocessed data)
    pretokenize_and_cache(
        vocab_src, spacy_src, spacy_tgt,
        directory=config.directory, max_padding=config.max_padding,
    )

    if config.distributed and device_type == "cuda":
        ngpus = torch.cuda.device_count()
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12356'
        print(f'Spawning {ngpus} GPU process(es) ...')
        mp.spawn(
            train_worker, nprocs=ngpus,
            args=(vocab_src, vocab_tgt, spacy_src, spacy_tgt, config, True),
        )
    else:
        train_worker(
            0, vocab_src, vocab_tgt, spacy_src, spacy_tgt, config, False
        )


def eval_model(vocab_src, vocab_tgt, spacy_src, spacy_tgt, config: TrainConfig):
    """Load a checkpoint and evaluate on the test set, reporting BLEU score."""
    assert config.resume_from is not None, "eval mode requires resume_from to be set in TrainConfig"

    device = torch.device(get_device())
    ckpt_path = find_checkpoint(config.resume_from)
    assert ckpt_path is not None, f"Checkpoint not found: {config.resume_from}"

    model = make_model(len(vocab_src), len(vocab_tgt))
    model.to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt.get("model", ckpt))
    model.eval()
    print(f"Loaded model from {ckpt_path} (epoch {ckpt.get('epoch', '?')})")

    _, _, test_data = load_dataset()  # uses SRC_LANGUAGE/TGT_LANGUAGE defaults from config.py
    itos = vocab_tgt.get_itos()
    bos_idx = vocab_src["<s>"]
    eos_idx = vocab_src["</s>"]
    blank_idx = vocab_src["<blank>"]

    def encode_src(text):
        tokens = [bos_idx] + vocab_src(tokenize(text, spacy_src)) + [eos_idx]
        if len(tokens) > config.max_padding:
            tokens = tokens[:config.max_padding - 1] + [eos_idx]
        return tokens

    from tqdm import tqdm
    hypotheses, references = [], []
    batches = [test_data[i:i + config.batch_size] for i in range(0, len(test_data), config.batch_size)]
    with torch.no_grad():
        for batch in tqdm(batches, desc="Evaluating", dynamic_ncols=True):
            src_texts = [pair[0] for pair in batch]
            ref_texts  = [pair[1] for pair in batch]
            encoded = [encode_src(t) for t in src_texts]
            max_src_len = max(len(e) for e in encoded)
            padded = [e + [blank_idx] * (max_src_len - len(e)) for e in encoded]
            src = torch.tensor(padded, dtype=torch.long, device=device)
            src_mask = (src != blank_idx).unsqueeze(-2)
            out = greedy_decode(model, src, src_mask, max_len=config.max_padding,
                                start_symbol=bos_idx, eos_symbol=eos_idx)
            for seq, ref_text in zip(out.tolist(), ref_texts):
                tokens = []
                for tok_id in seq:
                    if tok_id == bos_idx:
                        continue
                    if tok_id == eos_idx:
                        break
                    tokens.append(itos[tok_id])
                hypotheses.append(detokenize(tokens))
                references.append(ref_text)

    result = sacrebleu.corpus_bleu(hypotheses, [references])
    print(f"BLEU: {result.score:.2f}")
    return result.score


def parse_args() -> TrainConfig:
    defaults = TrainConfig()
    p = argparse.ArgumentParser(description="Train/eval the 2017 Transformer (de→en)")
    p.add_argument("--batch-size",        type=int,   default=defaults.batch_size)
    p.add_argument("--max-padding",       type=int,   default=defaults.max_padding)
    p.add_argument("--base-lr",           type=float, default=defaults.base_lr)
    p.add_argument("--warmup",            type=int,   default=defaults.warmup)
    p.add_argument("--accum-iter",        type=int,   default=defaults.accum_iter)
    p.add_argument("--num-epochs",        type=int,   default=defaults.num_epochs)
    p.add_argument("--checkpoint-every",  type=int,   default=defaults.checkpoint_every)
    p.add_argument("--file-prefix",       type=str,   default=defaults.file_prefix)
    p.add_argument("--directory",         type=str,   default=defaults.directory)
    p.add_argument("--resume-from",       type=str,   default=None)
    p.add_argument("--distributed",       action="store_true", default=defaults.distributed)
    p.add_argument("--mode",              type=str,   default=defaults.mode,
                   choices=["preprocess", "train", "eval"])
    p.add_argument("--label-smoothing",   type=float, default=defaults.label_smoothing)
    a = p.parse_args()
    return TrainConfig(
        batch_size=a.batch_size,
        max_padding=a.max_padding,
        base_lr=a.base_lr,
        warmup=a.warmup,
        accum_iter=a.accum_iter,
        num_epochs=a.num_epochs,
        checkpoint_every=a.checkpoint_every,
        file_prefix=a.file_prefix,
        directory=a.directory,
        resume_from=a.resume_from if a.resume_from is not None else a.file_prefix,
        distributed=a.distributed,
        mode=a.mode,
        label_smoothing=a.label_smoothing,
    )


if __name__ == "__main__":
    config = parse_args()
    spacy_src, spacy_tgt = load_tokenizers()
    vocab_src, vocab_tgt = load_vocab(spacy_src, spacy_tgt, directory=config.directory)
    if config.mode == "preprocess":
        preprocess_model(vocab_src, spacy_src, spacy_tgt, config)
    elif config.mode == "train":
        train_model(vocab_src, vocab_tgt, spacy_src, spacy_tgt, config)
    else:
        eval_model(vocab_src, vocab_tgt, spacy_src, spacy_tgt, config)
