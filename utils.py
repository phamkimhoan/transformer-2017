import collections
import subprocess
import sys
from dataclasses import dataclass
from model import subsequent_mask
import time
import torch.nn as nn
import torch
import spacy
import os
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.functional import pad
from tqdm import tqdm

from datasets import load_dataset as hf_load_dataset

from config import (
    DATASET_NAME,
    SRC_LANGUAGE,
    TGT_LANGUAGE,
    MAX_VOCABULARY_SIZE,
    MIN_VOCAB_FREQ,
)


# Drop-in for torchtext.data.functional.to_map_style_dataset
class _MapStyleDataset(Dataset):
    def __init__(self, iterable):
        self._data = list(iterable)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]


def to_map_style_dataset(iter_dataset):
    return _MapStyleDataset(iter_dataset)


# Drop-in for torchtext.vocab.build_vocab_from_iterator
class _Vocab:
    def __init__(self, stoi, itos):
        self._stoi = stoi
        self._itos = itos

    def __call__(self, tokens):
        unk = self._stoi.get("<unk>", 0)
        return [self._stoi.get(t, unk) for t in tokens]

    def __len__(self):
        return len(self._itos)

    def __getitem__(self, token):
        return self._stoi.get(token, self._stoi.get("<unk>", 0))

    def get_stoi(self):
        return self._stoi

    def get_itos(self):
        return self._itos

    def set_default_index(self, idx):
        self._stoi["<unk>"] = idx


def build_vocab_from_iterator(iterator, min_freq=1, specials=None, max_tokens=None):
    counter = collections.Counter()
    for tokens in iterator:
        counter.update(tokens)
    specials = specials or []
    tokens = [
        tok
        for tok, cnt in counter.most_common()
        if cnt >= min_freq and tok not in specials
    ]
    if max_tokens is not None:
        tokens = tokens[: max_tokens - len(specials)]
    itos = specials + tokens
    stoi = {tok: i for i, tok in enumerate(itos)}
    return _Vocab(stoi, itos)


class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src, tgt=None, pad=2):  # 2 = <blank>
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask


@dataclass
class TrainState:
    """Track number of steps, examples, and tokens processed"""
    step: int = 0
    accum_step: int = 0
    samples: int = 0
    tokens: int = 0


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.detach().clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        true_dist.masked_fill_((target == self.padding_idx).unsqueeze(1), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())


class DummyOptimizer:
    param_groups = [{"lr": 0}]

    def step(self): pass

    def zero_grad(self, set_to_none=False): pass


class DummyScheduler:
    def step(self):
        pass


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        x = self.generator(x)
        sloss = (
            self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1))
            / norm
        )
        return sloss.data * norm, sloss


def run_epoch(
    data_iter,
    model,
    loss_compute,
    optimizer,
    scheduler,
    mode="train",
    accum_iter=1,
    train_state=None,
    total=None,
    desc="",
):
    """Train or evaluate a single epoch."""
    if train_state is None:
        train_state = TrainState()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    start = time.time()

    bar = tqdm(data_iter, total=total, desc=desc, dynamic_ncols=True, leave=True)
    for i, batch in enumerate(bar):
        out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        if mode == "train" or mode == "train+log":
            loss_node = loss_node / accum_iter
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens
            if (i + 1) % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
                scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens

        elapsed = time.time() - start
        tok_per_sec = float(tokens) / elapsed if elapsed > 0 else 0.0
        lr = optimizer.param_groups[0]["lr"]
        bar.set_postfix(
            loss=f"{float(loss) / float(batch.ntokens):.3f}",
            lr=f"{lr:.2e}",
            tok_s=f"{tok_per_sec:.0f}",
        )

        del loss
        del loss_node

    return total_loss / total_tokens, train_state


def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


# Load spacy tokenizer models, download them if they haven't been
# downloaded already


def load_tokenizers(src_lang=SRC_LANGUAGE, tgt_lang=TGT_LANGUAGE):
    spacy_models = {
        "de": "de_core_news_sm",
        "en": "en_core_web_sm",
        "fr": "fr_core_news_sm",
    }

    src_model = spacy_models.get(src_lang, f"{src_lang}_core_web_sm")
    tgt_model = spacy_models.get(tgt_lang, f"{tgt_lang}_core_web_sm")

    try:
        spacy_src = spacy.load(src_model)
    except IOError:
        subprocess.run([sys.executable, "-m", "spacy", "download", src_model], check=True)
        spacy_src = spacy.load(src_model)

    try:
        spacy_tgt = spacy.load(tgt_model)
    except IOError:
        subprocess.run([sys.executable, "-m", "spacy", "download", tgt_model], check=True)
        spacy_tgt = spacy.load(tgt_model)

    return spacy_src, spacy_tgt


def tokenize(text, tokenizer):
    return [tok.text for tok in tokenizer.tokenizer(text)]


def detokenize(tokens):
    import re
    text = " ".join(tokens)
    text = re.sub(r" ([.,!?;:\)])", r"\1", text)
    text = re.sub(r"([\(]) ", r"\1", text)
    text = re.sub(r" '(s|t|re|ve|ll|d|m)\b", r"'\1", text)
    return text


def yield_tokens(data_iter, tokenizer, index):
    for from_to_tuple in data_iter:
        yield tokenizer(from_to_tuple[index])


# Replaces torchtext.datasets.DATASET_NAME — returns (train, validation, test)
# each as a list of (SRC_LANGUAGE_text, TGT_LANGUAGE_text) tuples
def load_dataset(
    dataset_name=DATASET_NAME,
    src_lang=SRC_LANGUAGE,
    tgt_lang=TGT_LANGUAGE,
):
    ds = hf_load_dataset(dataset_name)

    def to_pairs(split):
        return [
            (row[src_lang], row[tgt_lang])
            for row in ds[split]
        ]

    return to_pairs("train"), to_pairs("validation"), to_pairs("test")


def build_vocabulary(
    spacy_src,
    spacy_tgt,
    src_lang=SRC_LANGUAGE,
    tgt_lang=TGT_LANGUAGE,
    max_vocab=MAX_VOCABULARY_SIZE,
    min_freq=MIN_VOCAB_FREQ,
):
    print(f"Building shared {src_lang.upper()}+{tgt_lang.upper()} Vocabulary ...")
    train, val, _ = load_dataset(src_lang=src_lang, tgt_lang=tgt_lang, dataset_name=DATASET_NAME)
    all_pairs = train + val

    def yield_all_tokens(data):
        yield from yield_tokens(data, lambda t: tokenize(t, spacy_src), index=0)
        yield from yield_tokens(data, lambda t: tokenize(t, spacy_tgt), index=1)

    vocab = build_vocab_from_iterator(
        yield_all_tokens(all_pairs),
        min_freq=min_freq,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
        max_tokens=max_vocab,
    )
    vocab.set_default_index(vocab["<unk>"])
    return vocab


def load_vocab(
    spacy_src,
    spacy_tgt,
    directory=None,
    src_lang=SRC_LANGUAGE,
    tgt_lang=TGT_LANGUAGE,
    max_vocab=MAX_VOCABULARY_SIZE,
    min_freq=MIN_VOCAB_FREQ,
    dataset=DATASET_NAME,
):
    if directory is None:
        directory = os.path.dirname(os.path.abspath(__file__))
    vocab_path = os.path.join(directory, "vocab.pt")
    current_config = {
        "src": src_lang, "tgt": tgt_lang,
        "max_vocab": max_vocab, "min_freq": min_freq, "dataset": dataset,
    }

    if os.path.exists(vocab_path):
        saved = torch.load(vocab_path, weights_only=False)
        if saved.get("config") != current_config:
            print("Vocab config changed — rebuilding...")
            os.remove(vocab_path)
        else:
            vocab = _Vocab(saved["stoi"], saved["itos"])
            print(f"Loaded vocab from {vocab_path}\nVocabulary size: {len(vocab)}")
            return vocab, vocab

    print("No vocab.pt found — building from scratch (this may take a while)...")
    vocab = build_vocabulary(spacy_src, spacy_tgt, src_lang, tgt_lang, max_vocab, min_freq)
    torch.save({"stoi": vocab.get_stoi(), "itos": vocab.get_itos(), "config": current_config}, vocab_path)
    print(f"Saved vocab to {vocab_path}\nVocabulary size: {len(vocab)}")
    return vocab, vocab


def collate_batch(
    batch,
    src_pipeline,
    tgt_pipeline,
    src_vocab,
    tgt_vocab,
    device,
    max_padding=128,
    pad_id=2,
):
    bs_id = torch.tensor([0], device=device)  # <s> token id
    eos_id = torch.tensor([1], device=device)  # </s> token id
    src_list, tgt_list = [], []
    for _src, _tgt in batch:
        processed_src = torch.cat(
            [
                bs_id,
                torch.tensor(
                    src_vocab(src_pipeline(_src)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        processed_tgt = torch.cat(
            [
                bs_id,
                torch.tensor(
                    tgt_vocab(tgt_pipeline(_tgt)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        if len(processed_src) > max_padding:
            processed_src = processed_src[:max_padding]
            processed_src[-1] = eos_id[0]
        if len(processed_tgt) > max_padding:
            processed_tgt = processed_tgt[:max_padding]
            processed_tgt[-1] = eos_id[0]
        src_list.append(pad(processed_src, (0, max_padding - len(processed_src)), value=pad_id))
        tgt_list.append(pad(processed_tgt, (0, max_padding - len(processed_tgt)), value=pad_id))

    src = torch.stack(src_list)
    tgt = torch.stack(tgt_list)
    return (src, tgt)


def create_dataloaders(
    device,
    vocab_src,
    vocab_tgt,
    spacy_src,
    spacy_tgt,
    batch_size=12000,
    max_padding=128,
    is_distributed=True,
):
    def tokenize_src(text):
        return tokenize(text, spacy_src)

    def tokenize_tgt(text):
        return tokenize(text, spacy_tgt)

    def collate_fn(batch):
        return collate_batch(
            batch,
            tokenize_src,
            tokenize_tgt,
            vocab_src,
            vocab_tgt,
            device,
            max_padding=max_padding,
            pad_id=vocab_src.get_stoi()["<blank>"],
        )

    train_iter, valid_iter, test_iter = load_dataset()

    train_iter_map = to_map_style_dataset(
        train_iter
    )  # DistributedSampler needs a dataset len()
    train_sampler = DistributedSampler(train_iter_map) if is_distributed else None
    valid_iter_map = to_map_style_dataset(valid_iter)
    valid_sampler = DistributedSampler(valid_iter_map) if is_distributed else None

    train_dataloader = DataLoader(
        train_iter_map,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    valid_dataloader = DataLoader(
        valid_iter_map,
        batch_size=batch_size,
        shuffle=(valid_sampler is None),
        sampler=valid_sampler,
        collate_fn=collate_fn,
    )
    return train_dataloader, valid_dataloader


def find_checkpoint(path):
    if path is None:
        return None
    if os.path.exists(path):
        print(f'Loading checkpoint: {path}')
        return path
    # treat as prefix glob: find latest epoch file matching <path>epoch*.pt
    import glob
    matches = sorted(glob.glob(f"{path}epoch*.pt"))
    if matches:
        latest = matches[-1]
        print(f'Loading checkpoint: {latest}')
        return latest
    print(f'Checkpoint not found: {path}')
    return None


def save_checkpoint(ckpt_dict, directory=None, prefix="model_"):
    if directory is None:
        directory = os.path.dirname(os.path.abspath(__file__))
    epoch = ckpt_dict.get("epoch", 0)
    path = os.path.join(directory, f"{prefix}epoch{epoch:03d}.pt")
    torch.save(ckpt_dict, path)
    size_mb = os.path.getsize(path) / 1_000_000
    print(f"Checkpoint saved: {path}  ({size_mb:.1f} MB)")
    return path


def greedy_decode(model, src, src_mask, max_len, start_symbol, eos_symbol=None):
    batch_size = src.size(0)
    memory = model.encode(src, src_mask)
    ys = torch.full((batch_size, 1), start_symbol, dtype=src.dtype, device=src.device)
    done = torch.zeros(batch_size, dtype=torch.bool, device=src.device)
    for _ in range(max_len - 1):
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).to(src.device)
        )
        _, next_word = torch.max(model.generator(out[:, -1]), dim=1)
        ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)
        if eos_symbol is not None:
            done |= (next_word == eos_symbol)
            if done.all():
                break
    return ys
