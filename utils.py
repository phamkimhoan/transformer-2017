import collections
from model import subsequent_mask
import time
import torch.nn as nn
import torch
import spacy
import os
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.functional import pad

from datasets import load_dataset as hf_load_dataset

from config import (
    DATASET_NAME,
    DATASET_LANGUAGE_PAIR,
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


class TrainState:
    """Track number of steps, examples, and tokens processed"""

    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed


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
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())


class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        self.state = {}

    def step(self):
        pass

    def zero_grad(self, set_to_none=False):
        pass


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
    train_state=TrainState(),
):
    """Train a single epoch"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        # loss_node = loss_node / accum_iter
        if mode == "train" or mode == "train+log":
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                    "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                    + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0
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


def load_tokenizers():

    spacy_models = {
        "de": "de_core_news_sm",
        "en": "en_core_web_sm",
        "fr": "fr_core_news_sm",
    }

    src_model = spacy_models.get(SRC_LANGUAGE, f"{SRC_LANGUAGE}_core_web_sm")
    tgt_model = spacy_models.get(TGT_LANGUAGE, f"{TGT_LANGUAGE}_core_web_sm")

    try:
        spacy_src = spacy.load(src_model)
    except IOError:
        os.system(f"python -m spacy download {src_model}")
        spacy_src = spacy.load(src_model)

    try:
        spacy_tgt = spacy.load(tgt_model)
    except IOError:
        os.system(f"python -m spacy download {tgt_model}")
        spacy_tgt = spacy.load(tgt_model)

    return spacy_src, spacy_tgt


def tokenize(text, tokenizer):
    return [tok.text for tok in tokenizer.tokenizer(text)]


def yield_tokens(data_iter, tokenizer, index):
    for from_to_tuple in data_iter:
        yield tokenizer(from_to_tuple[index])


# Replaces torchtext.datasets.DATASET_NAME — returns (train, validation, test)
# each as a list of (SRC_LANGUAGE_text, TGT_LANGUAGE_text) tuples
def load_dataset():
    ds = hf_load_dataset(DATASET_NAME, DATASET_LANGUAGE_PAIR)

    def to_pairs(split):
        return [
            (row["translation"][SRC_LANGUAGE], row["translation"][TGT_LANGUAGE])
            for row in ds[split]
        ]

    return to_pairs("train"), to_pairs("validation"), to_pairs("test")


def build_vocabulary(spacy_src, spacy_tgt):
    def tokenize_src(text):
        return tokenize(text, spacy_src)

    def tokenize_tgt(text):
        return tokenize(text, spacy_tgt)

    print(
        f"Building shared {SRC_LANGUAGE.upper()}+{TGT_LANGUAGE.upper()} Vocabulary ..."
    )
    train, val, test = load_dataset()
    all_pairs = train + val + test

    def yield_all_tokens(data):
        yield from yield_tokens(data, tokenize_src, index=0)
        yield from yield_tokens(data, tokenize_tgt, index=1)

    vocab = build_vocab_from_iterator(
        yield_all_tokens(all_pairs),
        min_freq=MIN_VOCAB_FREQ,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
        max_tokens=MAX_VOCABULARY_SIZE,
    )
    vocab.set_default_index(vocab["<unk>"])
    return vocab, vocab


def load_vocab(spacy_src, spacy_tgt, directory=None):
    if directory is None:
        directory = os.path.dirname(os.path.abspath(__file__))
    vocab_path = os.path.join(directory, "vocab.pt")

    if os.path.exists(vocab_path):
        print(f"Loading vocab from {vocab_path}")
        vocab_src, vocab_tgt = torch.load(vocab_path)
    else:
        print("No vocab.pt found — building from scratch (this may take a while)...")
        vocab_src, vocab_tgt = build_vocabulary(spacy_src, spacy_tgt)
        torch.save((vocab_src, vocab_tgt), vocab_path)
        print(f"Saved vocab to {vocab_path}")

    print(f"Finished.\nVocabulary size: {len(vocab_src)}")
    return vocab_src, vocab_tgt


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
        src_list.append(
            # warning - overwrites values for negative values of padding - len
            pad(
                processed_src,
                (
                    0,
                    max_padding - len(processed_src),
                ),
                value=pad_id,
            )
        )
        tgt_list.append(
            pad(
                processed_tgt,
                (0, max_padding - len(processed_tgt)),
                value=pad_id,
            )
        )

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
    # def create_dataloaders(batch_size=12000):
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


def find_checkpoint(directory=None, prefix="model_"):
    if directory is None:
        directory = os.path.dirname(os.path.abspath(__file__))
    matches = [
        f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(".pt")
    ]
    if not matches:
        print(f"No checkpoint found in {directory}")
        return None
    path = max(matches, key=lambda f: os.path.getmtime(os.path.join(directory, f)))
    path = os.path.join(directory, path)
    print(f"Found checkpoint: {path}")
    return path


def save_checkpoint(ckpt_dict, directory=None, prefix="model_"):
    if directory is None:
        directory = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(directory, f"{prefix}checkpoint.pt")
    torch.save(ckpt_dict, path)
    size_mb = os.path.getsize(path) / 1_000_000
    print(f"Checkpoint saved: {path}  ({size_mb:.1f} MB)")
    return path


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    return ys
