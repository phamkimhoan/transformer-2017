from dataclasses import dataclass
from typing import Literal, Optional

NUMBER_OF_LAYERS = 2
DIM_MODEL = 256
DIM_FEEDFORWARD = 512
NUMBER_OF_ATTENTION_HEAD = 4
DROP_OUT = 0.1

DATASET_NAME = "bentrevett/multi30k"
SRC_LANGUAGE = "de"
TGT_LANGUAGE = "en"

MAX_VOCABULARY_SIZE = 10000
MIN_VOCAB_FREQ = 2


@dataclass
class TrainConfig:
    batch_size: int = 32
    max_padding: int = 128
    base_lr: float = 1.0
    warmup: int = 3000
    accum_iter: int = 10
    num_epochs: int = 1
    checkpoint_every: int = 1
    file_prefix: str = "small_transformer_multi30k_de_en_"
    directory: Optional[str] = None
    resume_from: Optional[str] = "small_transformer_multi30k_de_en_"
    distributed: bool = False
    mode: Literal["train", "eval"] = "train"
