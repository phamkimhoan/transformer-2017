from dataclasses import dataclass
from typing import Literal, Optional

# NUMBER_OF_LAYERS = 2
# DIM_MODEL = 256
# DIM_FEEDFORWARD = 512
# NUMBER_OF_ATTENTION_HEAD = 4
# DROP_OUT = 0.1

# Original Model
NUMBER_OF_LAYERS = 6
DIM_MODEL = 512
DIM_FEEDFORWARD = 2048
NUMBER_OF_ATTENTION_HEAD = 8
DROP_OUT = 0.1

# DATASET_NAME = "bentrevett/multi30k"
DATASET_NAME = "wmt14"
DATASET_CONFIG = "de-en"   # 2nd arg to hf_load_dataset; None for multi30k
SRC_LANGUAGE = "de"
TGT_LANGUAGE = "en"

MAX_VOCABULARY_SIZE = 37000
MIN_VOCAB_FREQ = 2


@dataclass
class TrainConfig:
    batch_size: int = 128
    max_padding: int = 128
    base_lr: float = 1.0
    warmup: int = 3000
    accum_iter: int = 10
    num_epochs: int = 1
    checkpoint_every: int = 1
    file_prefix: str = "small_transformer_multi30k_de_en_"
    directory: Optional[str] = None
    resume_from: Optional[str] = None
    distributed: bool = False
    mode: Literal["preprocess", "train", "test"] = "train"
    label_smoothing: float = 0.1
