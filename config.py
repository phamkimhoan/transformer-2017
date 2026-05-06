NUMBER_OF_LAYERS = 2
DIM_MODEL = 256
DIM_FEEDFORWARD = 512
NUMBER_OF_ATTENTION_HEAD = 4
DROP_OUT = 0.1

DATASET_NAME = "bentrevett/multi30k"
DATASET_LANGUAGE_PAIR = "de-en"
SRC_LANGUAGE = "de"
TGT_LANGUAGE = "en"

MAX_VOCABULARY_SIZE = 10000
MIN_VOCAB_FREQ = 2

TRAIN_CONFIG = {
    "batch_size": 32,
    "max_padding": 128,
    "base_lr": 1.0,
    "warmup": 3000,
    "accum_iter": 10,
    "num_epochs": 8,
    "file_prefix": "model_",
    "directory": None,
    "distributed": False,
}