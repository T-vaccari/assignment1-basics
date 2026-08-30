import pickle
from pathlib import Path

import numpy as np

from cs336_basics.bpe.tokenizer import Tokenizer


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

VOCAB_PATH = DATA_DIR / "tinystories_vocab.pkl"
MERGES_PATH = DATA_DIR / "tinystories_merges.pkl"

TRAIN_PATH = DATA_DIR / "TinyStoriesV2-GPT4-train.txt"
VALID_PATH = DATA_DIR / "TinyStoriesV2-GPT4-valid.txt"

TRAIN_OUTPUT = DATA_DIR / "tinystories_train_tokens.npy"
VALID_OUTPUT = DATA_DIR / "tinystories_valid_tokens.npy"


with open(VOCAB_PATH, "rb") as f:
   vocab = pickle.load(f)

with open(MERGES_PATH, "rb") as f:
   merges = pickle.load(f)


tokenizer = Tokenizer(
   vocab=vocab,
   merges=merges,
   special_tokens=["<|endoftext|>"],
)

def tokenize_file(input_path, output_path):
   with open(input_path, "r", encoding="utf-8") as f:
      token_ids = np.fromiter(
         tokenizer.encode_iterable(f),
         dtype=np.uint16,
      )

   np.save(output_path, token_ids)

   print(f"Saved: {output_path}")
   print(f"Tokens: {len(token_ids):,}")
   print(f"Size: {token_ids.nbytes / 1e6:.2f} MB")


if __name__ == "__main__":
   # tokenize_file(VALID_PATH, VALID_OUTPUT)
   tokenize_file(TRAIN_PATH, TRAIN_OUTPUT)