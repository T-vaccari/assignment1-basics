import time
import pickle

from cs336_basics.bpe.bpe_tokenizer import train_bpe


INPUT_PATH = "data/TinyStoriesV2-GPT4-train.txt"
VOCAB_SIZE = 10_000
SPECIAL_TOKENS = ["<|endoftext|>"]


def main():
   start = time.perf_counter()

   vocab, merges = train_bpe(
        INPUT_PATH,
        VOCAB_SIZE,
        SPECIAL_TOKENS,
   )

   elapsed = time.perf_counter() - start

   print(f"Training time: {elapsed:.2f} s")
   print(f"Vocabulary size: {len(vocab)}")
   print(f"Number of merges: {len(merges)}")

   with open("tinystories_bpe.pkl", "wb") as f:
      pickle.dump(
        {
            "vocab": vocab,
            "merges": merges,
        },
        f,
      )
   # TODO: inspect/report longest learned token


if __name__ == "__main__":
    main()