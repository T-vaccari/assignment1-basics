import time
import pickle

from cs336_basics.bpe.bpe_training import train_bpe


INPUT_PATH = "data/owt_train.txt"  
VOCAB_SIZE = 32_000
SPECIAL_TOKENS = ["<|endoftext|>"]

OUTPUT_PATH = "scripts/owt_bpe.pkl"


def main():
    start = time.perf_counter()

    vocab, merges = train_bpe(
        INPUT_PATH,
        VOCAB_SIZE,
        SPECIAL_TOKENS,
    )

    elapsed = time.perf_counter() - start

    print(f"Training time:   {elapsed:.2f} s")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(
            {
                "vocab": vocab,
                "merges": merges,
            },
            f,
        )

    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()