# src/data/split.py
from pathlib import Path
import random

FEATURES_DIR = Path("data/processed/features")
SPLIT_DIR = Path("data/splits")

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


def create_splits():
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    # 🔥 CHANGE: use processed FEATURES, not raw audio
    samples = [p.stem for p in FEATURES_DIR.glob("*.npy")]

    if not samples:
        raise RuntimeError("No processed feature files found!")

    random.shuffle(samples)

    n = len(samples)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_ids = samples[:n_train]
    val_ids = samples[n_train:n_train + n_val]
    test_ids = samples[n_train + n_val:]

    (SPLIT_DIR / "train.txt").write_text("\n".join(train_ids))
    (SPLIT_DIR / "val.txt").write_text("\n".join(val_ids))
    (SPLIT_DIR / "test.txt").write_text("\n".join(test_ids))

    print(" Dataset split created successfully (FROM FEATURES)")
    print(f"Train: {len(train_ids)}")
    print(f"Val:   {len(val_ids)}")
    print(f"Test:  {len(test_ids)}")


if __name__ == "__main__":
    create_splits()
