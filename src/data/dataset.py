#src/data/dataset.py
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.data.annotation_loader import load_split_ids
from src.data.label_builder import BLANK_ID


FEATURES_DIR = Path("data/processed/features")
FRAME_LABELS_DIR = Path("data/processed/frame_labels")


class FrameDataset(Dataset):
    def __init__(self, split_name: str):
        self.split_name = split_name
        self.sample_ids = load_split_ids(split_name)

        if len(self.sample_ids) == 0:
            raise RuntimeError(f"[EMPTY DATASET] split={split_name}")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_id = self.sample_ids[idx]

        feat_path = FEATURES_DIR / f"{sample_id}.npy"
        label_path = FRAME_LABELS_DIR / f"{sample_id}.npy"

        if not feat_path.exists():
            raise FileNotFoundError(f"[FEATURE MISSING] {feat_path}")
        if not label_path.exists():
            raise FileNotFoundError(f"[LABEL MISSING] {label_path}")

        features = np.load(feat_path)
        labels = np.load(label_path)

        if features.shape[0] != labels.shape[0]:
            raise RuntimeError(
                f"[LENGTH MISMATCH] {sample_id}: "
                f"features={features.shape[0]} labels={labels.shape[0]}"
            )

        x = torch.from_numpy(features).float()
        y = torch.from_numpy(labels).long()

        return x, y


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xs, ys = zip(*batch)
    lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)

    max_len = lengths.max().item()
    batch_size = len(xs)
    feat_dim = xs[0].shape[1]

    x_padded = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32)
    y_padded = torch.full(
        (batch_size, max_len),
        fill_value=BLANK_ID,
        dtype=torch.long,
    )

    for i, (x, y) in enumerate(zip(xs, ys)):
        T = x.shape[0]
        x_padded[i, :T] = x
        y_padded[i, :T] = y

    return x_padded, y_padded, lengths


def build_dataloader(
    split_name: str,
    batch_size: int = 16,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    dataset = FrameDataset(split_name)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
    )

    print(
        f"✅ DataLoader built | split={split_name} | "
        f"samples={len(dataset)} | batch_size={batch_size}"
    )

    return loader


if __name__ == "__main__":
    train_loader = build_dataloader(
        split_name="train",
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    x, y, lengths = next(iter(train_loader))

    print("\n🔎 STEP 5 SANITY CHECK")
    print("x shape      :", x.shape)
    print("y shape      :", y.shape)
    print("lengths      :", lengths)
