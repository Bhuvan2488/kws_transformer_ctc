import json
import numpy as np
from pathlib import Path
from typing import List, Dict

import torch
from torch.utils.data import Dataset


class SpeechDataset(Dataset):
    """
    Dataset for Transformer + CTC training.

    Returns:
        audio: Tensor (T, F)
        tokens: Tensor (L,)
        audio_length: int
        token_length: int
    """

    def __init__(
        self,
        split_file: Path,
        features_dir: Path,
        tokens_dir: Path,
    ):
        self.sample_ids = self._load_split(split_file)
        self.features_dir = features_dir
        self.tokens_dir = tokens_dir

        if len(self.sample_ids) == 0:
            raise RuntimeError(f"Split file {split_file} is empty!")

    def _load_split(self, split_file: Path) -> List[str]:
        with open(split_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]

        # ---------------------------
        # Load audio features
        # ---------------------------
        feat_path = self.features_dir / f"{sample_id}.npy"
        if not feat_path.exists():
            raise FileNotFoundError(f"Missing feature file: {feat_path}")

        features = np.load(feat_path)

        # ---------------------------
        # Load tokenized text
        # ---------------------------
        token_path = self.tokens_dir / f"{sample_id}.json"
        if not token_path.exists():
            raise FileNotFoundError(f"Missing token file: {token_path}")

        with open(token_path, "r", encoding="utf-8") as f:
            token_data = json.load(f)

        # IMPORTANT FIX:
        # token_data is a LIST, not a dict
        tokens = np.array(token_data, dtype=np.int64)

        return {
            "audio": torch.tensor(features, dtype=torch.float32),
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "audio_length": features.shape[0],
            "token_length": len(tokens),
        }


# ------------------------------------------------------------
# COLLATE FUNCTION (FOR DATALOADER)
# ------------------------------------------------------------
def collate_fn(batch: List[Dict]):
    """
    Pads variable-length audio & token sequences for CTC.
    """

    audio_lengths = torch.tensor(
        [item["audio_length"] for item in batch], dtype=torch.long
    )
    token_lengths = torch.tensor(
        [item["token_length"] for item in batch], dtype=torch.long
    )

    max_audio_len = int(audio_lengths.max())
    max_token_len = int(token_lengths.max())

    batch_size = len(batch)
    feat_dim = batch[0]["audio"].shape[1]

    padded_audio = torch.zeros(batch_size, max_audio_len, feat_dim)
    padded_tokens = torch.zeros(batch_size, max_token_len, dtype=torch.long)

    for i, item in enumerate(batch):
        padded_audio[i, : item["audio_length"]] = item["audio"]
        padded_tokens[i, : item["token_length"]] = item["tokens"]

    return {
        "audio": padded_audio,
        "tokens": padded_tokens,
        "audio_lengths": audio_lengths,
        "token_lengths": token_lengths,
    }


# ------------------------------------------------------------
# DEBUG RUN
# ------------------------------------------------------------
if __name__ == "__main__":
    dataset = SpeechDataset(
        split_file=Path("data/splits/train.txt"),
        features_dir=Path("data/processed/features"),
        tokens_dir=Path("data/processed/tokenize_text"),
    )

    sample = dataset[0]
    print("Audio shape:", sample["audio"].shape)
    print("Token length:", sample["token_length"])
