# src/inference/word_timestamp_extractor.py
from pathlib import Path
from typing import List, Dict
import json
import numpy as np
import torch
import re

from src.data.label_builder import BLANK_ID
from src.inference.timestamp_extractor import segment_frames_to_times
from src.model.model import FrameAlignmentModel


PREDICTIONS_DIR = Path("outputs/predictions")
CHECKPOINT_DIR = Path("outputs/checkpoints")
OUTPUT_JSON = PREDICTIONS_DIR / "aligned_words.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_latest_checkpoint() -> Path:
    ckpts = list(CHECKPOINT_DIR.glob("model_epoch_*.pt"))
    if not ckpts:
        raise FileNotFoundError("[NO CHECKPOINT FOUND]")

    def epoch_num(p: Path) -> int:
        m = re.search(r"model_epoch_(\d+)\.pt", p.name)
        return int(m.group(1)) if m else -1

    ckpts.sort(key=epoch_num)
    return ckpts[-1]


def load_label_map() -> Dict[int, str]:
    ckpt = torch.load(get_latest_checkpoint(), map_location=DEVICE)
    label_map = ckpt["label_map"]
    return {int(v): k for k, v in label_map.items()}


def extract_word_segments(frame_preds: np.ndarray) -> List[Dict]:
    segments = []
    prev_label = BLANK_ID
    start_frame = None

    for i, label in enumerate(frame_preds):
        if label != prev_label:
            if prev_label != BLANK_ID:
                segments.append(
                    {
                        "label_id": prev_label,
                        "start_frame": start_frame,
                        "end_frame": i - 1,
                    }
                )
            if label != BLANK_ID:
                start_frame = i
            prev_label = label

    if prev_label != BLANK_ID:
        segments.append(
            {
                "label_id": prev_label,
                "start_frame": start_frame,
                "end_frame": len(frame_preds) - 1,
            }
        )

    return segments


def extract_word_timestamps(sample_id: str) -> List[Dict]:
    pred_path = PREDICTIONS_DIR / f"frame_preds_{sample_id}.npy"
    if not pred_path.exists():
        raise FileNotFoundError(f"[PREDICTIONS MISSING] {pred_path}")

    frame_preds = np.load(pred_path)
    if frame_preds.ndim != 1:
        raise RuntimeError(f"[INVALID PRED SHAPE] {frame_preds.shape}")

    id_to_word = load_label_map()
    segments = extract_word_segments(frame_preds)

    word_entries = []

    for seg in segments:
        label_id = seg["label_id"]
        word = id_to_word[label_id]

        start_time, end_time = segment_frames_to_times(
            seg["start_frame"], seg["end_frame"]
        )

        word_entries.append(
            {
                "sample_id": sample_id,
                "word": word,
                "start_time": round(start_time, 2),
                "end_time": round(end_time, 2),
            }
        )

    return word_entries


def append_to_aligned_words(entries: List[Dict]) -> None:
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    existing = []
    if OUTPUT_JSON.exists():
        existing = json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))

    existing.extend(entries)

    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    print(f" Updated alignment file: {OUTPUT_JSON}")
    print(f" Added {len(entries)} word segments")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="STEP 9 — Word timestamp extraction")
    parser.add_argument("--sample_id", type=str, required=True)

    args = parser.parse_args()

    entries = extract_word_timestamps(args.sample_id)
    append_to_aligned_words(entries)

    print("\n STEP 9 COMPLETED SUCCESSFULLY")
