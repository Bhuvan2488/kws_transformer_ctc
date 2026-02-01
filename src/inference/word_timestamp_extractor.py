# src/inference/word_timestamp_extractor.py
from pathlib import Path
from typing import List, Dict
import json
import numpy as np
import torch

from src.data.label_builder import BLANK_ID
from src.inference.timestamp_extractor import segment_frames_to_times

PREDICTIONS_DIR = Path("outputs/predictions")
CHECKPOINT_DIR = Path("outputs/checkpoints")
OUTPUT_JSON = PREDICTIONS_DIR / "aligned_words.json"


def load_label_map():
    ckpts = sorted(
        CHECKPOINT_DIR.glob("model_epoch_*.pt"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    ckpt = torch.load(ckpts[-1], map_location="cpu")
    lm = ckpt["label_map"]
    return {int(v): k for k, v in lm.items()}


def extract_word_segments(frame_preds: np.ndarray):
    segments, prev, start = [], BLANK_ID, None
    for i, lab in enumerate(frame_preds):
        if lab != prev:
            if prev != BLANK_ID:
                segments.append({"label_id": prev, "start_frame": start, "end_frame": i - 1})
            if lab != BLANK_ID:
                start = i
            prev = lab
    if prev != BLANK_ID:
        segments.append({"label_id": prev, "start_frame": start, "end_frame": len(frame_preds) - 1})
    return segments


def extract_word_timestamps(sample_id: str):
    preds = np.load(PREDICTIONS_DIR / f"frame_preds_{sample_id}.npy")
    id_to_word = load_label_map()
    segments = extract_word_segments(preds)

    out = []
    for s in segments:
        st, et = segment_frames_to_times(s["start_frame"], s["end_frame"])
        out.append({
            "sample_id": sample_id,
            "word": id_to_word[s["label_id"]],
            "start_time": round(st, 2),
            "end_time": round(et, 2),
        })
    return out
