#scripts/infer_alignment.py
import os
import sys
sys.path.append(os.getcwd())

from pathlib import Path
import argparse

from src.inference.predict_frames import predict_frames
from src.inference.word_timestamp_extractor import (
    extract_word_timestamps,
    append_to_aligned_words,
)


def infer_alignment(sample_id: str) -> None:
    print("\n INFERENCE PIPELINE STARTED")
    print(f" Sample ID: {sample_id}")

    print("\n STEP 8 — Predicting frame labels")
    frame_pred_path = predict_frames(sample_id)

    print("\n STEP 9 — Extracting word timestamps")
    word_entries = extract_word_timestamps(sample_id)
    append_to_aligned_words(word_entries)

    print("\n INFERENCE PIPELINE COMPLETED SUCCESSFULLY")
    print(f" Results appended to: outputs/predictions/aligned_words.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run STEP 8 + STEP 9: frame prediction and word alignment"
    )
    parser.add_argument(
        "--sample_id",
        type=str,
        required=True,
        help="Sample ID (without extension)",
    )

    args = parser.parse_args()
    infer_alignment(args.sample_id)
