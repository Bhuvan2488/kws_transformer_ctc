# scripts/infer_alignment.py
import os
import sys
sys.path.append(os.getcwd())

from pathlib import Path
from src.data.annotation_loader import load_split_ids
from src.inference.predict_frames import predict_frames
from src.inference.word_timestamp_extractor import (
    extract_word_timestamps,
    append_to_aligned_words,
    OUTPUT_JSON,
)

def infer_test_set():
    print("\n INFERENCE PIPELINE — TEST SET")

    test_ids = load_split_ids("test")
    print(f" Total test samples: {len(test_ids)}")

    # Reset predictions file
    if OUTPUT_JSON.exists():
        OUTPUT_JSON.unlink()
        print(" Cleared old aligned_words.json")

    for i, sample_id in enumerate(test_ids, 1):
        print(f"\n [{i}/{len(test_ids)}] Processing sample: {sample_id}")

        predict_frames(sample_id)
        entries = extract_word_timestamps(sample_id)
        append_to_aligned_words(entries)

    print("\n TEST SET INFERENCE COMPLETED")

if __name__ == "__main__":
    infer_test_set()
