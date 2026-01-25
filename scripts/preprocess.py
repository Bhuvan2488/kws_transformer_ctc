# scripts/preprocess.py
import os
import sys
sys.path.append(os.getcwd())

from pathlib import Path
import argparse

from src.data.annotation_loader import build_sample_index
from src.data.clean import clean_sample_index
from src.data.feature_extraction import extract_features
from src.data.label_builder import build_labels_for_dataset


FEATURES_DIR = Path("data/processed/features")
FRAME_LABELS_DIR = Path("data/processed/frame_labels")
SPLITS_DIR = Path("data/splits")


def prune_split_files(valid_ids: set):
    split_files = [
        SPLITS_DIR / "train.txt",
        SPLITS_DIR / "val.txt",
        SPLITS_DIR / "test.txt",
    ]

    for split_file in split_files:
        if not split_file.exists():
            print(f" Missing split file: {split_file}")
            continue

        original = split_file.read_text().splitlines()
        filtered = [sid for sid in original if sid in valid_ids]

        split_file.write_text("\n".join(filtered) + "\n", encoding="utf-8")

        print(
            f" Pruned {split_file.name} | "
            f"kept={len(filtered)} removed={len(original) - len(filtered)}"
        )


def run_preprocess(split_name: str):
    print("\n PREPROCESS PIPELINE STARTED")
    print(f" Split: {split_name}")

    print("\n STEP 1 — DATA INGESTION")
    sample_index = build_sample_index(split_name)

    print("\n STEP 2 — DATA VALIDATION (STRICT)")
    clean_index = clean_sample_index(sample_index)

    print("\n STEP 2.5 — PRUNING SPLIT FILES (GLOBAL)")
    prune_split_files(set(clean_index.keys()))

    print("\n STEP 3 — FEATURE EXTRACTION")
    extract_features(
        clean_sample_index=clean_index,
        output_dir=FEATURES_DIR,
    )

    print("\n STEP 4 — FRAME LABEL GENERATION")
    build_labels_for_dataset(
        clean_sample_index=clean_index,
        features_dir=FEATURES_DIR,
        output_dir=FRAME_LABELS_DIR,
    )

    print("\n PREPROCESS PIPELINE COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run preprocessing (Steps 1–4)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
    )

    args = parser.parse_args()
    run_preprocess(args.split)
