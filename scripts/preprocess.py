#scripts/preprocess.py
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


def run_preprocess(split_name: str) -> None:
    print("\n🚀 PREPROCESS PIPELINE STARTED")
    print(f"📂 Split: {split_name}")

    print("\n📥 STEP 1 — DATA INGESTION")
    sample_index = build_sample_index(split_name)

    print("\n🧹 STEP 2 — DATA VALIDATION")
    clean_index = clean_sample_index(sample_index)

    print("\n🔊 STEP 3 — FEATURE EXTRACTION")
    extract_features(
        clean_sample_index=clean_index,
        output_dir=FEATURES_DIR,
    )

    print("\n🏷️ STEP 4 — FRAME LABEL GENERATION")
    build_labels_for_dataset(
        clean_sample_index=clean_index,
        features_dir=FEATURES_DIR,
        output_dir=FRAME_LABELS_DIR,
    )

    print("\n✅ PREPROCESS PIPELINE COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run preprocessing (Steps 1–4) for KWS alignment model"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split to preprocess",
    )

    args = parser.parse_args()
    run_preprocess(args.split)



