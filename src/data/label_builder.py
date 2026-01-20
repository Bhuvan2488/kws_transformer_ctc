#src/data/label_builder.py
from pathlib import Path
from typing import Dict, Tuple
import json
import numpy as np
import math

SAMPLE_RATE = 16000
HOP_LENGTH = 160

BLANK_LABEL = "BLANK"
BLANK_ID = 0


def parse_annotation_file(annotation_path: Path) -> list[Tuple[float, float, str]]:
    segments = []

    for line_num, line in enumerate(
        annotation_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        line = line.strip()
        if not line:
            continue

        parts = line.split("\t")
        if len(parts) != 3:
            raise RuntimeError(
                f"[INVALID ANNOTATION FORMAT] {annotation_path} line {line_num}"
            )

        start, end, word = parts
        segments.append((float(start), float(end), word))

    return segments


def time_to_frame(time_sec: float) -> int:
    return int(math.floor(time_sec * SAMPLE_RATE / HOP_LENGTH))


def build_frame_labels(
    segments: list[Tuple[float, float, str]],
    num_frames: int,
    label_map: Dict[str, int],
) -> np.ndarray:
    labels = np.full(shape=(num_frames,), fill_value=BLANK_ID, dtype=np.int64)

    for start_sec, end_sec, word in segments:
        if word not in label_map:
            label_map[word] = len(label_map)

        word_id = label_map[word]

        start_frame = time_to_frame(start_sec)
        end_frame = int(math.ceil(end_sec * SAMPLE_RATE / HOP_LENGTH))

        start_frame = max(0, start_frame)
        end_frame = min(num_frames, end_frame)

        for f in range(start_frame, end_frame):
            labels[f] = word_id

    return labels


def build_labels_for_dataset(
    clean_sample_index: Dict[str, Dict[str, Path]],
    features_dir: Path,
    output_dir: Path,
) -> None:
    print("\n🏷️ STEP 4 — FRAME LABEL GENERATION STARTED")

    output_dir.mkdir(parents=True, exist_ok=True)

    label_map: Dict[str, int] = {BLANK_LABEL: BLANK_ID}
    total_samples = len(clean_sample_index)

    for sample_id, paths in clean_sample_index.items():
        feature_path = features_dir / f"{sample_id}.npy"
        annotation_path = paths["annotation_path"]

        if not feature_path.exists():
            raise FileNotFoundError(f"[FEATURE FILE MISSING] {feature_path}")

        features = np.load(feature_path)
        if features.ndim != 2:
            raise RuntimeError(f"[INVALID FEATURE SHAPE] {feature_path}")

        T = features.shape[0]

        segments = parse_annotation_file(annotation_path)
        frame_labels = build_frame_labels(segments, T, label_map)

        if len(frame_labels) != T:
            raise RuntimeError(
                f"[FRAME LABEL LENGTH MISMATCH] {sample_id}: {len(frame_labels)} != {T}"
            )

        out_path = output_dir / f"{sample_id}.npy"
        np.save(out_path, frame_labels)

        print(f"✅ Labels saved: {out_path} | frames={T}")

    label_map_path = output_dir / "label_map.json"
    with label_map_path.open("w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)

    print("\n📊 STEP 4 SUMMARY")
    print(f"Total samples processed : {total_samples}")
    print(f"Total labels (incl BLANK): {len(label_map)}")
    print(f"Label map saved to       : {label_map_path}")
    print("🏁 STEP 4 COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    from src.data.annotation_loader import build_sample_index
    from src.data.clean import clean_sample_index

    FEATURES_DIR = Path("data/processed/features")
    FRAME_LABELS_DIR = Path("data/processed/frame_labels")

    sample_index = build_sample_index("train")
    clean_index = clean_sample_index(sample_index)

    build_labels_for_dataset(
        clean_sample_index=clean_index,
        features_dir=FEATURES_DIR,
        output_dir=FRAME_LABELS_DIR,
    )
