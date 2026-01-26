# src/data/clean.py
from pathlib import Path
from typing import Dict


def is_invalid_annotation(annotation_path: Path) -> bool:
    lines = annotation_path.read_text(encoding="utf-8").splitlines()

    for line_num, line in enumerate(lines, 1):
        line = line.strip()

        if not line:
            continue

        # Rule 1: reject comments
        if "#" in line:
            print(f" '#' detected | {annotation_path.name}:{line_num}")
            return True

        parts = line.split("\t")
        if len(parts) != 3:
            print(f" Bad format | {annotation_path.name}:{line_num} → {line}")
            return True

        start, end, word = parts

        # Rule 2: timestamps must be numeric
        try:
            start = float(start)
            end = float(end)
        except ValueError:
            print(f" Non-numeric timestamp | {annotation_path.name}:{line_num}")
            return True

        # Rule 3: valid time range
        if start < 0 or end <= start:
            print(f" Invalid time range | {annotation_path.name}:{line_num}")
            return True

    return False


def clean_sample_index(
    sample_index: Dict[str, Dict[str, Path]]
) -> Dict[str, Dict[str, Path]]:
    total = len(sample_index)
    valid_samples = {}
    invalid_count = 0

    for sample_id, paths in sample_index.items():
        ann_path = paths["annotation_path"]

        if is_invalid_annotation(ann_path):
            invalid_count += 1
            print(f" Removed invalid sample: {sample_id}")
            continue

        valid_samples[sample_id] = paths

    print("\n STEP 2 — DATA VALIDATION REPORT")
    print(f"Total samples   : {total}")
    print(f"Valid samples   : {len(valid_samples)}")
    print(f"Invalid samples : {invalid_count}")

    if not valid_samples:
        raise RuntimeError(" All samples are invalid after cleanup!")

    return valid_samples


if __name__ == "__main__":
    from src.data.annotation_loader import build_sample_index

    idx = build_sample_index("train")
    clean_idx = clean_sample_index(idx)

    k = next(iter(clean_idx))
    print("\n Example valid sample:")
    print(k, clean_idx[k])
