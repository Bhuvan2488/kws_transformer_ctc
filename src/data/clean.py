from pathlib import Path
from typing import Dict


def is_invalid_annotation(annotation_path: Path) -> bool:
    content = annotation_path.read_text(encoding="utf-8")
    return "#" in content


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
            print(f"❌ Invalid annotation detected (#): {sample_id}")
            continue

        valid_samples[sample_id] = paths

    print("\n📊 STEP 2 — DATA VALIDATION REPORT")
    print(f"Total samples   : {total}")
    print(f"Valid samples   : {len(valid_samples)}")
    print(f"Invalid samples : {invalid_count}")

    if len(valid_samples) == 0:
        raise RuntimeError("All samples are invalid after STEP 2 cleanup!")

    return valid_samples


if __name__ == "__main__":
    from src.data.annotation_loader import build_sample_index

    sample_index = build_sample_index("train")
    clean_index = clean_sample_index(sample_index)

    first_key = next(iter(clean_index))
    print("\n✅ Sample retained after cleaning:")
    print(first_key, clean_index[first_key])
