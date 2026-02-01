#src/data/annotation_loader.py
from pathlib import Path
from typing import Dict

ANNOTATION_DIR = Path("data/raw/annotations")
SPLIT_DIR = Path("data/splits")


def get_annotation_path(sample_id: str) -> Path:
    ann_path = ANNOTATION_DIR / f"{sample_id}_Annotated.txt"

    if not ann_path.exists():
        raise FileNotFoundError(f"[ANNOTATION MISSING] {ann_path}")

    return ann_path


def load_split_ids(split_name: str) -> list[str]:
    split_file = SPLIT_DIR / f"{split_name}.txt"

    if not split_file.exists():
        raise FileNotFoundError(f"[SPLIT FILE MISSING] {split_file}")

    ids = [
        line.strip()
        for line in split_file.read_text().splitlines()
        if line.strip()
    ]

    if not ids:
        raise RuntimeError(f"[EMPTY SPLIT FILE] {split_file}")

    return ids



def build_sample_index(split_name: str) -> Dict[str, Dict[str, Path]]:
    from src.data.audio_loader import get_audio_path

    sample_index = {}
    sample_ids = load_split_ids(split_name)

    for sample_id in sample_ids:
        audio_path = get_audio_path(sample_id)
        ann_path = get_annotation_path(sample_id)

        sample_index[sample_id] = {
            "audio_path": audio_path,
            "annotation_path": ann_path
        }

    print(f" Loaded {len(sample_index)} samples for split='{split_name}'")
    return sample_index


if __name__ == "__main__":
    index = build_sample_index("train")
    first_key = next(iter(index))
    print(first_key, index[first_key])

