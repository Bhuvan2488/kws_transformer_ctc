import os
import sys
sys.path.append(os.getcwd())

from pathlib import Path
from src.data.text_loader import is_invalid_annotation

def cleanup_invalid_samples(
    audio_dir: Path,
    transcript_dir: Path,
    annotation_dir: Path,
):
    for ann_file in annotation_dir.glob("*_Annotated.txt"):
        if is_invalid_annotation(ann_file):
            base = ann_file.stem.replace("_Annotated", "")

            audio = audio_dir / f"{base}.mp3"
            transcript = transcript_dir / f"{base}.txt"

            ann_file.unlink(missing_ok=True)
            audio.unlink(missing_ok=True)
            transcript.unlink(missing_ok=True)

            print(f"Deleted invalid sample: {base}")

if __name__ == "__main__":
    cleanup_invalid_samples(
        Path("data/raw/audio"),
        Path("data/raw/transcripts"),
        Path("data/raw/annotations"),
    )
