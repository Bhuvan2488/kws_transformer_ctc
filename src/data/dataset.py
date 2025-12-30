from pathlib import Path


def is_invalid_annotation(annotation_path: Path) -> bool:
    """
    Returns True if '#' is present anywhere in annotation file
    """
    content = annotation_path.read_text(encoding="utf-8")
    return "#" in content


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
