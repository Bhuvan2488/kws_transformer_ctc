from pathlib import Path
from src.data.text_loader import is_invalid_annotation

def cleanup_invalid_samples(
    audio_dir: Path,
    transcript_dir: Path,
    annotation_dir: Path,
):
    """
    Deletes annotation, audio, and transcript files
    if annotation content is '#'
    """

    for ann_file in annotation_dir.glob("*_Annotated.txt"):
        if is_invalid_annotation(ann_file):
            base_name = ann_file.name.replace("_Annotated.txt", "")

            audio_file = audio_dir / f"{base_name}.wav"
            transcript_file = transcript_dir / f"{base_name}.txt"

            # delete files if they exist
            if ann_file.exists():
                ann_file.unlink()

            if audio_file.exists():
                audio_file.unlink()

            if transcript_file.exists():
                transcript_file.unlink()
