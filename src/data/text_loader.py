from pathlib import Path

def is_invalid_annotation(annotation_path: Path) -> bool:
    """
    Returns True if annotation file content is exactly '#'
    """
    content = annotation_path.read_text(encoding="utf-8").strip()
    return content == "#"
