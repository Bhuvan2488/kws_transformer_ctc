from pathlib import Path

def is_invalid_annotation(annotation_path: Path) -> bool:
    content = annotation_path.read_text(encoding="utf-8")
    return "#" in content

if __name__ == "__main__":
    ann = Path("data/raw/annotations/sample_Annotated.txt")
    print(is_invalid_annotation(ann))
