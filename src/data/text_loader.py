from pathlib import Path

def load_transcript(transcript_path: Path) -> str:
    return transcript_path.read_text(encoding="utf-8").strip()

if __name__ == "__main__":
    txt = Path("data/raw/transcripts/sample.txt")
    print(load_transcript(txt))
