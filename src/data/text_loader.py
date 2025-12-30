from pathlib import Path
from typing import List, Dict
import json

CTC_BLANK = "<BLANK>"  # ID = 0


def load_transcript(transcript_path: Path) -> str:
    """Load transcript text."""
    return transcript_path.read_text(encoding="utf-8").strip()


def build_vocab(transcripts: List[str]) -> Dict[str, int]:
    """Build character-level vocab with CTC blank at index 0."""
    chars = sorted(set("".join(transcripts)))
    vocab = {CTC_BLANK: 0}
    for i, ch in enumerate(chars, start=1):
        vocab[ch] = i
    return vocab


def encode_text(text: str, vocab: Dict[str, int]) -> List[int]:
    """Encode text into character IDs."""
    return [vocab[ch] for ch in text if ch in vocab]


def tokenize_transcripts(
    transcript_dir: Path,
    output_dir: Path,
    vocab_path: Path,
):
    """Tokenize all transcripts and save encoded outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    transcript_files = sorted(transcript_dir.glob("*.txt"))
    transcripts = [load_transcript(f) for f in transcript_files]

    vocab = build_vocab(transcripts)

    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    vocab_path.write_text(json.dumps(vocab, indent=2), encoding="utf-8")

    for f in transcript_files:
        token_ids = encode_text(load_transcript(f), vocab)
        out_path = output_dir / f"{f.stem}.json"
        out_path.write_text(json.dumps(token_ids), encoding="utf-8")


if __name__ == "__main__":
    sample = Path("data/raw/transcripts/sample.txt")
    if sample.exists():
        print(load_transcript(sample))

    tokenize_transcripts(
        transcript_dir=Path("data/raw/transcripts"),
        output_dir=Path("data/processed/tokenize_text"),
        vocab_path=Path("data/processed/tokenize_text/vocab.json"),
    )
