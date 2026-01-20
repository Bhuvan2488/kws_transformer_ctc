#src/data/audio_loader.py
from pathlib import Path
import librosa
import numpy as np

AUDIO_DIR = Path("data/raw/audio")


def get_audio_path(sample_id: str) -> Path:
    audio_path = AUDIO_DIR / f"{sample_id}.mp3"

    if not audio_path.exists():
        raise FileNotFoundError(f"[AUDIO MISSING] {audio_path}")

    return audio_path


def load_audio(audio_path: Path, sr: int = 16000) -> np.ndarray:
    audio, _ = librosa.load(audio_path, sr=sr)
    return audio


if __name__ == "__main__":
    test_id = next(p.stem for p in AUDIO_DIR.glob("*.mp3") if p.name != ".gitkeep")
    path = get_audio_path(test_id)
    audio = load_audio(path)
    print(f"Loaded audio: {path}, shape={audio.shape}")
