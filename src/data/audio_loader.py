import librosa
import numpy as np
from pathlib import Path

def load_audio(audio_path: Path, sr: int = 16000) -> np.ndarray:
    audio, _ = librosa.load(audio_path, sr=sr)
    return audio

if __name__ == "__main__":
    audio = load_audio(Path("data/raw/audio/sample.wav"))
    print(audio.shape)
