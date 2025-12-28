import librosa
import numpy as np
from pathlib import Path

def load_audio(audio_path: Path, sr: int = 16000) -> np.ndarray:
    """
    Load audio file and return waveform
    """
    audio, _ = librosa.load(audio_path, sr=sr)
    return audio
