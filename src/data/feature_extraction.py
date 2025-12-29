import os
import sys
sys.path.append(os.getcwd())

from pathlib import Path
import numpy as np
import librosa

from src.data.audio_loader import load_audio


def extract_logmel(
    audio: np.ndarray,
    sr: int = 16000,
    n_fft: int = 400,
    hop_length: int = 160,
    n_mels: int = 80,
) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    logmel = librosa.power_to_db(mel, ref=np.max)
    return logmel.T  # (time, features)


def normalize(features: np.ndarray) -> np.ndarray:
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True) + 1e-8
    return (features - mean) / std


def process_audio_file(audio_path: Path, output_dir: Path) -> None:
    audio = load_audio(audio_path)
    features = extract_logmel(audio)
    features = normalize(features)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"{audio_path.stem}.npy", features)


def process_directory(
    audio_dir: Path,
    output_dir: Path,
    extensions=(".wav", ".mp3", ".flac"),
) -> None:
    for audio_path in audio_dir.iterdir():
        if audio_path.suffix.lower() in extensions:
            process_audio_file(audio_path, output_dir)


if __name__ == "__main__":
    AUDIO_DIR = Path("data/raw/audio")
    OUTPUT_DIR = Path("data/processed/features")

    process_directory(AUDIO_DIR, OUTPUT_DIR)
    print("Feature extraction completed.")

