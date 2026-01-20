#src/data/feature_extraction.py
from pathlib import Path
from typing import Dict
import numpy as np
import librosa

from src.data.audio_loader import load_audio


SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 80
EPS = 1e-8


def extract_logmel(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    n_mels: int = N_MELS,
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
    return logmel.T


def normalize(features: np.ndarray) -> np.ndarray:
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    return (features - mean) / (std + EPS)


def process_sample(
    sample_id: str,
    audio_path: Path,
    output_dir: Path,
) -> None:
    audio = load_audio(audio_path, sr=SAMPLE_RATE)

    if audio.size == 0:
        raise RuntimeError(f"[EMPTY AUDIO] {audio_path}")

    features = extract_logmel(audio)
    features = normalize(features)

    if features.ndim != 2 or features.shape[1] != N_MELS:
        raise RuntimeError(f"[INVALID FEATURE SHAPE] {sample_id}: {features.shape}")

    features = features.astype(np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{sample_id}.npy"
    np.save(out_path, features)

    loaded = np.load(out_path)
    if loaded.shape != features.shape:
        raise RuntimeError(f"[SAVE ERROR] Shape mismatch for {out_path}")

    print(f"✅ Features saved: {out_path} | shape={features.shape}")


def extract_features(
    clean_sample_index: Dict[str, Dict[str, Path]],
    output_dir: Path,
) -> None:
    print("\n🔊 STEP 3 — FEATURE EXTRACTION STARTED")
    print(f"Samples to process: {len(clean_sample_index)}")

    for sample_id, paths in clean_sample_index.items():
        audio_path = paths["audio_path"]

        if audio_path.suffix.lower() != ".mp3":
            raise RuntimeError(
                f"[INVALID AUDIO FORMAT] {audio_path} (only .mp3 allowed)"
            )

        process_sample(sample_id, audio_path, output_dir)

    print("\n📦 STEP 3 COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    from src.data.annotation_loader import build_sample_index
    from src.data.clean import clean_sample_index

    FEATURES_DIR = Path("data/processed/features")

    sample_index = build_sample_index("train")
    clean_index = clean_sample_index(sample_index)
    extract_features(clean_index, FEATURES_DIR)

