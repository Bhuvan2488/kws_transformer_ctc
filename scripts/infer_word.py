# scripts/infer_word.py
import os
import sys
sys.path.append(os.getcwd())

from pathlib import Path
import argparse
import json
import torch
import numpy as np

from src.data.feature_extraction import extract_logmel, normalize, SAMPLE_RATE, N_MELS
from src.data.audio_loader import load_audio
from src.model.model import FrameAlignmentModel
from src.inference.word_timestamp_extractor import extract_word_segments
from src.inference.timestamp_extractor import segment_frames_to_times

LABEL_MAP_PATH = Path("data/processed/frame_labels/label_map.json")
CHECKPOINT_DIR = Path("outputs/checkpoints")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_label_maps():
    label_map = json.loads(LABEL_MAP_PATH.read_text())
    word_to_id = {k: int(v) for k, v in label_map.items()}
    id_to_word = {int(v): k for k, v in label_map.items()}
    return word_to_id, id_to_word


def get_latest_checkpoint():
    ckpts = sorted(
        CHECKPOINT_DIR.glob("model_epoch_*.pt"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    if not ckpts:
        raise RuntimeError("No checkpoints found")
    return ckpts[-1]


def load_model(num_classes: int):
    model = FrameAlignmentModel(num_classes=num_classes)
    ckpt = torch.load(get_latest_checkpoint(), map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()
    return model


def extract_features_from_audio(audio_path: Path) -> np.ndarray:
    audio = load_audio(audio_path, sr=SAMPLE_RATE)
    features = extract_logmel(audio)
    features = normalize(features)

    if features.ndim != 2 or features.shape[1] != N_MELS:
        raise RuntimeError(f"Invalid feature shape: {features.shape}")

    return features.astype(np.float32)


def infer_word(audio_path: Path, target_word: str):
    word_to_id, id_to_word = load_label_maps()

    if target_word not in word_to_id:
        print(f" Word '{target_word}' not in training vocabulary")
        return

    features = extract_features_from_audio(audio_path)
    T = features.shape[0]

    x = torch.from_numpy(features).unsqueeze(0).to(DEVICE)
    lengths = torch.tensor([T], dtype=torch.long).to(DEVICE)

    model = load_model(num_classes=len(word_to_id))

    with torch.no_grad():
        logits = model(x, lengths)
        frame_preds = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy()

    segments = extract_word_segments(frame_preds)
    target_id = word_to_id[target_word]

    matches = []
    for seg in segments:
        if seg["label_id"] == target_id:
            start_t, end_t = segment_frames_to_times(
                seg["start_frame"], seg["end_frame"]
            )
            matches.append(
                {
                    "word": target_word,
                    "start_time": round(start_t, 3),
                    "end_time": round(end_t, 3),
                }
            )

    if not matches:
        print(f" Word '{target_word}' NOT found in audio")
        return

    print("\n WORD ALIGNMENT RESULT")
    for m in matches:
        print(json.dumps(m, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Infer word timestamp from audio using trained Transformer"
    )
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file")
    parser.add_argument("--word", type=str, required=True, help="Target word")

    args = parser.parse_args()
    infer_word(Path(args.audio), args.word)
