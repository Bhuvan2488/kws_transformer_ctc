# src/inference/predict_frames.py
from pathlib import Path
import re
import torch
import numpy as np

from src.model.model import FrameAlignmentModel


FEATURES_DIR = Path("data/processed/features")
CHECKPOINT_DIR = Path("outputs/checkpoints")
PREDICTIONS_DIR = Path("outputs/predictions")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_latest_checkpoint() -> Path:
    ckpts = list(CHECKPOINT_DIR.glob("model_epoch_*.pt"))
    if not ckpts:
        raise FileNotFoundError("[NO CHECKPOINT FOUND] outputs/checkpoints/")

    def epoch_num(p: Path) -> int:
        m = re.search(r"model_epoch_(\d+)\.pt", p.name)
        return int(m.group(1)) if m else -1

    ckpts.sort(key=epoch_num)
    return ckpts[-1]


def load_model(checkpoint_path: Path):
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)

    num_classes = ckpt["num_classes"]
    label_map = ckpt["label_map"]

    model = FrameAlignmentModel(num_classes=num_classes)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()

    return model, label_map


def predict_frames(sample_id: str) -> Path:
    feature_path = FEATURES_DIR / f"{sample_id}.npy"
    if not feature_path.exists():
        raise FileNotFoundError(f"[FEATURE FILE MISSING] {feature_path}")

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    features = np.load(feature_path)
    if features.ndim != 2 or features.shape[1] != 80:
        raise RuntimeError(f"[INVALID FEATURE SHAPE] {sample_id}: {features.shape}")

    T = features.shape[0]

    x = torch.from_numpy(features).float().unsqueeze(0).to(DEVICE)
    lengths = torch.tensor([T], dtype=torch.long).to(DEVICE)

    checkpoint_path = get_latest_checkpoint()
    model, label_map = load_model(checkpoint_path)

    print(f" Using checkpoint: {checkpoint_path.name}")
    print(f" Frames           : {T}")
    print(f" Num classes      : {len(label_map)}")
    print(f" Device           : {DEVICE}")

    with torch.no_grad():
        logits = model(x, lengths)
        preds = torch.argmax(logits, dim=-1)

    preds = preds.squeeze(0).cpu().numpy().astype(np.int64)

    if preds.shape[0] != T:
        raise RuntimeError(f"[PREDICTION LENGTH MISMATCH] {preds.shape[0]} != {T}")

    out_path = PREDICTIONS_DIR / f"frame_preds_{sample_id}.npy"
    np.save(out_path, preds)

    print(f" Frame predictions saved: {out_path}")
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="STEP 8 — Frame-level inference")
    parser.add_argument("--sample_id", type=str, required=True)

    args = parser.parse_args()
    predict_frames(args.sample_id)
