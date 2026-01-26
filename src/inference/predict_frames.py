#src/inference/predict_frames.py
from pathlib import Path
import re
import json
import torch
import numpy as np

from src.model.model import FrameAlignmentModel


FEATURES_DIR = Path("data/processed/features")
CHECKPOINT_DIR = Path("outputs/checkpoints")
PREDICTIONS_DIR = Path("outputs/predictions")
LABEL_MAP_PATH = Path("data/processed/frame_labels/label_map.json")

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


def load_model(checkpoint_path: Path, num_classes: int) -> FrameAlignmentModel:
    model = FrameAlignmentModel(num_classes=num_classes)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)

    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()

    return model


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

    label_map = json.loads(LABEL_MAP_PATH.read_text())
    num_classes = len(label_map)

    checkpoint_path = get_latest_checkpoint()
    model = load_model(checkpoint_path, num_classes)

    print(f" Using checkpoint: {checkpoint_path.name}")
    print(f" Frames           : {T}")
    print(f" Num classes      : {num_classes}")
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
    parser.add_argument(
        "--sample_id",
        type=str,
        required=True,
        help="Sample ID (without extension)",
    )

    args = parser.parse_args()
    predict_frames(args.sample_id)
