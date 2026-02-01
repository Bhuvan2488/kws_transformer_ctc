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


def get_latest_checkpoint():
    ckpts = list(CHECKPOINT_DIR.glob("model_epoch_*.pt"))
    ckpts.sort(key=lambda p: int(re.search(r"_(\d+)\.pt$", p.name).group(1)))
    return ckpts[-1]


def predict_frames(sample_id: str):
    features = np.load(FEATURES_DIR / f"{sample_id}.npy")
    T = features.shape[0]

    x = torch.from_numpy(features).float().unsqueeze(0).to(DEVICE)
    lengths = torch.tensor([T], dtype=torch.long).to(DEVICE)

    ckpt = torch.load(get_latest_checkpoint(), map_location=DEVICE)
    label_map = ckpt["label_map"]
    num_classes = len(label_map)

    model = FrameAlignmentModel(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        preds = torch.argmax(model(x, lengths), dim=-1)

    out = preds.squeeze(0).cpu().numpy()
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PREDICTIONS_DIR / f"frame_preds_{sample_id}.npy", out)
