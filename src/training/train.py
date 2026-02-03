# src/training/train.py
from pathlib import Path
import json
import re
import torch
import torch.nn as nn

from src.data.dataset import build_dataloader
from src.data.label_builder import BLANK_ID
from src.model.model import FrameAlignmentModel
from src.training.optimizer import build_optimizer
from src.training.scheduler import build_scheduler


NUM_EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-2

CHECKPOINT_DIR = Path("outputs/checkpoints")
LOG_DIR = CHECKPOINT_DIR / "logs"
LOG_FILE = LOG_DIR / "train.log"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(msg: str):
    print(msg)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def masked_cross_entropy(logits, targets, lengths, criterion):
    B, T, C = logits.shape
    logits = logits.view(B * T, C)
    targets = targets.view(B * T)
    mask = torch.arange(T, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
    mask = mask.reshape(B * T)
    return criterion(logits[mask], targets[mask])


def _find_latest_checkpoint(checkpoint_dir):
    ckpts = list(checkpoint_dir.glob("model_epoch_*.pt"))
    if not ckpts:
        return None
    ckpts.sort(key=lambda p: int(re.search(r"_(\d+)\.pt$", p.name).group(1)))
    return ckpts[-1]


def train():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text("")

    label_map_path = Path("data/processed/frame_labels/label_map.json")
    label_map = json.loads(label_map_path.read_text())
    num_classes = len(label_map)

    train_loader = build_dataloader("train", BATCH_SIZE, shuffle=True)

    model = FrameAlignmentModel(num_classes=num_classes).to(DEVICE)
    optimizer = build_optimizer(model, LEARNING_RATE, WEIGHT_DECAY)
    scheduler = build_scheduler(optimizer)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 1
    latest = _find_latest_checkpoint(CHECKPOINT_DIR)
    if latest:
        ckpt = torch.load(latest, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0

        for x, y, lengths in train_loader:
            x, y, lengths = x.to(DEVICE), y.to(DEVICE), lengths.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x, lengths)
            loss = masked_cross_entropy(logits, y, lengths, criterion)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)

        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "loss": avg_loss,
                "label_map": label_map,   # 🔥 ONLY ADDITION
            },
            CHECKPOINT_DIR / f"model_epoch_{epoch}.pt",
        )


if __name__ == "__main__":
    train()
