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


NUM_EPOCHS = 40
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


def masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    lengths: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    B, T, C = logits.shape

    logits = logits.view(B * T, C)
    targets = targets.view(B * T)

    mask = torch.arange(T, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
    mask = mask.reshape(B * T)

    logits = logits[mask]
    targets = targets[mask]

    return criterion(logits, targets)


def _find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    ckpts = list(checkpoint_dir.glob("model_epoch_*.pt"))
    if not ckpts:
        return None

    def extract_epoch(p: Path) -> int:
        m = re.search(r"model_epoch_(\d+)\.pt$", p.name)
        return int(m.group(1)) if m else -1

    ckpts.sort(key=extract_epoch)
    latest = ckpts[-1]
    return latest


def train():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text("")

    label_map_path = Path("data/processed/frame_labels") / "label_map.json"
    label_map = json.loads(label_map_path.read_text())
    num_classes = len(label_map)

    log("🚀 STEP 7 — TRAINING STARTED")
    log(f"Device        : {DEVICE}")
    log(f"Num classes   : {num_classes}")
    log(f"BLANK_ID      : {BLANK_ID}")
    log(f"Epochs        : {NUM_EPOCHS}")
    log(f"Batch size    : {BATCH_SIZE}")
    log(f"Learning rate : {LEARNING_RATE}")

    train_loader = build_dataloader(
        split_name="train",
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    model = FrameAlignmentModel(num_classes=num_classes)
    model.to(DEVICE)

    optimizer = build_optimizer(
        model,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = build_scheduler(optimizer)

    criterion = nn.CrossEntropyLoss()

    # ✅ AUTO-RESUME (minimal change)
    start_epoch = 1
    latest_ckpt = _find_latest_checkpoint(CHECKPOINT_DIR)
    if latest_ckpt is not None:
        ckpt = torch.load(latest_ckpt, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = int(ckpt["epoch"]) + 1
        log(f"✅ Resuming from {latest_ckpt.name} (next epoch={start_epoch})")

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_batches = 0

        for x, y, lengths in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            lengths = lengths.to(DEVICE)

            optimizer.zero_grad()

            logits = model(x, lengths)
            loss = masked_cross_entropy(logits, y, lengths, criterion)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

        avg_loss = total_loss / total_batches
        scheduler.step(avg_loss)

        lr = optimizer.param_groups[0]["lr"]

        log(f"[Epoch {epoch:03d}] loss={avg_loss:.6f} lr={lr:.6e}")

        ckpt_path = CHECKPOINT_DIR / f"model_epoch_{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "loss": avg_loss,
            },
            ckpt_path,
        )

    log("🏁 TRAINING COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    train()
