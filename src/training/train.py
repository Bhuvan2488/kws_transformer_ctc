import sys, os
sys.path.append(os.getcwd())

from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader

from src.data.dataset import SpeechDataset, collate_fn
from src.model.model import KWSCTCModel
from src.training.optimizer import build_optimizer
from src.training.scheduler import build_scheduler


# ============================================================
# CRITICAL: WINDOWS CPU SAFETY (PREVENT BACKWARD FREEZE)
# ============================================================
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 8
EPOCHS = 20
LR = 3e-4
WEIGHT_DECAY = 1e-2
WARMUP_STEPS = 0

FEATURE_DIM = 80
VOCAB_PATH = Path("data/processed/tokenize_text/vocab.json")

TRAIN_SPLIT = Path("data/splits/train.txt")
VAL_SPLIT = Path("data/splits/val.txt")

FEATURES_DIR = Path("data/processed/features")
TOKENS_DIR = Path("data/processed/tokenize_text")

CKPT_DIR = Path("outputs/checkpoints")
LOG_DIR = Path("outputs/logs")

CKPT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# UTILS
# ============================================================
def load_vocab_size():
    print(">>> Loading vocab", flush=True)
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    print(f">>> Vocab size: {len(vocab)}", flush=True)
    return len(vocab)


# ============================================================
# TRAIN / VALIDATE
# ============================================================
def train_one_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0.0

    print(">>> Entered training loop", flush=True)

    for step, batch in enumerate(loader):
        if step == 0:
            print(">>> First training batch loaded", flush=True)

        audio = batch["audio"].to(DEVICE)
        tokens = batch["tokens"].to(DEVICE)
        audio_lengths = batch["audio_lengths"].to(DEVICE)
        token_lengths = batch["token_lengths"].to(DEVICE)

        loss = model(
            audio=audio,
            audio_lengths=audio_lengths,
            tokens=tokens,
            token_lengths=token_lengths,
        )

        optimizer.zero_grad()
        loss.backward()  # SAFE NOW
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        if scheduler:
            scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader):
    model.eval()
    total_loss = 0.0

    print(">>> Entered validation loop", flush=True)

    for step, batch in enumerate(loader):
        if step == 0:
            print(">>> First validation batch loaded", flush=True)

        audio = batch["audio"].to(DEVICE)
        tokens = batch["tokens"].to(DEVICE)
        audio_lengths = batch["audio_lengths"].to(DEVICE)
        token_lengths = batch["token_lengths"].to(DEVICE)

        loss = model(
            audio=audio,
            audio_lengths=audio_lengths,
            tokens=tokens,
            token_lengths=token_lengths,
        )

        total_loss += loss.item()

    return total_loss / len(loader)


# ============================================================
# MAIN
# ============================================================
def main():
    print(">>> TRAINING STARTED", flush=True)

    vocab_size = load_vocab_size()

    train_ds = SpeechDataset(TRAIN_SPLIT, FEATURES_DIR, TOKENS_DIR)
    val_ds = SpeechDataset(VAL_SPLIT, FEATURES_DIR, TOKENS_DIR)

    print(f">>> Train samples: {len(train_ds)}", flush=True)
    print(f">>> Val samples: {len(val_ds)}", flush=True)

    # WINDOWS-SAFE DATALOADERS
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False,
    )

    # 🔥 TEMPORARILY REDUCED MODEL (STABILITY)
    model = KWSCTCModel(
        input_dim=FEATURE_DIM,
        vocab_size=vocab_size,
        model_dim=128,   # reduced from 256
    ).to(DEVICE)

    optimizer = build_optimizer(model, LR, WEIGHT_DECAY)
    scheduler = build_scheduler(optimizer, WARMUP_STEPS)

    log_file = LOG_DIR / "train.log"

    for epoch in range(1, EPOCHS + 1):
        print(f"\n=== EPOCH {epoch} START ===", flush=True)

        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler)
        val_loss = validate(model, val_loader)

        msg = f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        print(msg, flush=True)

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

        ckpt_path = CKPT_DIR / f"model_epoch_{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            },
            ckpt_path,
        )

    print(">>> TRAINING FINISHED", flush=True)


if __name__ == "__main__":
    main()
