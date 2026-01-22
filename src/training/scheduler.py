#src/training/scheduler.py
from torch.optim.lr_scheduler import ReduceLROnPlateau


def build_scheduler(
    optimizer,
    factor: float = 0.5,
    patience: int = 5,
    min_lr: float = 1e-6,
):
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=factor,
        patience=patience,
        min_lr=min_lr
    )
    return scheduler
