import torch


def build_optimizer(model, lr: float = 3e-4, weight_decay: float = 1e-2):
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
