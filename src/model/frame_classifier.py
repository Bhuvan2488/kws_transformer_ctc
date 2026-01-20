#src/model/frame_classifier.py
import torch
import torch.nn as nn


class FrameClassifier(nn.Module):
    def __init__(self, d_model: int, num_classes: int):
        super().__init__()
        self.proj = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
