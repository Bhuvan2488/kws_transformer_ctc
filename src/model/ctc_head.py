import torch.nn as nn


class CTCHead(nn.Module):
    def __init__(self, model_dim: int, vocab_size: int):
        super().__init__()
        self.fc = nn.Linear(model_dim, vocab_size)

    def forward(self, x):
        """
        x: (B, T, D)
        returns: (B, T, vocab)
        """
        return self.fc(x)
