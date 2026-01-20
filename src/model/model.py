#src/model/model.py
import torch
import torch.nn as nn

from src.model.transformer_encoder import TransformerEncoder
from src.model.frame_classifier import FrameClassifier


class FrameAlignmentModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_dim: int = 80,
        d_model: int = 256,
        num_layers: int = 6,
        num_heads: int = 4,
        dim_ff: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)

        self.encoder = TransformerEncoder(
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            dim_ff=dim_ff,
            dropout=dropout,
        )

        self.classifier = FrameClassifier(
            d_model=d_model,
            num_classes=num_classes,
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.encoder(x, lengths)
        logits = self.classifier(x)
        return logits


if __name__ == "__main__":
    from src.data.dataset import build_dataloader
    import json
    from pathlib import Path

    label_map = json.loads(
        (Path("data/processed/frame_labels") / "label_map.json").read_text()
    )
    num_classes = len(label_map)

    loader = build_dataloader(
        split_name="train",
        batch_size=2,
        shuffle=False,
    )

    x, y, lengths = next(iter(loader))

    model = FrameAlignmentModel(num_classes=num_classes)
    logits = model(x, lengths)

    print("\n🔎 STEP 6 SANITY CHECK")
    print("Input x      :", x.shape)
    print("Lengths      :", lengths)
    print("Logits       :", logits.shape)
