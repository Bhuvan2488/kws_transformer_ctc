import torch
import torch.nn as nn


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        model_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, model_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

    def forward(self, x, lengths):
        x = self.input_proj(x)

        max_len = x.size(1)
        mask = torch.arange(max_len, device=lengths.device)[None, :] >= lengths[:, None]

        x = self.encoder(x, src_key_padding_mask=mask)
        return x
