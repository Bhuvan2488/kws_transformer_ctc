#src/model/transformer_encoder.py
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, T, D)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """
    Transformer encoder stack with padding mask support.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_layers: int = 6,
        num_heads: int = 4,
        dim_ff: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: (B, T, D)
        lengths: (B,)
        """
        # Padding mask: True = PAD (to be ignored)
        B, T, _ = x.shape
        device = x.device

        pad_mask = torch.arange(T, device=device).expand(B, T) >= lengths.unsqueeze(1)

        x = self.positional_encoding(x)
        x = self.encoder(x, src_key_padding_mask=pad_mask)

        return x

