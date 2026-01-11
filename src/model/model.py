import os
import sys
sys.path.append(os.getcwd())

import torch
import torch.nn as nn


from src.model.transformer_encoder import TransformerEncoder
from src.model.ctc_head import CTCHead


class KWSCTCModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        vocab_size: int,
        model_dim: int = 256,
    ):
        super().__init__()

        self.encoder = TransformerEncoder(
            input_dim=input_dim,
            model_dim=model_dim,
        )

        self.ctc_head = CTCHead(
            model_dim=model_dim,
            vocab_size=vocab_size,
        )

        self.ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

    def forward(
        self,
        audio,
        audio_lengths,
        tokens=None,
        token_lengths=None,
    ):
        """
        audio: (B, T, F)
        audio_lengths: (B,)
        tokens: (B, L) or None
        token_lengths: (B,) or None
        """

        enc_out = self.encoder(audio, audio_lengths)
        logits = self.ctc_head(enc_out)          # (B, T, V)
        log_probs = logits.log_softmax(dim=-1)   # required for CTC

        if tokens is None:
            return log_probs

        # CTC expects (T, B, V)
        log_probs = log_probs.transpose(0, 1)

        loss = self.ctc_loss(
            log_probs,
            tokens,
            audio_lengths,
            token_lengths,
        )

        return loss
