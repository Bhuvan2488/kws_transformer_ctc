# src/inference/ctc_decode.py

import json
import torch
from typing import List, Tuple, Dict, Union


def load_vocab(vocab_path: str) -> Dict[str, int]:
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    return vocab


def get_blank_id(vocab: Dict[str, int]) -> int:
    return vocab["<BLANK>"]


def greedy_ctc_decode(
    logits: Union[torch.Tensor, List],
    blank_id: int = 0,
) -> List[List[Tuple[int, int, int]]]:
    """
    Args:
        logits:
            (T, V) or (B, T, V) — raw logits or log-probs
        blank_id:
            CTC blank index (must match training)

    Returns:
        batch_char_spans:
            [
              [(char_id, start_frame, end_frame), ...],   # sample 1
              [(char_id, start_frame, end_frame), ...],   # sample 2
            ]
    """

    if isinstance(logits, list):
        logits = torch.tensor(logits)

    if logits.dim() == 2:
        logits = logits.unsqueeze(0)  # (1, T, V)

    assert logits.dim() == 3, "Expected logits shape (B, T, V)"

    B, T, V = logits.shape
    preds = torch.argmax(logits, dim=-1)  # (B, T)

    batch_results = []

    for b in range(B):
        prev_token = None
        start = None
        spans = []

        for t in range(T):
            token = preds[b, t].item()

            if token == prev_token:
                continue

            if prev_token is not None and prev_token != blank_id:
                spans.append((prev_token, start, t - 1))

            if token != blank_id:
                start = t
            else:
                start = None

            prev_token = token

        # flush last token
        if prev_token is not None and prev_token != blank_id:
            spans.append((prev_token, start, T - 1))

        batch_results.append(spans)

    return batch_results


# -----------------------------
# DEBUG / STANDALONE TEST
# -----------------------------
if __name__ == "__main__":
    # fake logits: (T=6, V=4)
    fake_logits = torch.tensor([
        [0.1, 3.0, 0.2, 0.1],
        [0.1, 3.2, 0.1, 0.1],
        [4.0, 0.1, 0.1, 0.1],  # blank
        [0.1, 0.1, 3.1, 0.1],
        [0.1, 0.1, 3.2, 0.1],
        [0.1, 0.1, 0.1, 3.0],
    ])

    spans = greedy_ctc_decode(fake_logits, blank_id=0)
    print(spans)
