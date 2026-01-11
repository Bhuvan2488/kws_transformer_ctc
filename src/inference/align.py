# src/inference/align.py
import os
import sys
sys.path.append(os.getcwd())

import json
import torch
from pathlib import Path

from src.inference.ctc_decode import greedy_ctc_decode
from src.inference.timestamp_extractor import extract_word_timestamps


@torch.no_grad()
def align_from_logits(
    logits: torch.Tensor,
    vocab: dict,
    output_path: Path,
):
    id2char = {v: k for k, v in vocab.items()}
    blank_id = vocab["<BLANK>"]

    char_spans_batch = greedy_ctc_decode(logits, blank_id=blank_id)

    results = []
    for char_spans in char_spans_batch:
        words = extract_word_timestamps(
            char_spans,
            id2char=id2char,
            hop_length=160,
            sample_rate=16000,
        )
        results.append(words)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results


# --------------------------------------------------
# STANDALONE TEST
# --------------------------------------------------
if __name__ == "__main__":
    # fake vocab
    vocab = {
        "<BLANK>": 0,
        "h": 1,
        "i": 2,
        " ": 3,
        "t": 4,
        "e": 5,
        "r": 6,
    }

    # fake logits: (T=10, V=7)
    fake_logits = torch.tensor([
        [0, 5, 0, 0, 0, 0, 0],  # h
        [0, 6, 0, 0, 0, 0, 0],
        [0, 0, 6, 0, 0, 0, 0],  # i
        [5, 0, 0, 0, 0, 0, 0],  # blank
        [0, 0, 0, 6, 0, 0, 0],  # space
        [0, 0, 0, 0, 6, 0, 0],  # t
        [0, 0, 0, 0, 0, 6, 0],  # e
        [0, 0, 0, 0, 0, 0, 6],  # r
        [0, 0, 0, 0, 0, 6, 0],  # e
        [5, 0, 0, 0, 0, 0, 0],  # blank
    ])

    fake_logits = fake_logits.unsqueeze(0)  # (1, T, V)

    output = align_from_logits(
        logits=fake_logits,
        vocab=vocab,
        output_path=Path("outputs/predictions/aligned_words.json"),
    )

    print(json.dumps(output, indent=2))

