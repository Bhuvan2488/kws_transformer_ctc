# src/inference/align.py

import json
import torch
from pathlib import Path

from src.model.model import KWSCTCModel
from src.inference.ctc_decode import greedy_ctc_decode, load_vocab
from src.inference.timestamp_extractor import extract_word_timestamps


@torch.no_grad()
def align_audio(
    model_ckpt: Path,
    logits: torch.Tensor,
    vocab_path: Path,
    output_path: Path,
):
    """
    Args:
        logits:
            (T, V) or (B, T, V)
    """

    vocab = load_vocab(vocab_path)
    id2char = {v: k for k, v in vocab.items()}
    blank_id = vocab["<BLANK>"]

    char_spans_batch = greedy_ctc_decode(logits, blank_id=blank_id)

    all_results = []

    for char_spans in char_spans_batch:
        words = extract_word_timestamps(
            char_spans,
            id2char=id2char,
            hop_length=160,
            sample_rate=16000,
        )
        all_results.append(words)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    return all_results
