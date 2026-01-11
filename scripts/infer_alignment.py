# scripts/infer_alignment.py

import sys, os
sys.path.append(os.getcwd())

import argparse
import json
from pathlib import Path

import torch
import numpy as np

from src.data.audio_loader import load_audio
from src.data.feature_extraction import extract_logmel, normalize
from src.model.model import KWSCTCModel
from src.inference.ctc_decode import greedy_ctc_decode, load_vocab
from src.inference.timestamp_extractor import extract_word_timestamps


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def run_inference(
    audio_path: Path,
    checkpoint_path: Path,
    vocab_path: Path,
    output_path: Path,
):
    # ----------------------------
    # Load vocab
    # ----------------------------
    vocab = load_vocab(vocab_path)
    id2char = {v: k for k, v in vocab.items()}
    blank_id = vocab["<BLANK>"]
    vocab_size = len(vocab)

    # ----------------------------
    # Load audio → features
    # ----------------------------
    audio = load_audio(audio_path)
    features = extract_logmel(audio)
    features = normalize(features)

    features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    lengths = torch.tensor([features.shape[1]], dtype=torch.long)

    features = features.to(DEVICE)
    lengths = lengths.to(DEVICE)

    # ----------------------------
    # Load model
    # ----------------------------
    model = KWSCTCModel(
        input_dim=features.shape[-1],
        vocab_size=vocab_size,
        model_dim=128,  # MUST match training
    ).to(DEVICE)

    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ----------------------------
    # Forward pass → logits
    # ----------------------------
    logits = model.forward_logits(
        audio=features,
        audio_lengths=lengths,
    )  # (B, T, V)

    # ----------------------------
    # CTC decoding
    # ----------------------------
    char_spans_batch = greedy_ctc_decode(logits, blank_id=blank_id)
    char_spans = char_spans_batch[0]

    # ----------------------------
    # Word-level timestamps
    # ----------------------------
    words = extract_word_timestamps(
        char_spans,
        id2char=id2char,
        hop_length=160,
        sample_rate=16000,
    )

    # ----------------------------
    # Save output
    # ----------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(words, f, indent=2, ensure_ascii=False)

    return words


def main():
    parser = argparse.ArgumentParser("CTC Forced Alignment Inference")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, default=Path("data/processed/tokenize_text/vocab.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/predictions/aligned_words.json"))

    args = parser.parse_args()

    words = run_inference(
        audio_path=args.audio,
        checkpoint_path=args.checkpoint,
        vocab_path=args.vocab,
        output_path=args.output,
    )

    print("\nAligned words:")
    for w in words:
        print(w)


if __name__ == "__main__":
    main()

