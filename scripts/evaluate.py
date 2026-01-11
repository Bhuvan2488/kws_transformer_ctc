# scripts/evaluate.py
import os
import sys
sys.path.append(os.getcwd())

import json
from pathlib import Path

from src.evaluation.wer import wer
from src.evaluation.alignment_metrics import timestamp_errors


def load_annotation(annotation_path: Path):
    """
    Audacity annotation format:
    start_time  end_time  WORD
    """
    words = []
    with open(annotation_path, "r", encoding="utf-8") as f:
        for line in f:
            start, end, word = line.strip().split()
            words.append({
                "word": word.lower(),
                "start_time": float(start),
                "end_time": float(end),
            })
    return words


def main():
    # paths (single-file evaluation)
    annotation_path = Path("data/raw/annotations/q3g_Annotated.txt")
    prediction_path = Path("outputs/predictions/aligned_words.json")

    # load ground truth
    gt_words = load_annotation(annotation_path)

    # load predictions
    with open(prediction_path, "r", encoding="utf-8") as f:
        preds = json.load(f)

    # handle batch or single output
    if isinstance(preds, list) and isinstance(preds[0], list):
        preds = preds[0]

    # text for WER
    gt_text = [w["word"] for w in gt_words]
    pred_text = [w["word"].lower() for w in preds]

    # compute metrics
    wer_score = wer(gt_text, pred_text)
    time_metrics = timestamp_errors(gt_words, preds)

    # report
    print("\n=== Evaluation Results ===")
    print(f"WER: {wer_score:.4f}")
    print(f"Mean timestamp error: {time_metrics['mean_error']*1000:.2f} ms")
    print(f"Median timestamp error: {time_metrics['median_error']*1000:.2f} ms")


if __name__ == "__main__":
    main()

