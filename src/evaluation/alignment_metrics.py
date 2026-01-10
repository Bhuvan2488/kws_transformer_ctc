# src/evaluation/alignment_metrics.py

import numpy as np


def timestamp_errors(gt_words, pred_words):
    """
    Computes absolute timestamp errors (seconds) for aligned words.
    Assumes same word order.
    """
    errors = []

    for gt, pred in zip(gt_words, pred_words):
        start_err = abs(gt["start_time"] - pred["start_time"])
        end_err = abs(gt["end_time"] - pred["end_time"])
        errors.append((start_err + end_err) / 2)

    return {
        "mean_error": float(np.mean(errors)),
        "median_error": float(np.median(errors)),
    }
