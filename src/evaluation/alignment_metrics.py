# src/evaluation/alignment_metrics.py
from pathlib import Path
from typing import Dict, List, Tuple
import json
import statistics

from src.data.annotation_loader import load_split_ids

PREDICTIONS_PATH = Path("outputs/predictions/aligned_words.json")
ANNOTATION_DIR = Path("data/raw/annotations")
OUTPUT_REPORT = Path("outputs/predictions/eval_report.json")

def load_predictions() -> List[Dict]:
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"[PREDICTIONS MISSING] {PREDICTIONS_PATH}")
    return json.loads(PREDICTIONS_PATH.read_text(encoding="utf-8"))

def load_ground_truth(sample_id: str) -> List[Tuple[str, float, float]]:
    ann_path = ANNOTATION_DIR / f"{sample_id}_Annotated.txt"
    gt = []
    for line in ann_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        s, e, w = line.split("\t")
        gt.append((w, float(s), float(e)))
    return gt

def find_best_match(word, pred_start, gt_segments, used):
    candidates = [
        (i, gt)
        for i, gt in enumerate(gt_segments)
        if gt[0] == word and i not in used
    ]
    if not candidates:
        return None, None
    return min(candidates, key=lambda x: abs(x[1][1] - pred_start))

def evaluate():
    print("\n STEP 10 — TEST SET EVALUATION")

    test_ids = set(load_split_ids("test"))
    predictions = load_predictions()

    predictions = [p for p in predictions if p["sample_id"] in test_ids]

    start_err, end_err = [], []
    total_gt = 0
    matched = 0

    grouped: Dict[str, List[Dict]] = {}
    for p in predictions:
        grouped.setdefault(p["sample_id"], []).append(p)

    for sample_id, preds in grouped.items():
        gt = load_ground_truth(sample_id)
        total_gt += len(gt)
        used = set()

        for p in preds:
            idx, match = find_best_match(
                p["word"], p["start_time"], gt, used
            )
            if match is None:
                continue

            used.add(idx)
            matched += 1
            _, gs, ge = match

            start_err.append(abs(p["start_time"] - gs))
            end_err.append(abs(p["end_time"] - ge))

    def mean(x): return sum(x) / len(x) if x else None
    def median(x): return statistics.median(x) if x else None

    report = {
        "test_samples": len(test_ids),
        "total_gt_words": total_gt,
        "matched_words": matched,
        "word_detection_recall": matched / total_gt if total_gt else None,
        "mean_start_error_sec": mean(start_err),
        "mean_end_error_sec": mean(end_err),
        "median_start_error_sec": median(start_err),
        "median_end_error_sec": median(end_err),
    }

    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT.write_text(json.dumps(report, indent=2))

    print("\n TEST SET EVALUATION COMPLETED")
    print(json.dumps(report, indent=2))

    return report

if __name__ == "__main__":
    evaluate()
