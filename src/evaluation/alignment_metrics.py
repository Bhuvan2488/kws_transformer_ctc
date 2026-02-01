#src/evaluation/alignment_metrics.py
from pathlib import Path
from typing import Dict, List, Tuple
import json
import statistics
import matplotlib.pyplot as plt

PREDICTIONS_PATH = Path("outputs/predictions/aligned_words.json")
ANNOTATION_DIR = Path("data/raw/annotations")
TEST_SPLIT = Path("data/splits/test.txt")
OUTPUT_DIR = Path("outputs/predictions")
REPORT_PATH = OUTPUT_DIR / "eval_report.json"


def load_test_ids() -> List[str]:
    if not TEST_SPLIT.exists():
        raise FileNotFoundError(f"[TEST SPLIT MISSING] {TEST_SPLIT}")
    return [l.strip() for l in TEST_SPLIT.read_text().splitlines() if l.strip()]


def load_predictions() -> Dict[str, List[Dict]]:
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"[PREDICTIONS MISSING] {PREDICTIONS_PATH}")

    preds = json.loads(PREDICTIONS_PATH.read_text(encoding="utf-8"))
    grouped: Dict[str, List[Dict]] = {}

    for p in preds:
        grouped.setdefault(p["sample_id"], []).append(p)

    return grouped


def load_ground_truth(sample_id: str) -> List[Tuple[str, float, float]]:
    ann_path = ANNOTATION_DIR / f"{sample_id}_Annotated.txt"
    if not ann_path.exists():
        return []

    gt = []
    for line in ann_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        start, end, word = line.split("\t")
        gt.append((word, float(start), float(end)))

    return gt


def match_prediction(
    pred_word: str,
    pred_start: float,
    gt_segments: List[Tuple[str, float, float]],
    used_gt: set,
):
    candidates = [
        (i, gt)
        for i, gt in enumerate(gt_segments)
        if gt[0] == pred_word and i not in used_gt
    ]

    if not candidates:
        return None

    idx, best = min(candidates, key=lambda x: abs(x[1][1] - pred_start))
    used_gt.add(idx)
    return best


def evaluate_full_testset() -> Dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    test_ids = load_test_ids()
    predictions = load_predictions()

    start_errors = []
    end_errors = []

    total_gt_words = 0
    matched_words = 0
    missed_predictions = 0
    evaluated_samples = 0

    for sample_id in test_ids:
        if sample_id not in predictions:
            continue

        gt_segments = load_ground_truth(sample_id)
        if not gt_segments:
            continue

        total_gt_words += len(gt_segments)
        evaluated_samples += 1

        used_gt = set()

        for p in predictions[sample_id]:
            match = match_prediction(
                p["word"],
                p["start_time"],
                gt_segments,
                used_gt,
            )

            if match is None:
                missed_predictions += 1
                continue

            _, gs, ge = match
            start_errors.append(abs(p["start_time"] - gs))
            end_errors.append(abs(p["end_time"] - ge))
            matched_words += 1

    def mean(x):
        return round(float(sum(x) / len(x)), 4) if x else None

    def median(x):
        return round(float(statistics.median(x)), 4) if x else None

    precision = matched_words / (matched_words + missed_predictions) if matched_words else 0
    recall = matched_words / total_gt_words if total_gt_words else 0

    report = {
        "test_samples_total": len(test_ids),
        "test_samples_evaluated": evaluated_samples,
        "total_gt_words": total_gt_words,
        "matched_words": matched_words,
        "missed_predictions": missed_predictions,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "mean_start_error_sec": mean(start_errors),
        "mean_end_error_sec": mean(end_errors),
        "median_start_error_sec": median(start_errors),
        "median_end_error_sec": median(end_errors),
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if start_errors:
        plt.figure()
        plt.hist(start_errors, bins=30)
        plt.title("Start Time Error Distribution")
        plt.xlabel("Seconds")
        plt.ylabel("Count")
        plt.savefig(OUTPUT_DIR / "start_time_error_hist.png")
        plt.close()

    if end_errors:
        plt.figure()
        plt.hist(end_errors, bins=30)
        plt.title("End Time Error Distribution")
        plt.xlabel("Seconds")
        plt.ylabel("Count")
        plt.savefig(OUTPUT_DIR / "end_time_error_hist.png")
        plt.close()

    print("\nEvaluation completed successfully")
    print(f"Report saved to {REPORT_PATH}")
    print(f"Plots saved to {OUTPUT_DIR}")

    return report


if __name__ == "__main__":
    evaluate_full_testset()
