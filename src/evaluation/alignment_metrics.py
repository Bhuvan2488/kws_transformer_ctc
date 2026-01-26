#src/evaluation/alignment_metrics.py
from pathlib import Path
from typing import Dict, List, Tuple
import json
import statistics

PREDICTIONS_PATH = Path("outputs/predictions/aligned_words.json")
ANNOTATION_DIR = Path("data/raw/annotations")
OUTPUT_REPORT = Path("outputs/predictions/eval_report.json")


def load_predictions() -> List[Dict]:
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"[PREDICTIONS MISSING] {PREDICTIONS_PATH}")

    preds = json.loads(PREDICTIONS_PATH.read_text(encoding="utf-8"))

    if not isinstance(preds, list):
        raise RuntimeError("[INVALID FORMAT] aligned_words.json must be a list")

    return preds


def load_ground_truth(sample_id: str) -> List[Tuple[str, float, float]]:
    ann_path = ANNOTATION_DIR / f"{sample_id}_Annotated.txt"
    if not ann_path.exists():
        raise FileNotFoundError(f"[ANNOTATION MISSING] {ann_path}")

    gt = []
    for line_num, line in enumerate(
        ann_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        line = line.strip()
        if not line:
            continue

        parts = line.split("\t")
        if len(parts) != 3:
            raise RuntimeError(
                f"[INVALID ANNOTATION FORMAT] {ann_path} line {line_num}"
            )

        start, end, word = parts
        gt.append((word, float(start), float(end)))

    return gt


def find_best_match(
    word: str,
    pred_start: float,
    gt_segments: List[Tuple[str, float, float]],
    used_idx: set,
) -> Tuple[int, Tuple[str, float, float]] | Tuple[None, None]:
    candidates = [
        (i, gt)
        for i, gt in enumerate(gt_segments)
        if gt[0] == word and i not in used_idx
    ]

    if not candidates:
        return None, None

    idx, best = min(candidates, key=lambda x: abs(x[1][1] - pred_start))
    return idx, best


def evaluate():
    print("\n STEP 10 — ALIGNMENT EVALUATION STARTED")

    predictions = load_predictions()

    start_errors = []
    end_errors = []
    missed = 0

    grouped: Dict[str, List[Dict]] = {}
    for p in predictions:
        sid = p["sample_id"]
        grouped.setdefault(sid, []).append(p)

    for sample_id, preds in grouped.items():
        gt_segments = load_ground_truth(sample_id)
        used_gt = set()

        for p in preds:
            word = p["word"]
            ps, pe = p["start_time"], p["end_time"]

            idx, gt = find_best_match(word, ps, gt_segments, used_gt)
            if gt is None:
                missed += 1
                continue

            used_gt.add(idx)
            _, gs, ge = gt

            start_errors.append(abs(ps - gs))
            end_errors.append(abs(pe - ge))

    def safe_mean(x):
        return float(sum(x) / len(x)) if x else None

    def safe_median(x):
        return float(statistics.median(x)) if x else None

    report = {
        "total_predictions": len(predictions),
        "evaluated_words": len(start_errors),
        "missed_words": missed,
        "mean_start_error_sec": safe_mean(start_errors),
        "mean_end_error_sec": safe_mean(end_errors),
        "median_start_error_sec": safe_median(start_errors),
        "median_end_error_sec": safe_median(end_errors),
    }

    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n STEP 10 — EVALUATION COMPLETED")
    print(json.dumps(report, indent=2))
    print(f"\n Report saved to: {OUTPUT_REPORT}")

    return report


if __name__ == "__main__":
    evaluate()
