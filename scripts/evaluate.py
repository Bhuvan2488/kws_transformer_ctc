#scripts/evaluate.py
import os
import sys
sys.path.append(os.getcwd())

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.evaluation.alignment_metrics import evaluate_full_testset


def main():
    print("\nRunning STEP 10 — FULL TEST SET EVALUATION")

    report = evaluate_full_testset()

    print("\nFINAL EVALUATION SUMMARY")
    for k, v in report.items():
        print(f"{k:30s}: {v}")


if __name__ == "__main__":
    main()


