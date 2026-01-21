#scripts/evaluate.py
import os
import sys
sys.path.append(os.getcwd())

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.evaluation.alignment_metrics import evaluate


def main():
    print("🚀 Running STEP 10 — Evaluation")
    report = evaluate()

    print("\n📊 FINAL EVALUATION SUMMARY")
    for k, v in report.items():
        print(f"{k:25s}: {v}")


if __name__ == "__main__":
    main()

