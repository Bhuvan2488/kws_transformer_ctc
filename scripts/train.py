# scripts/train.py
import os
import sys
from pathlib import Path

sys.path.append(os.getcwd())

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.training.train import train


def main():
    print("Launching training pipeline...")
    train()


if __name__ == "__main__":
    main()

