# scripts/preprocess.py

import subprocess
import sys


def run(cmd):
    print(f"\n>>> Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    return result


def main():
    python = sys.executable  # ensures correct venv / colab python

    # STEP 2: Data cleaning / validation
    run([python, "src/data/clean.py"])

    # STEP 3: Feature extraction
    run([python, "src/data/feature_extraction.py"])

    # STEP 4: Text tokenization
    run([python, "src/data/text_loader.py"])

    # STEP 5: Dataset split
    run([python, "src/data/split.py"])

    print("\n✅ Preprocessing pipeline completed successfully.")


if __name__ == "__main__":
    main()
