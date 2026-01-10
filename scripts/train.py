# scripts/train.py

import sys
import subprocess


def main():
    python = sys.executable

    print("\n>>> Starting training pipeline\n")

    subprocess.run(
        [python, "src/training/train.py"],
        check=True
    )

    print("\n✅ Training completed successfully")


if __name__ == "__main__":
    main()
