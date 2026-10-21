#!/usr/bin/env python3
"""Train all five spatial CV folds using configs/main.yaml (80 epochs per fold)."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training.cross_validate import main


if __name__ == "__main__":
    main()
