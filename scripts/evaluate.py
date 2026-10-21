#!/usr/bin/env python3
"""Evaluate five CV checkpoints and export fold mean/SD and pooled OOF metrics."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.cross_validate import main


if __name__ == "__main__":
    main()
