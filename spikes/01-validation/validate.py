"""Spike 1: Model Validation Back-Test for wcsim PRD §5.5.

Loads pre-WC Elo + FIFA ratings and historical match results, predicts every
match under Elo / FIFA / Blend rating modes under two scoring conventions
(90-min and post-ET), computes Brier + draw-rate + Elo-mode calibration,
and emits results/brier.json plus results/calibration.png.

Throwaway. Single-threaded NumPy. No CLI, no package, no unit tests.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


HERE = Path(__file__).parent
DATA = HERE / "data" / "raw"
RESULTS = HERE / "results"


def main() -> None:
    raise NotImplementedError("Tasks 4-11 fill this in.")


if __name__ == "__main__":
    main()
