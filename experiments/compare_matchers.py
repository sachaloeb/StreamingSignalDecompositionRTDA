#!/usr/bin/env python
"""Compare matcher distance strategies on the baseline signal.

Runs the streaming pipeline for each matcher configuration,
collects per-window metrics, and writes a summary CSV to
``results/matcher_comparison.csv``.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os

os.chdir(ROOT)

import numpy as np
import pandas as pd
import yaml

from experiments.run_experiment import run

STRATEGIES: list[dict[str, object]] = [
    {"distance": "d_corr", "freq_weight": 0.0},
    {"distance": "d_freq", "freq_weight": 1.0},
    {"distance": "hybrid", "freq_weight": 0.3},
    {"distance": "hybrid", "freq_weight": 0.5},
]

CONFIG_PATH = "experiments/configs/baseline.yaml"


def _tag(strategy: dict[str, object]) -> str:
    """Build a short filesystem-safe tag for a strategy."""
    d = strategy["distance"]
    w = strategy["freq_weight"]
    return f"{d}_fw{w}"


def main() -> None:
    """Execute comparison and save summary."""
    with open(CONFIG_PATH) as fh:
        base_cfg = yaml.safe_load(fh)

    summary_rows: list[dict[str, object]] = []

    for strat in STRATEGIES:
        tag = _tag(strat)
        print(f"\n>>> Running strategy: {tag}")

        cfg = copy.deepcopy(base_cfg)
        cfg["matcher"].update(strat)
        out_dir = f"results/matcher_{tag}"

        run(config_dict=cfg, output_dir=out_dir)

        csv_path = Path(out_dir) / "metrics.csv"
        if not csv_path.exists():
            print(f"  WARNING: {csv_path} not found, skipping")
            continue

        df = pd.read_csv(csv_path)

        mean_conf = float(
            df["matching_confidence"].mean()
        ) if "matching_confidence" in df.columns else np.nan

        mean_fd = float(
            df["freq_drift"].mean()
        ) if "freq_drift" in df.columns else np.nan

        mean_qrf = float(
            df["qrf"].mean()
        ) if "qrf" in df.columns else np.nan

        std_qrf = float(
            df["qrf"].std()
        ) if "qrf" in df.columns else np.nan

        summary_rows.append({
            "strategy": str(strat["distance"]),
            "freq_weight": strat["freq_weight"],
            "mean_confidence": mean_conf,
            "mean_freq_drift": mean_fd,
            "mean_qrf": mean_qrf,
            "std_qrf": std_qrf,
        })

    summary_df = pd.DataFrame(summary_rows)
    out_path = Path("results/matcher_comparison.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_path, index=False)
    print(f"\nSummary saved to {out_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
