"""Plot sensitivity results for max_components sweep.

Produces dual-axis QRF-and-time-vs-max_components plot.

Usage
-----
    python experiments/plot_max_components_grid.py \
        [--input results/sensitivity/max_components/max_components_grid.csv]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    pass

ENGINE_COLORS = {
    "ssd": "#666666",
    "ssd_optimized_fwhm": "#1f77b4",
}

ENGINE_LABELS = {
    "ssd": "Baseline SSD",
    "ssd_optimized_fwhm": "OptimizedSSD-FWHM",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str,
        default="results/sensitivity/max_components/max_components_grid.csv",
    )
    parser.add_argument(
        "--outdir", type=str,
        default="results/sensitivity/max_components/plots",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    engines = df["engine"].unique()

    agg = df.groupby(["max_components", "engine"]).agg(
        qrf_median=("median_qrf_db", "mean"),
        time_mean=("mean_time_ms", "mean"),
        qrf_std=("median_qrf_db", "std"),
        time_std=("mean_time_ms", "std"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax2 = ax.twinx()

    for eng in engines:
        eg = agg[agg["engine"] == eng]
        color = ENGINE_COLORS.get(eng, "#333333")
        label = ENGINE_LABELS.get(eng, eng)

        ax.errorbar(
            eg["max_components"], eg["qrf_median"], yerr=eg["qrf_std"],
            color=color, marker="o", linewidth=1.5, capsize=3,
            label=f"{label} — QRF",
        )
        ax2.errorbar(
            eg["max_components"], eg["time_mean"], yerr=eg["time_std"],
            color=color, marker="s", linestyle="--", linewidth=1.5,
            capsize=3, alpha=0.7,
            label=f"{label} — time",
        )

    ax.set_xlabel("max_components")
    ax.set_ylabel("Median QRF (dB)", color="blue")
    ax2.set_ylabel("Mean per-window time (ms)", color="red")
    ax.set_title(
        "Sensitivity: max_components → QRF & per-window time\n"
        "(chirp_plus_sinusoid, N=3000, fs=1000)",
        fontsize=12,
    )

    # Merged legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="best", fontsize=8)

    for ext in ("png", "pdf"):
        path = out_dir / f"max_components_qrf_vs_time.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
