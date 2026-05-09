"""Plot sensitivity results for nmse_threshold sweep.

Produces dual-axis QRF-and-time-vs-threshold plots, one panel per signal.

Usage
-----
    python experiments/plot_nmse_threshold_grid.py \
        [--input results/sensitivity/nmse_threshold/nmse_threshold_grid.csv]
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
        default="results/sensitivity/nmse_threshold/nmse_threshold_grid.csv",
    )
    parser.add_argument(
        "--outdir", type=str,
        default="results/sensitivity/nmse_threshold/plots",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    signals = df["signal"].unique()
    engines = df["engine"].unique()

    fig, axes = plt.subplots(
        1, len(signals), figsize=(7 * len(signals), 5),
        constrained_layout=True,
    )
    if len(signals) == 1:
        axes = [axes]

    for ax, sig_name in zip(axes, signals):
        sub = df[df["signal"] == sig_name]
        agg = sub.groupby(["threshold", "engine"]).agg(
            qrf_median=("median_qrf_db", "mean"),
            time_mean=("mean_time_ms", "mean"),
            qrf_std=("median_qrf_db", "std"),
            time_std=("mean_time_ms", "std"),
        ).reset_index()

        ax2 = ax.twinx()

        for eng in engines:
            eg = agg[agg["engine"] == eng]
            color = ENGINE_COLORS.get(eng, "#333333")
            label = ENGINE_LABELS.get(eng, eng)

            ax.errorbar(
                eg["threshold"], eg["qrf_median"], yerr=eg["qrf_std"],
                color=color, marker="o", linewidth=1.5, capsize=3,
                label=f"{label} — QRF",
            )
            ax2.errorbar(
                eg["threshold"], eg["time_mean"], yerr=eg["time_std"],
                color=color, marker="s", linestyle="--", linewidth=1.5,
                capsize=3, alpha=0.7,
                label=f"{label} — time",
            )

        ax.set_xscale("log")
        ax.set_xlabel("nmse_threshold")
        ax.set_ylabel("Median QRF (dB)", color="blue")
        ax2.set_ylabel("Mean per-window time (ms)", color="red")
        ax.set_title(sig_name)

    # Collect legends
    handles1, labels1 = axes[0].get_legend_handles_labels()
    handles2, labels2 = axes[0].get_legend_handles_labels()
    if hasattr(axes[0], "_twinx"):
        pass  # already captured
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="upper center", ncol=len(engines),
        bbox_to_anchor=(0.5, -0.02), fontsize=8,
    )

    fig.suptitle(
        "Sensitivity: nmse_threshold → QRF & per-window time",
        fontsize=13, fontweight="bold",
    )

    for ext in ("png", "pdf"):
        path = out_dir / f"nmse_threshold_qrf_vs_time.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
