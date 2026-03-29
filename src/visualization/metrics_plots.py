"""Streaming-pipeline metrics visualization.

Plots per-window quality metrics loaded from a CSV file produced
by ``experiments.run_experiment.run``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_metrics_over_windows(
    metrics_csv_path: str,
    save_path: str | None = None,
) -> None:
    """Plot QRF, freq_drift, energy_continuity, and matching_confidence.

    Parameters
    ----------
    metrics_csv_path : str
        Path to a CSV with columns produced by ``run_experiment``.
    save_path : str or None, optional
        If given, save figure to this path and close it.

    Notes
    -----
    Creates a 2x2 panel figure.  Missing columns are silently skipped.
    """
    df = pd.read_csv(metrics_csv_path)

    panels = [
        ("qrf", "blue", [("y", 0, "grey", "--")]),
        ("freq_drift", "orange", []),
        ("energy_continuity", "green", []),
        (
            "matching_confidence", "red",
            [("y", 0.5, "grey", "--")],
        ),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes_flat = axes.ravel()

    for idx, (col, color, hlines) in enumerate(panels):
        ax = axes_flat[idx]
        if col not in df.columns:
            ax.set_visible(False)
            continue
        ax.plot(df.index, df[col], color=color, linewidth=0.8)
        for _, yval, hcolor, lstyle in hlines:
            ax.axhline(yval, color=hcolor, linestyle=lstyle,
                       linewidth=0.6)
        ax.set_ylabel(col)
        ax.set_title(col)

    for ax in axes_flat[2:]:
        ax.set_xlabel("Window index")

    stem = Path(metrics_csv_path).stem
    fig.suptitle(
        f"Streaming Pipeline Metrics \u2014 {stem}", fontsize=13,
    )
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
