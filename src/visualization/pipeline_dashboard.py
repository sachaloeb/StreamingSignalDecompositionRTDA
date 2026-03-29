"""Unified pipeline dashboard combining all major visualizations.

Produces a single large figure with the original signal, stacked
components, trajectory overlay, and streaming quality metrics.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec


def plot_pipeline_dashboard(
    signal: np.ndarray,
    components: list[np.ndarray],
    trajectory_store: object,
    metrics_csv_path: str,
    fs: float = 1.0,
    save_path: str | None = None,
) -> None:
    """Render a full-page dashboard of the streaming SSD pipeline.

    Parameters
    ----------
    signal : np.ndarray
        Original input signal.
    components : list[np.ndarray]
        Extracted SSD components (excluding residual).
    trajectory_store : TrajectoryStore
        Instance with ``get_all()`` and ``get(i)`` methods.
    metrics_csv_path : str
        Path to the per-window metrics CSV.
    fs : float, optional
        Sampling frequency in Hz.  Default 1.0.
    save_path : str or None, optional
        If given, save figure to this path and close it.

    Notes
    -----
    The figure uses a ``GridSpec`` with four row groups:
    original signal, stacked components, trajectory overlay,
    and twin-axis QRF / matching-confidence plot.
    """
    r = len(components)
    height_ratios = [2] + [2] * max(r, 1) + [2, 2]
    n_rows = len(height_ratios)

    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(
        n_rows, 1, figure=fig, height_ratios=height_ratios,
        hspace=0.35,
    )

    cmap = plt.cm.tab10
    t_sig = np.arange(len(signal)) / fs

    ax0 = fig.add_subplot(gs[0])
    ax0.plot(t_sig, signal, color="black", linewidth=0.8)
    ax0.set_ylabel("Amplitude")
    ax0.set_title("Original Signal")

    for i, comp in enumerate(components):
        ax = fig.add_subplot(gs[1 + i], sharex=ax0)
        tc = np.arange(len(comp)) / fs
        ax.plot(tc, comp, color=cmap(i % 10), linewidth=0.8)
        ax.set_ylabel("Amplitude")
        ax.set_title(f"Component {i + 1}")

    row_traj = 1 + max(r, 1)
    ax_traj = fig.add_subplot(gs[row_traj], sharex=ax0)
    ax_traj.plot(
        t_sig, signal, color="lightgrey", alpha=0.25,
        label="Signal",
    )
    trajs = trajectory_store.get_all()
    for k in sorted(trajs.keys()):
        arr = trajectory_store.get(k)
        t_k = np.arange(len(arr)) / fs
        ax_traj.plot(
            t_k, arr, color=cmap(k % 10), linewidth=0.8,
            label=f"Component {k + 1}",
        )
    ax_traj.legend(loc="upper right", fontsize=7)
    ax_traj.set_ylabel("Amplitude")
    ax_traj.set_title("Component Trajectories")

    row_met = row_traj + 1
    ax_qrf = fig.add_subplot(gs[row_met])
    df = pd.read_csv(metrics_csv_path)
    if "qrf" in df.columns:
        ax_qrf.plot(
            df.index, df["qrf"], color="blue", linewidth=0.8,
            label="QRF (dB)",
        )
        ax_qrf.set_ylabel("QRF (dB)", color="blue")
        ax_qrf.tick_params(axis="y", labelcolor="blue")
    ax_qrf.set_xlabel("Window index")
    ax_qrf.set_title("QRF & Matching Confidence")

    if "matching_confidence" in df.columns:
        ax_mc = ax_qrf.twinx()
        ax_mc.plot(
            df.index, df["matching_confidence"],
            color="red", linewidth=0.8,
            label="Matching Confidence",
        )
        ax_mc.set_ylabel("Matching Confidence", color="red")
        ax_mc.tick_params(axis="y", labelcolor="red")

    fig.suptitle(
        "Streaming SSD Pipeline Dashboard", fontsize=15, y=0.98,
    )

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
