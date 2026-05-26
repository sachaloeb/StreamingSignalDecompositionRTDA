"""Re-render the pipeline dashboard with improved layout.

Reads saved trajectory and metrics data from ``results/demo_run/``
and produces an improved dashboard with ``constrained_layout``,
larger figure size, and an optional component cap.

Usage
-----
    python experiments/replot_pipeline_dashboard.py [--max-components 8]
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

# Default paths
DEFAULT_TRAJ = ROOT / "results" / "demo_run" / "baseline" / "trajectories.npz"
DEFAULT_METRICS = ROOT / "results" / "demo_run" / "baseline" / "metrics.csv"
DEFAULT_OUTDIR = ROOT / "results" / "demo_run"
FS = 1000.0  # sampling frequency used in the demo run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-render pipeline dashboard with improved layout.",
    )
    parser.add_argument(
        "--trajectories", type=str, default=str(DEFAULT_TRAJ),
        help="Path to trajectories.npz",
    )
    parser.add_argument(
        "--metrics", type=str, default=str(DEFAULT_METRICS),
        help="Path to per-window metrics CSV",
    )
    parser.add_argument(
        "--outdir", type=str, default=str(DEFAULT_OUTDIR),
        help="Output directory for the new dashboard",
    )
    parser.add_argument(
        "--max-components", type=int, default=8,
        help="Maximum number of component subplots to show (default 8)",
    )
    parser.add_argument(
        "--fs", type=float, default=FS,
        help="Sampling frequency in Hz (default 1000)",
    )
    args = parser.parse_args()

    # ── Load data ──────────────────────────────────────────────────
    traj_data = np.load(args.trajectories)
    all_keys = sorted(traj_data.keys(), key=int)
    trajectories = {int(k): traj_data[k] for k in all_keys}

    df = pd.read_csv(args.metrics)
    fs = args.fs

    # ── Select top components by time-integrated energy ───────────
    energies = {
        k: float(np.nansum(arr ** 2))
        for k, arr in trajectories.items()
    }
    sorted_by_energy = sorted(energies, key=energies.get, reverse=True)

    max_plot = args.max_components
    n_extra = max(0, len(sorted_by_energy) - max_plot)
    plot_keys = sorted(sorted_by_energy[:max_plot])

    # ── Reconstruct a proxy for the original signal ───────────────
    # Sum all trajectories (NaN → 0) to approximate the original signal
    max_len = max(len(trajectories[k]) for k in trajectories)
    signal_approx = np.zeros(max_len)
    for k in trajectories:
        arr = trajectories[k].copy()
        arr[np.isnan(arr)] = 0.0
        signal_approx[: len(arr)] += arr

    cmap = plt.cm.tab10
    t_sig = np.arange(max_len) / fs

    # ── Figure layout ─────────────────────────────────────────────
    n_comp = len(plot_keys)
    # Rows: signal + n_comp components + trajectory overlay + metrics
    n_rows = 1 + n_comp + 2
    fig_height = 1.2 * (n_comp + 3)

    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(14, max(fig_height, 8)),
        constrained_layout=True,
        gridspec_kw={
            "height_ratios": [2] + [1.5] * n_comp + [2, 2],
        },
    )

    # ── Row 0: Approximate original signal ────────────────────────
    ax0 = axes[0]
    ax0.plot(t_sig, signal_approx, color="black", linewidth=0.8)
    ax0.set_ylabel("Amplitude")
    ax0.set_title("Reconstructed Signal (sum of all components)")

    # ── Rows 1..n_comp: Individual components ─────────────────────
    for i, k in enumerate(plot_keys):
        arr = trajectories[k]
        t_k = np.arange(len(arr)) / fs
        ax = axes[1 + i]
        ax.plot(t_k, arr, color=cmap(k % 10), linewidth=0.8)
        ax.set_ylabel("Amp.")
        ax.set_title(f"Component {k + 1}", fontsize=10)

    if n_extra > 0:
        axes[n_comp].annotate(
            f"+ {n_extra} more component(s) not shown",
            xy=(0.5, -0.25), xycoords="axes fraction",
            ha="center", fontsize=9, fontstyle="italic", color="grey",
        )

    # ── Trajectory overlay ────────────────────────────────────────
    ax_traj = axes[1 + n_comp]
    ax_traj.plot(
        t_sig, signal_approx, color="lightgrey", alpha=0.25,
        label="Signal (approx.)",
    )
    for k in plot_keys:
        arr = trajectories[k]
        t_k = np.arange(len(arr)) / fs
        ax_traj.plot(
            t_k, arr, color=cmap(k % 10), linewidth=0.8,
            label=f"Component {k + 1}",
        )
    ax_traj.legend(loc="upper right", fontsize=7, ncol=2)
    ax_traj.set_ylabel("Amplitude")
    ax_traj.set_title("Component Trajectories (overlay)")

    # ── Metrics: QRF & Matching Confidence ────────────────────────
    ax_met = axes[1 + n_comp + 1]
    if "qrf" in df.columns:
        ax_met.plot(
            df.index, df["qrf"], color="blue", linewidth=0.8,
            label="QRF (dB)",
        )
        ax_met.set_ylabel("QRF (dB)", color="blue")
        ax_met.tick_params(axis="y", labelcolor="blue")

    ax_met.set_xlabel("Window index")
    ax_met.set_title("QRF & Matching Confidence")

    if "matching_confidence" in df.columns:
        ax_mc = ax_met.twinx()
        ax_mc.plot(
            df.index, df["matching_confidence"],
            color="red", linewidth=0.8,
            label="Matching Confidence",
        )
        ax_mc.set_ylabel("Matching Confidence", color="red")
        ax_mc.tick_params(axis="y", labelcolor="red")

    fig.suptitle(
        "Streaming SSD Pipeline Dashboard",
        fontsize=15, fontweight="bold",
    )

    # ── Save ──────────────────────────────────────────────────────
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = out / f"07_pipeline_dashboard_v2.{ext}"
        fig.savefig(path, dpi=300)
        print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()