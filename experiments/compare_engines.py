#!/usr/bin/env python
"""Compare streaming baseline SSD, offline SSD, and streaming SSD with FWHM.

Metrics reported
----------------
- Generated components: visual comparison of extracted component trajectories
- Global QRF: 20*log10(||x|| / ||x - x_hat||) over the full covered region
- Global NMSE: ||x - x_hat||^2 / ||x||^2 over the full covered region

Usage
-----
    python experiments/compare_engines.py [--output-dir results/engine_comparison]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
os.chdir(ROOT)

import copy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yaml

from experiments.synthetic.generators import chirp_plus_sinusoid
from src.engines import get_engine
from src.metrics.stability import qrf, nmse
from src.streaming.component_matcher import ComponentMatcher
from src.streaming.trajectory_store import TrajectoryStore
from src.streaming.window_manager import WindowManager


CONFIG_PATH = "experiments/configs/baseline.yaml"


# ---------------------------------------------------------------------------
# Streaming runner
# ---------------------------------------------------------------------------

def _run_streaming(
    signal: np.ndarray,
    fs: float,
    window_len: int,
    stride: int,
    max_components: int,
    engine_name: str,
    engine_kwargs: dict,
    matcher_kwargs: dict,
    label: str,
) -> tuple[dict[int, np.ndarray], list[float]]:
    """Run the full streaming pipeline and return trajectories + per-window QRF.

    Returns
    -------
    trajectories : dict[int, np.ndarray]
        Reconstructed component trajectories (length == len(signal), NaN where
        not yet written).
    per_window_qrf : list[float]
        QRF value recorded at each processed window.
    """
    N = len(signal)
    wm = WindowManager(window_len=window_len, stride=stride, fs=fs)
    engine = get_engine(engine_name, fs=fs, **engine_kwargs)
    matcher = ComponentMatcher(
        fs=fs,
        max_trajectories=max_components,
        **matcher_kwargs,
    )
    store = TrajectoryStore(max_components=max_components, max_len=N)

    per_window_qrf: list[float] = []
    overlap = wm.overlap

    print(f"  [{label}] streaming {N} samples ...")
    for i in range(N):
        window = wm.push(float(signal[i]))
        if window is None:
            continue

        components = engine.fit(window)
        components_no_res = components[:-1]

        matching = dict(matcher.match_stateful(components_no_res, overlap))

        window_start = i - window_len + 1
        store.update(window_start, components_no_res, matching, overlap)

        recon = (
            np.sum(components_no_res, axis=0)
            if components_no_res
            else np.zeros_like(window)
        )
        per_window_qrf.append(qrf(window, recon))

    return store.get_all(), per_window_qrf


# ---------------------------------------------------------------------------
# Offline runner
# ---------------------------------------------------------------------------

def _run_offline(
    signal: np.ndarray,
    fs: float,
    engine_name: str,
    engine_kwargs: dict,
    label: str,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Run SSD on the full signal at once.

    Returns
    -------
    components_no_res : list[np.ndarray]
        Extracted components (residual excluded).
    residual : np.ndarray
        SSD residual (last element of engine.fit output).
    """
    print(f"  [{label}] offline decomposition ({len(signal)} samples) ...")
    engine = get_engine(engine_name, fs=fs, **engine_kwargs)
    all_comps = engine.fit(signal)
    return all_comps[:-1], all_comps[-1]


# ---------------------------------------------------------------------------
# Global metric helpers
# ---------------------------------------------------------------------------

def _global_metrics_streaming(
    signal: np.ndarray,
    trajectories: dict[int, np.ndarray],
) -> tuple[float, float]:
    """Compute global QRF and NMSE from streaming trajectories.

    Uses nansum across all trajectory arrays to reconstruct the signal,
    then masks positions where no trajectory contributed (all-NaN columns).
    Metrics are evaluated only over the covered region.
    """
    N = len(signal)
    if not trajectories:
        return float("nan"), float("nan")

    # Stack trajectories: shape (n_traj, N)
    traj_matrix = np.stack(
        [v[:N] if len(v) >= N else np.pad(v, (0, N - len(v)), constant_values=np.nan)
         for v in trajectories.values()],
        axis=0,
    )

    # Positions covered by at least one trajectory
    covered = ~np.all(np.isnan(traj_matrix), axis=0)
    if not np.any(covered):
        return float("nan"), float("nan")

    recon = np.nansum(traj_matrix, axis=0)

    sig_cov = signal[covered]
    rec_cov = recon[covered]
    res_cov = sig_cov - rec_cov

    global_qrf = qrf(sig_cov, rec_cov)
    global_nmse = nmse(res_cov, sig_cov)
    return global_qrf, global_nmse


def _global_metrics_offline(
    signal: np.ndarray,
    components: list[np.ndarray],
    residual: np.ndarray,
) -> tuple[float, float]:
    """Compute global QRF and NMSE for the offline case."""
    recon = np.sum(components, axis=0) if components else np.zeros_like(signal)
    res = signal - recon
    global_qrf = qrf(signal, recon)
    global_nmse = nmse(res, signal)
    return global_qrf, global_nmse


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_components(
    signal: np.ndarray,
    results: dict[str, dict],
    out_path: Path,
) -> None:
    """Multi-panel figure showing original signal and extracted components per method."""
    # Determine max component count across methods
    max_comps = max(r["n_comps"] for r in results.values())
    n_methods = len(results)
    n_rows = 1 + n_methods  # original + one row per method

    fig, axes = plt.subplots(
        n_rows, max(max_comps, 1),
        figsize=(4 * max(max_comps, 1), 2.5 * n_rows),
        squeeze=False,
    )
    t = np.arange(len(signal))

    # Row 0: original signal (span all columns)
    for ax in axes[0]:
        ax.set_visible(False)
    # Use a single wide axis for the original signal
    ax_orig = fig.add_subplot(n_rows, 1, 1)
    ax_orig.plot(t, signal, color="k", lw=0.8)
    ax_orig.set_title("Original signal", fontsize=10)
    ax_orig.set_xlim(0, len(signal))
    ax_orig.set_xlabel("Sample")

    N = len(signal)
    for row_idx, (label, res) in enumerate(results.items(), start=1):
        comps = res["components"]  # list of np.ndarray (may vary in length)
        n_c = len(comps)
        for col_idx in range(max_comps):
            ax = axes[row_idx, col_idx]
            if col_idx < n_c:
                raw = comps[col_idx]
                # Pad or trim to signal length for a uniform x-axis
                if len(raw) < N:
                    comp = np.pad(raw, (0, N - len(raw)), constant_values=np.nan)
                else:
                    comp = raw[:N]
                ax.plot(t, comp, lw=0.7)
                ax.set_xlim(0, len(signal))
                ax.set_xlabel("Sample")
                if col_idx == 0:
                    ax.set_ylabel(label, fontsize=8)
                ax.set_title(f"Comp {col_idx}", fontsize=8)
            else:
                ax.set_visible(False)

    fig.suptitle("Component comparison", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved component plot -> {out_path}")


def _plot_metrics(
    results: dict[str, dict],
    per_window_qrf: dict[str, list[float]],
    out_path: Path,
) -> None:
    """Bar charts for global QRF and NMSE, plus per-window QRF line for streaming."""
    labels = list(results.keys())
    global_qrfs = [results[l]["global_qrf"] for l in labels]
    global_nmses = [results[l]["global_nmse"] for l in labels]

    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # --- Top left: Global QRF bar chart ---
    ax1 = fig.add_subplot(gs[0, 0])
    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
    bars = ax1.bar(labels, global_qrfs, color=colors)
    ax1.set_title("Global QRF (dB)\nhigher is better", fontsize=10)
    ax1.set_ylabel("QRF [dB]")
    ax1.tick_params(axis="x", rotation=15)
    for bar, val in zip(bars, global_qrfs):
        if np.isfinite(val):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2,
                f"{val:.2f}",
                ha="center", va="bottom", fontsize=8,
            )

    # --- Top right: Global NMSE bar chart ---
    ax2 = fig.add_subplot(gs[0, 1])
    bars2 = ax2.bar(labels, global_nmses, color=colors)
    ax2.set_title("Global NMSE\nlower is better", fontsize=10)
    ax2.set_ylabel("NMSE")
    ax2.tick_params(axis="x", rotation=15)
    for bar, val in zip(bars2, global_nmses):
        if np.isfinite(val):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02,
                f"{val:.4f}",
                ha="center", va="bottom", fontsize=8,
            )

    # --- Bottom: Per-window QRF traces for streaming conditions ---
    ax3 = fig.add_subplot(gs[1, :])
    streaming_labels = [l for l in labels if l in per_window_qrf and per_window_qrf[l]]
    for i, label in enumerate(streaming_labels):
        vals = np.array(per_window_qrf[label], dtype=float)
        finite = np.isfinite(vals)
        ax3.plot(
            np.where(finite)[0], vals[finite],
            label=label, lw=1.2, color=colors[labels.index(label)],
        )
    ax3.set_title("Per-window QRF (streaming conditions)", fontsize=10)
    ax3.set_xlabel("Window index")
    ax3.set_ylabel("QRF [dB]")
    ax3.legend(fontsize=8)

    fig.suptitle("Engine comparison: QRF and NMSE", fontsize=12)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved metrics plot -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(output_dir: str = "results/engine_comparison") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(CONFIG_PATH) as fh:
        base_cfg = yaml.safe_load(fh)

    sig_cfg = base_cfg["signal"]
    fs = sig_cfg["fs"]
    signal = chirp_plus_sinusoid(
        N=sig_cfg["N"],
        fs=fs,
        f_sin=sig_cfg["f_sin"],
        f_start=sig_cfg["f_start"],
        f_end=sig_cfg["f_end"],
        snr_db=sig_cfg.get("snr_db"),
    )

    window_len = base_cfg["streaming"]["window_len"]
    stride = base_cfg["streaming"]["stride"]
    max_components = base_cfg["streaming"]["max_components"]

    engine_kwargs = {
        k: v for k, v in base_cfg["engine"].items() if k != "name"
    }
    matcher_kwargs = {
        k: v for k, v in base_cfg["matcher"].items()
        if k in ("distance", "freq_weight", "lookback", "max_cost")
    }

    # -----------------------------------------------------------------------
    # Run conditions
    # -----------------------------------------------------------------------

    print("\n=== Streaming baseline SSD ===")
    trajs_streaming, pwqrf_streaming = _run_streaming(
        signal, fs, window_len, stride, max_components,
        engine_name="ssd",
        engine_kwargs=copy.deepcopy(engine_kwargs),
        matcher_kwargs=copy.deepcopy(matcher_kwargs),
        label="streaming_baseline",
    )
    gqrf_streaming, gnmse_streaming = _global_metrics_streaming(signal, trajs_streaming)

    print("\n=== Offline baseline SSD ===")
    offline_comps, offline_residual = _run_offline(
        signal, fs,
        engine_name="ssd",
        engine_kwargs=copy.deepcopy(engine_kwargs),
        label="offline_baseline",
    )
    gqrf_offline, gnmse_offline = _global_metrics_offline(signal, offline_comps, offline_residual)

    print("\n=== Streaming SSD + FWHM ===")
    fwhm_engine_kwargs = copy.deepcopy(engine_kwargs)
    fwhm_engine_kwargs["spectral_method"] = "fwhm"
    trajs_fwhm, pwqrf_fwhm = _run_streaming(
        signal, fs, window_len, stride, max_components,
        engine_name="ssd_optimized",
        engine_kwargs=fwhm_engine_kwargs,
        matcher_kwargs=copy.deepcopy(matcher_kwargs),
        label="streaming_fwhm",
    )
    gqrf_fwhm, gnmse_fwhm = _global_metrics_streaming(signal, trajs_fwhm)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    results: dict[str, dict] = {
        "streaming_baseline": {
            "global_qrf": gqrf_streaming,
            "global_nmse": gnmse_streaming,
            "n_comps": len(trajs_streaming),
            # Convert trajectories to component arrays (NaN → 0 for plotting)
            "components": [
                np.where(np.isnan(v), 0.0, v)
                for v in trajs_streaming.values()
            ],
        },
        "offline_baseline": {
            "global_qrf": gqrf_offline,
            "global_nmse": gnmse_offline,
            "n_comps": len(offline_comps),
            "components": offline_comps,
        },
        "streaming_fwhm": {
            "global_qrf": gqrf_fwhm,
            "global_nmse": gnmse_fwhm,
            "n_comps": len(trajs_fwhm),
            "components": [
                np.where(np.isnan(v), 0.0, v)
                for v in trajs_fwhm.values()
            ],
        },
    }

    per_window_qrf = {
        "streaming_baseline": pwqrf_streaming,
        "streaming_fwhm": pwqrf_fwhm,
    }

    print("\n" + "=" * 60)
    print(f"{'Method':<25} {'Global QRF (dB)':>16} {'Global NMSE':>12}")
    print("-" * 60)
    for label, res in results.items():
        qrf_str = f"{res['global_qrf']:.4f}" if np.isfinite(res['global_qrf']) else "inf"
        nmse_str = f"{res['global_nmse']:.6f}" if np.isfinite(res['global_nmse']) else "nan"
        print(f"  {label:<23} {qrf_str:>16} {nmse_str:>12}")
        print(f"    n_components = {res['n_comps']}")
    print("=" * 60)

    # Save summary CSV
    import csv
    csv_path = out / "summary.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["method", "global_qrf_db", "global_nmse", "n_components"],
        )
        writer.writeheader()
        for label, res in results.items():
            writer.writerow({
                "method": label,
                "global_qrf_db": res["global_qrf"],
                "global_nmse": res["global_nmse"],
                "n_components": res["n_comps"],
            })
    print(f"\nSummary CSV -> {csv_path}")

    # Save per-window QRF
    pw_csv_path = out / "per_window_qrf.csv"
    max_len = max((len(v) for v in per_window_qrf.values()), default=0)
    with open(pw_csv_path, "w", newline="") as fh:
        cols = ["window_index"] + list(per_window_qrf.keys())
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for i in range(max_len):
            row: dict[str, object] = {"window_index": i}
            for label, vals in per_window_qrf.items():
                row[label] = vals[i] if i < len(vals) else ""
            writer.writerow(row)
    print(f"Per-window QRF -> {pw_csv_path}")

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    print("\nGenerating plots ...")
    _plot_components(signal, results, out / "components_comparison.png")
    _plot_metrics(results, per_window_qrf, out / "metrics_comparison.png")

    print(f"\nAll outputs written to: {out.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare streaming baseline, offline, and FWHM SSD engines."
    )
    parser.add_argument(
        "--output-dir",
        default="results/engine_comparison",
        help="Directory for results (default: results/engine_comparison)",
    )
    args = parser.parse_args()
    main(args.output_dir)