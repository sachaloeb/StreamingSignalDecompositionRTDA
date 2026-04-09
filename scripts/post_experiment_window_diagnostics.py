#!/usr/bin/env python3
"""Inspect flagged windows from a streaming SSD experiment run.

Regenerates the signal from the saved config, extracts each flagged
window, runs SSD on it, and produces per-window diagnostic plots
(original signal, extracted components, and component PSDs).

Usage
-----
    python scripts/post_experiment_window_diagnostics.py \
        --results-dir results/demo_run \
        [--flagged 5 27 29 30 31 58] \
        [--top-k 10]

If ``--flagged`` is omitted the script auto-selects the *top-k*
windows by ``energy_continuity`` from ``metrics.csv``.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from experiments.synthetic.generators import (
    chirp_plus_sinusoid,
    component_onset,
    rossler,
    two_sinusoids,
)
from src.engines.ssd import SSD

_GENERATORS = {
    "two_sinusoids": two_sinusoids,
    "chirp_plus_sinusoid": chirp_plus_sinusoid,
    "rossler": rossler,
    "component_onset": component_onset,
}


def _load_config(results_dir: Path) -> dict:
    """Load the YAML config that was copied into *results_dir*."""
    cfg_path = results_dir / "config_used.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"No config_used.yaml in {results_dir}. "
            "Run the experiment first."
        )
    with open(cfg_path) as fh:
        return yaml.safe_load(fh)


def _generate_signal(cfg: dict) -> np.ndarray:
    sig_cfg = dict(cfg["signal"])
    sig_type = sig_cfg.pop("type")
    gen = _GENERATORS.get(sig_type)
    if gen is None:
        raise ValueError(f"Unknown signal type '{sig_type}'")
    return gen(**sig_cfg)


def _load_metrics(results_dir: Path) -> list[dict]:
    csv_path = results_dir / "metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No metrics.csv in {results_dir}")
    with open(csv_path) as fh:
        return list(csv.DictReader(fh))


def _auto_flag(rows: list[dict], top_k: int) -> list[int]:
    """Return indices of the *top_k* windows by energy_continuity."""
    scored: list[tuple[int, float]] = []
    for i, r in enumerate(rows):
        val = r.get("energy_continuity", "nan")
        try:
            ec = float(val)
        except (ValueError, TypeError):
            ec = 0.0
        if not np.isfinite(ec):
            ec = 0.0
        scored.append((i, ec))
    scored.sort(key=lambda t: t[1], reverse=True)
    return [idx for idx, _ in scored[:top_k]]


def _extract_window(
    signal: np.ndarray,
    window_idx: int,
    window_len: int,
    stride: int,
) -> np.ndarray:
    """Slice the window that was emitted at *window_idx*."""
    start = window_idx * stride
    end = start + window_len
    return signal[start:end]


def _plot_window(
    window: np.ndarray,
    components: list[np.ndarray],
    residual: np.ndarray,
    fs: float,
    window_idx: int,
    metrics_row: dict,
    save_path: str,
) -> None:
    n_comps = len(components)
    n_rows = 1 + n_comps + 1  # original + each component + PSD panel
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 2.5 * n_rows))
    t = np.arange(len(window)) / fs

    # --- original window ---
    axes[0].plot(t, window, color="black", linewidth=0.8)
    axes[0].set_title(
        f"Window {window_idx}  |  "
        f"QRF={_fmt(metrics_row.get('qrf'))} dB   "
        f"EC={_fmt(metrics_row.get('energy_continuity'))}   "
        f"SVD-drift={_fmt(metrics_row.get('singular_value_drift'))}   "
        f"MC={_fmt(metrics_row.get('matching_confidence'))}",
        fontsize=9,
    )
    axes[0].set_ylabel("Original")

    # --- each component ---
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_comps, 1)))
    for i, comp in enumerate(components):
        ax = axes[1 + i]
        ax.plot(t, comp, color=colors[i], linewidth=0.7)
        ax.set_ylabel(f"Comp {i}")

    # --- PSD of each component ---
    ax_psd = axes[-1]
    for i, comp in enumerate(components):
        nperseg = min(len(comp), 256)
        freqs, psd = welch(comp, fs=fs, nperseg=nperseg)
        ax_psd.semilogy(freqs, psd, color=colors[i], label=f"C{i}")
    ax_psd.set_xlabel("Frequency (Hz)")
    ax_psd.set_ylabel("PSD")
    ax_psd.legend(fontsize=7, ncol=min(n_comps, 6))
    ax_psd.set_title("Component PSDs")

    for ax in axes[:-1]:
        ax.set_xticklabels([])

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _fmt(val: object) -> str:
    """Format a metric value for the title."""
    try:
        v = float(val)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return "N/A"
    if not np.isfinite(v):
        return f"{v}"
    return f"{v:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect flagged windows from a streaming SSD run.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/demo_run",
        help="Path to the experiment results directory.",
    )
    parser.add_argument(
        "--flagged",
        type=int,
        nargs="*",
        default=None,
        help="Window indices to inspect (default: auto-select by EC).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of windows to auto-flag when --flagged is omitted.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    cfg = _load_config(results_dir)
    signal = _generate_signal(cfg)
    rows = _load_metrics(results_dir)

    fs = cfg["signal"]["fs"]
    window_len = cfg["streaming"]["window_len"]
    stride = cfg["streaming"]["stride"]

    flagged = args.flagged if args.flagged else _auto_flag(rows, args.top_k)
    flagged = [i for i in flagged if 0 <= i < len(rows)]

    outdir = results_dir / "inspection"
    os.makedirs(outdir, exist_ok=True)

    ssd = SSD(fs=fs, nmse_threshold=cfg["ssd"]["nmse_threshold"],
              max_iter=cfg["ssd"]["max_iter"])

    print(f"Inspecting {len(flagged)} windows: {flagged}")
    for idx in flagged:
        window = _extract_window(signal, idx, window_len, stride)
        if len(window) < window_len:
            print(f"  Skipping window {idx}: extends past signal end")
            continue

        components = ssd.fit(window)
        residual = components[-1]
        components_no_res = components[:-1]

        save_path = str(outdir / f"window_{idx:04d}.png")
        _plot_window(
            window, components_no_res, residual,
            fs, idx, rows[idx], save_path,
        )
        print(f"  Window {idx}: {len(components_no_res)} components -> {save_path}")

    print(f"Inspection plots saved to {outdir}")


if __name__ == "__main__":
    main()
