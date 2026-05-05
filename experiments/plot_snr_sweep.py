"""Plot the multi-seed SNR sweep results.

Reads snr_sweep_stats.csv and produces:
- snr_sweep_qrf.png/.pdf  — one panel per signal, QRF vs SNR with 95% CI
- snr_sweep_diff.png/.pdf — one panel per signal, (engine - baseline) QRF diff

Usage
-----
    python experiments/plot_snr_sweep.py \\
        --csv results/snr_sweep_multiseed/snr_sweep_stats.csv \\
        --out results/snr_sweep_multiseed/plots
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    pass

ENGINE_COLORS = {
    "ssd":                     "#666666",
    "ssd_optimized_fwhm":      "#1f77b4",
    "ssd_optimized_moment":    "#d62728",
    "ssd_optimized_gaussian":  "#2ca02c",
}
ENGINE_LABELS = {
    "ssd":                     "Baseline SSD",
    "ssd_optimized_fwhm":      "OptimizedSSD-fwhm",
    "ssd_optimized_moment":    "OptimizedSSD-moment",
    "ssd_optimized_gaussian":  "OptimizedSSD-gaussian+jac",
}


def _read_stats(csv_path: Path) -> list[dict]:
    rows = []
    with open(csv_path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append({
                "signal": row["signal"],
                "snr_db": float(row["snr_db"]),
                "engine": row["engine"],
                "median_qrf_db": float(row["median_qrf_db"])
                    if row["median_qrf_db"] not in ("nan", "") else float("nan"),
                "ci_lo": float(row["ci_lo"])
                    if row["ci_lo"] not in ("nan", "") else float("nan"),
                "ci_hi": float(row["ci_hi"])
                    if row["ci_hi"] not in ("nan", "") else float("nan"),
                "reject_null": row.get("reject_null", "False") == "True",
            })
    return rows


def _save_plot(fig: plt.Figure, out_dir: Path, name: str) -> None:
    for ext, dpi in [(".png", 300), (".pdf", None)]:
        path = out_dir / f"{name}{ext}"
        kw = {"dpi": dpi} if dpi else {}
        fig.savefig(path, **kw)
        print(f"Saved: {path}")
    plt.close(fig)


def plot_qrf_panel(rows: list[dict], out_dir: Path) -> None:
    """One panel per signal: QRF vs SNR with shaded 95% CI."""
    signals = sorted(set(r["signal"] for r in rows))
    engines = sorted(set(r["engine"] for r in rows))
    snrs = sorted(set(r["snr_db"] for r in rows))

    n_panels = len(signals)
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 5), sharey=False)
    if n_panels == 1:
        axes = [axes]

    for ax, sig in zip(axes, signals):
        for eng in engines:
            color = ENGINE_COLORS.get(eng, "#333333")
            label = ENGINE_LABELS.get(eng, eng)
            xs, ys, lo_errs, hi_errs = [], [], [], []
            for snr in snrs:
                subset = [r for r in rows
                          if r["signal"] == sig and r["snr_db"] == snr and r["engine"] == eng]
                if not subset:
                    continue
                r = subset[0]
                if not np.isfinite(r["median_qrf_db"]):
                    continue
                xs.append(snr)
                ys.append(r["median_qrf_db"])
                lo_errs.append(
                    max(0.0, r["median_qrf_db"] - r["ci_lo"])
                    if np.isfinite(r["ci_lo"]) else 0.0
                )
                hi_errs.append(
                    max(0.0, r["ci_hi"] - r["median_qrf_db"])
                    if np.isfinite(r["ci_hi"]) else 0.0
                )
            if not xs:
                continue
            ax.plot(xs, ys, "o-", color=color, label=label, linewidth=1.5)
            ax.fill_between(
                xs,
                [y - e for y, e in zip(ys, lo_errs)],
                [y + e for y, e in zip(ys, hi_errs)],
                color=color, alpha=0.15,
            )

        ax.set_xlabel("SNR (dB)", fontsize=10)
        ax.set_ylabel("Median QRF (dB)", fontsize=10)
        ax.set_title(sig.replace("_", " "), fontsize=10)
        ax.legend(fontsize=7)

    fig.suptitle("QRF vs SNR (shaded = 95% bootstrap CI)", fontsize=12)
    fig.tight_layout()
    _save_plot(fig, out_dir, "snr_sweep_qrf")


def plot_diff_panel(rows: list[dict], out_dir: Path) -> None:
    """One panel per signal: (engine - baseline) QRF difference with CI."""
    signals = sorted(set(r["signal"] for r in rows))
    engines = sorted(e for e in set(r["engine"] for r in rows) if e != "ssd")
    snrs = sorted(set(r["snr_db"] for r in rows))

    # Lookup baseline
    baseline_lookup: dict[tuple, float] = {}
    for r in rows:
        if r["engine"] == "ssd" and np.isfinite(r["median_qrf_db"]):
            baseline_lookup[(r["signal"], r["snr_db"])] = r["median_qrf_db"]

    n_panels = len(signals)
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 5), sharey=False)
    if n_panels == 1:
        axes = [axes]

    for ax, sig in zip(axes, signals):
        for eng in engines:
            color = ENGINE_COLORS.get(eng, "#333333")
            label = ENGINE_LABELS.get(eng, eng)
            xs, diffs, lo_diffs, hi_diffs, stars = [], [], [], [], []
            for snr in snrs:
                baseline_qrf = baseline_lookup.get((sig, snr))
                if baseline_qrf is None:
                    continue
                subset = [r for r in rows
                          if r["signal"] == sig and r["snr_db"] == snr and r["engine"] == eng]
                if not subset:
                    continue
                r = subset[0]
                if not np.isfinite(r["median_qrf_db"]):
                    continue
                diff = r["median_qrf_db"] - baseline_qrf
                xs.append(snr)
                diffs.append(diff)
                lo_diffs.append(
                    max(0.0, diff - (r["ci_lo"] - baseline_qrf))
                    if np.isfinite(r["ci_lo"]) else 0.0
                )
                hi_diffs.append(
                    max(0.0, (r["ci_hi"] - baseline_qrf) - diff)
                    if np.isfinite(r["ci_hi"]) else 0.0
                )
                stars.append(r["reject_null"])

            if not xs:
                continue
            ax.plot(xs, diffs, "o-", color=color, label=label, linewidth=1.5)
            ax.fill_between(
                xs,
                [d - lo for d, lo in zip(diffs, lo_diffs)],
                [d + hi for d, hi in zip(diffs, hi_diffs)],
                color=color, alpha=0.15,
            )
            # Star annotation for BH-rejected cells
            for x, d, star in zip(xs, diffs, stars):
                if star:
                    ax.annotate("*", (x, d), fontsize=14, ha="center", va="bottom",
                                color=color)

        ax.axhline(0.0, color="black", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel("SNR (dB)", fontsize=10)
        ax.set_ylabel("QRF diff vs baseline (dB)", fontsize=10)
        ax.set_title(sig.replace("_", " "), fontsize=10)
        ax.legend(fontsize=7)

    fig.suptitle("QRF difference vs baseline (shaded = 95% CI; * = BH-rejected)", fontsize=12)
    fig.tight_layout()
    _save_plot(fig, out_dir, "snr_sweep_diff")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot SNR sweep results")
    parser.add_argument(
        "--csv",
        default="results/snr_sweep_multiseed/snr_sweep_stats.csv",
    )
    parser.add_argument(
        "--out",
        default="results/snr_sweep_multiseed/plots",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_stats(csv_path)
    print(f"Loaded {len(rows)} rows from {csv_path}")

    plot_qrf_panel(rows, out_dir)
    plot_diff_panel(rows, out_dir)


if __name__ == "__main__":
    main()
