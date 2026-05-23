"""Plot long-stream stress test results.

Reads long_stream_metrics.csv files from baseline and optimized_fwhm subdirs
and produces:
- latency_over_time.png/.pdf
- latency_cdf.png/.pdf
- memory_over_time.png/.pdf
- active_trajectories_over_time.png/.pdf

Usage
-----
    python experiments/plot_long_stream.py \\
        --baseline   results/long_stream/baseline \\
        --optimized  results/long_stream/optimized_fwhm \\
        --out        results/long_stream/plots
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
    "Baseline SSD":         "#666666",
    "OptimizedSSD-fwhm":    "#1f77b4",
}

RT_BUDGET_MS = 150.0  # stride=150, fs=1000 → T_w = 150 ms


def _read_csv(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append({
                "window_index": int(row["window_index"]),
                "time_ms": float(row["time_ms"])
                    if row["time_ms"] not in ("nan", "") else float("nan"),
                "peak_memory_mib": float(row["peak_memory_mib"])
                    if row["peak_memory_mib"] not in ("nan", "") else float("nan"),
                "qrf_db": float(row["qrf_db"])
                    if row["qrf_db"] not in ("nan", "") else float("nan"),
                "matching_confidence": float(row["matching_confidence"])
                    if row["matching_confidence"] not in ("nan", "") else float("nan"),
                "active_trajectories": int(row["active_trajectories"]),
            })
    return rows


def _save_plot(fig: plt.Figure, out_dir: Path, name: str) -> None:
    for ext, dpi in [(".png", 300), (".pdf", None)]:
        path = out_dir / f"{name}{ext}"
        kw = {"dpi": dpi} if dpi else {}
        fig.savefig(path, **kw)
        print(f"Saved: {path}")
    plt.close(fig)


def plot_latency_over_time(datasets: dict[str, list[dict]], out_dir: Path) -> None:
    """Per-window processing time vs window index, one line per engine."""
    fig, ax = plt.subplots(figsize=(12, 5))

    for label, rows in datasets.items():
        color = ENGINE_COLORS.get(label, "#333333")
        xs = [r["window_index"] for r in rows]
        ys = [r["time_ms"] for r in rows]
        ax.plot(xs, ys, color=color, linewidth=0.8, alpha=0.8, label=label)

    ax.axhline(RT_BUDGET_MS, color="red", linestyle="--", linewidth=1.5,
               label=f"RT budget ({RT_BUDGET_MS:.0f} ms)")
    ax.set_xlabel("Window index", fontsize=11)
    ax.set_ylabel("Processing time (ms)", fontsize=11)
    ax.set_title("Per-window latency over stream", fontsize=12)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save_plot(fig, out_dir, "latency_over_time")


def plot_latency_cdf(datasets: dict[str, list[dict]], out_dir: Path) -> None:
    """Empirical CDF of per-window times, vertical lines at p95 and budget."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, rows in datasets.items():
        color = ENGINE_COLORS.get(label, "#333333")
        times = np.sort([r["time_ms"] for r in rows if np.isfinite(r["time_ms"])])
        cdf = np.arange(1, len(times) + 1) / len(times)
        ax.plot(times, cdf, color=color, linewidth=2, label=label)
        p95 = float(np.percentile(times, 95))
        ax.axvline(p95, color=color, linestyle=":", linewidth=1.5,
                   label=f"{label} p95={p95:.1f}ms")

    ax.axvline(RT_BUDGET_MS, color="red", linestyle="--", linewidth=2,
               label=f"RT budget ({RT_BUDGET_MS:.0f} ms)")
    ax.set_xlabel("Processing time (ms)", fontsize=11)
    ax.set_ylabel("Empirical CDF", fontsize=11)
    ax.set_title("Latency CDF", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    _save_plot(fig, out_dir, "latency_cdf")


def plot_memory_over_time(datasets: dict[str, list[dict]], out_dir: Path) -> None:
    """Peak memory (MiB) vs window index."""
    fig, ax = plt.subplots(figsize=(12, 4))

    for label, rows in datasets.items():
        color = ENGINE_COLORS.get(label, "#333333")
        xs = [r["window_index"] for r in rows]
        ys = [r["peak_memory_mib"] for r in rows]
        ax.plot(xs, ys, color=color, linewidth=0.8, alpha=0.8, label=label)

    ax.set_xlabel("Window index", fontsize=11)
    ax.set_ylabel("Peak memory (MiB)", fontsize=11)
    ax.set_title("Peak memory over stream", fontsize=12)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save_plot(fig, out_dir, "memory_over_time")


def plot_active_trajectories(datasets: dict[str, list[dict]], out_dir: Path) -> None:
    """Active trajectory count over window index."""
    fig, ax = plt.subplots(figsize=(12, 4))

    for label, rows in datasets.items():
        color = ENGINE_COLORS.get(label, "#333333")
        xs = [r["window_index"] for r in rows]
        ys = [r["active_trajectories"] for r in rows]
        ax.plot(xs, ys, color=color, linewidth=1.2, alpha=0.85, label=label)

    ax.set_xlabel("Window index", fontsize=11)
    ax.set_ylabel("Active trajectories", fontsize=11)
    ax.set_title("Active trajectory count over stream", fontsize=12)
    ax.legend(fontsize=9)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    fig.tight_layout()
    _save_plot(fig, out_dir, "active_trajectories_over_time")


def plot_latency_hist(datasets: dict[str, list[dict]], out_dir: Path) -> None:
    """Latency distribution histograms with symlog x-scale, one panel per engine."""
    labels = list(datasets.keys())
    fig, axes = plt.subplots(len(labels), 1, figsize=(8, 3 * len(labels)), sharex=True)
    if len(labels) == 1:
        axes = [axes]

    for ax, label in zip(axes, labels):
        rows = datasets[label]
        color = ENGINE_COLORS.get(label, "#333333")
        times = np.array([r["time_ms"] for r in rows if np.isfinite(r["time_ms"])])
        p95 = float(np.percentile(times, 95)) if len(times) > 0 else float("nan")
        ax.hist(times, bins=60, color=color, edgecolor="white", linewidth=0.4,
                alpha=0.85, label=label)
        if np.isfinite(p95):
            ax.axvline(p95, color="blue", linestyle="--", linewidth=1.2,
                       label=f"p95 = {p95:.1f} ms")
        ax.axvline(RT_BUDGET_MS, color="red", linestyle="--", linewidth=1.2,
                   label=f"RT budget = {RT_BUDGET_MS:.0f} ms")
        ax.set_xscale("symlog", linthresh=1)
        ax.set_ylabel("Count")
        ax.legend(fontsize=8, loc="upper right")

    axes[0].set_title("Per-window latency distribution (long stream)")
    axes[-1].set_xlabel("Per-window time (ms, symlog scale)")
    fig.tight_layout()
    _save_plot(fig, out_dir, "latency_hist")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot long-stream stress test results")
    parser.add_argument("--baseline",  default="results/long_stream/baseline")
    parser.add_argument("--optimized", default="results/long_stream/optimized_fwhm")
    parser.add_argument("--out",       default="results/long_stream/plots")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets: dict[str, list[dict]] = {}

    for label, dir_str in [
        ("Baseline SSD",     args.baseline),
        ("OptimizedSSD-fwhm", args.optimized),
    ]:
        csv_path = Path(dir_str) / "long_stream_metrics.csv"
        if csv_path.exists():
            rows = _read_csv(csv_path)
            datasets[label] = rows
            print(f"Loaded {len(rows)} rows for '{label}' from {csv_path}")
        else:
            print(f"[WARN] CSV not found: {csv_path} — skipping {label}")

    if not datasets:
        print("No data found. Run long_stream_test.py first.")
        return

    plot_latency_over_time(datasets, out_dir)
    plot_latency_cdf(datasets, out_dir)
    plot_latency_hist(datasets, out_dir)
    plot_memory_over_time(datasets, out_dir)
    plot_active_trajectories(datasets, out_dir)

    # Print summary
    for label, rows in datasets.items():
        times = [r["time_ms"] for r in rows if np.isfinite(r["time_ms"])]
        mems = [r["peak_memory_mib"] for r in rows if np.isfinite(r["peak_memory_mib"])]
        actives = [r["active_trajectories"] for r in rows]
        p95 = float(np.percentile(times, 95)) if times else float("nan")
        max_t = float(np.max(times)) if times else float("nan")
        peak_m = float(np.max(mems)) if mems else float("nan")
        print(
            f"{label}: p95={p95:.1f}ms, max={max_t:.1f}ms, "
            f"peak_mem={peak_m:.2f}MiB, "
            f"traj=[{int(np.min(actives))}, {int(np.max(actives))}]"
        )


if __name__ == "__main__":
    main()