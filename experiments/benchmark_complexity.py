"""Complexity and latency benchmarks for streaming decomposition engines.

Sweeps window_len and engine type, measuring per-window wall-clock time
and peak memory.  Generates CSV results and log-log scaling plots.

Usage
-----
    python experiments/benchmark_complexity.py
"""

from __future__ import annotations

import csv
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.synthetic.generators import chirp_plus_sinusoid
from src.engines import get_engine
from src.streaming.component_matcher import ComponentMatcher
from src.streaming.trajectory_store import TrajectoryStore
from src.streaming.window_manager import WindowManager


def _benchmark_config(
    signal: np.ndarray,
    engine_name: str,
    window_len: int,
    stride: int,
    fs: float = 1000.0,
    max_components: int = 10,
    **engine_kwargs: object,
) -> dict[str, object]:
    """Run one benchmark configuration and return metrics."""
    N = len(signal)
    wm = WindowManager(window_len=window_len, stride=stride, fs=fs)
    engine = get_engine(engine_name, fs=fs, **engine_kwargs)
    matcher = ComponentMatcher(
        distance="d_corr", fs=fs, lookback=3,
        max_cost=0.5, max_trajectories=max_components,
    )
    store = TrajectoryStore(max_components=max_components, max_len=N)

    window_times: list[float] = []

    tracemalloc.start()

    t_total_start = time.perf_counter()
    for sample_idx in range(N):
        window = wm.push(float(signal[sample_idx]))
        if window is None:
            continue

        t0 = time.perf_counter()

        components = engine.fit(window)
        components_no_res = components[:-1]
        matching = dict(
            matcher.match_stateful(components_no_res, wm.overlap)
        )
        window_start = sample_idx - wm.window_len + 1
        store.update(
            window_start, components_no_res, matching, wm.overlap,
        )

        t1 = time.perf_counter()
        window_times.append(t1 - t0)

    t_total_end = time.perf_counter()

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "engine": engine_name,
        "window_len": window_len,
        "stride": stride,
        "n_windows": len(window_times),
        "mean_time_per_window_s": float(np.mean(window_times)) if window_times else 0.0,
        "std_time_per_window_s": float(np.std(window_times)) if window_times else 0.0,
        "total_runtime_s": t_total_end - t_total_start,
        "peak_memory_mib": peak / (1024 * 1024),
    }


def main() -> None:
    """Run the benchmark sweep and generate outputs."""
    out_dir = ROOT / "results" / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)

    N = 10000
    fs = 1000.0
    signal = chirp_plus_sinusoid(
        N=N, f_sin=50.0, f_start=10.0, f_end=150.0, fs=fs,
    )

    window_lens = [100, 200, 400, 800, 1600]
    engine_configs = [
        ("ssd", "SSD", {}),
        ("ssd_incremental", "IncrementalSSD", {}),
        ("ssd_incremental", "rSVD-IncrementalSSD", {"use_rsvd": True}),
    ]

    results: list[dict[str, object]] = []

    for wl in window_lens:
        stride = wl // 2
        for engine_name, label, kwargs in engine_configs:
            print(f"Benchmarking {label}, window_len={wl}, stride={stride}...")
            row = _benchmark_config(
                signal, engine_name, wl, stride, fs=fs, **kwargs,
            )
            row["label"] = label
            results.append(row)
            print(
                f"  {row['n_windows']} windows, "
                f"mean={row['mean_time_per_window_s']:.4f}s/window, "
                f"peak_mem={row['peak_memory_mib']:.2f} MiB"
            )

    # Save CSV
    csv_path = out_dir / "complexity_results.csv"
    fieldnames = [
        "engine", "label", "window_len", "stride", "n_windows",
        "mean_time_per_window_s", "std_time_per_window_s",
        "total_runtime_s", "peak_memory_mib",
    ]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {csv_path}")

    # Generate plots
    _plot_results(results, out_dir)


def _plot_results(
    results: list[dict[str, object]],
    out_dir: Path,
) -> None:
    """Generate log-log scaling plots with empirical exponents."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels_seen: list[str] = []
    for r in results:
        if r["label"] not in labels_seen:
            labels_seen.append(r["label"])

    # --- Time per window vs window_len ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for label in labels_seen:
        subset = [r for r in results if r["label"] == label]
        wls = np.array([r["window_len"] for r in subset], dtype=float)
        times = np.array([r["mean_time_per_window_s"] for r in subset], dtype=float)

        ax.loglog(wls, times, "o-", label=label)

        # Fit empirical scaling exponent
        mask = times > 0
        if mask.sum() >= 2:
            coeffs = np.polyfit(np.log10(wls[mask]), np.log10(times[mask]), 1)
            ax.text(
                wls[mask][-1], times[mask][-1],
                f" α={coeffs[0]:.2f}",
                fontsize=9, va="bottom",
            )

    ax.set_xlabel("Window length")
    ax.set_ylabel("Mean time per window (s)")
    ax.set_title("Time per Window vs Window Length (log-log)")
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_dir / "time_vs_window_len.png", dpi=150)
    plt.close(fig)

    # --- Memory vs window_len ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for label in labels_seen:
        subset = [r for r in results if r["label"] == label]
        wls = np.array([r["window_len"] for r in subset], dtype=float)
        mems = np.array([r["peak_memory_mib"] for r in subset], dtype=float)

        ax.loglog(wls, mems, "s-", label=label)

    ax.set_xlabel("Window length")
    ax.set_ylabel("Peak memory (MiB)")
    ax.set_title("Peak Memory vs Window Length")
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_dir / "memory_vs_window_len.png", dpi=150)
    plt.close(fig)

    print(f"Plots saved to {out_dir}")


if __name__ == "__main__":
    main()