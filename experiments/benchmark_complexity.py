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

from experiments.synthetic.generators import chirp_plus_sinusoid, n_sinusoids
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
    if engine_name == "ssd_rank1" and "stride" not in engine_kwargs:
        engine_kwargs = {**engine_kwargs, "stride": stride}
    engine = get_engine(engine_name, fs=fs, **engine_kwargs)
    matcher = ComponentMatcher(
        distance="d_corr", fs=fs, lookback=10,
        max_cost=0.6, max_trajectories=max_components,
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


def _benchmark_components(
    n_components_list: list[int],
    engine_name: str,
    window_len: int,
    stride: int,
    fs: float = 1000.0,
    N: int = 10000,
    snr_db: float | None = None,
    **engine_kwargs: object,
) -> list[dict[str, object]]:
    """Sweep over number of sinusoidal components at fixed window size.

    Frequencies are evenly spaced between 20 Hz and 400 Hz.

    Parameters
    ----------
    n_components_list : list[int]
        Numbers of sinusoidal components to test.
    engine_name : str
        Engine identifier for :func:`get_engine`.
    window_len : int
        Window length in samples.
    stride : int
        Stride in samples.
    fs : float
        Sampling frequency in Hz.
    N : int
        Signal length in samples.
    snr_db : float or None
        SNR for AWGN corruption; ``None`` means clean signal.
    **engine_kwargs
        Extra arguments forwarded to the engine constructor.

    Returns
    -------
    list[dict[str, object]]
        One row per (n_components, engine) combination.
    """
    rows: list[dict[str, object]] = []
    for n_comp in n_components_list:
        freqs = list(np.linspace(20.0, 400.0, n_comp))
        signal = n_sinusoids(N=N, frequencies=freqs, fs=fs, snr_db=snr_db)
        row = _benchmark_config(
            signal, engine_name, window_len, stride, fs=fs,
            max_components=n_comp + 2, **engine_kwargs,
        )
        row["n_components"] = n_comp
        row["snr_db"] = snr_db if snr_db is not None else float("inf")
        rows.append(row)
        print(
            f"  n_comp={n_comp}, snr={snr_db}, "
            f"mean={row['mean_time_per_window_s']:.4f}s/window"
        )
    return rows


def _benchmark_noise(
    snr_db_list: list[float | None],
    engine_name: str,
    window_len: int,
    stride: int,
    fs: float = 1000.0,
    N: int = 10000,
    n_components: int = 3,
    **engine_kwargs: object,
) -> list[dict[str, object]]:
    """Sweep over SNR levels at fixed window size and component count.

    Parameters
    ----------
    snr_db_list : list[float or None]
        SNR values in dB to test; ``None`` means clean (no noise).
    engine_name : str
        Engine identifier for :func:`get_engine`.
    window_len : int
        Window length in samples.
    stride : int
        Stride in samples.
    fs : float
        Sampling frequency in Hz.
    N : int
        Signal length in samples.
    n_components : int
        Number of sinusoidal components in the test signal.
    **engine_kwargs
        Extra arguments forwarded to the engine constructor.

    Returns
    -------
    list[dict[str, object]]
        One row per (snr_db, engine) combination.
    """
    rows: list[dict[str, object]] = []
    freqs = list(np.linspace(20.0, 400.0, n_components))
    for snr_db in snr_db_list:
        signal = n_sinusoids(N=N, frequencies=freqs, fs=fs, snr_db=snr_db)
        row = _benchmark_config(
            signal, engine_name, window_len, stride, fs=fs,
            max_components=n_components + 2, **engine_kwargs,
        )
        row["n_components"] = n_components
        row["snr_db"] = snr_db if snr_db is not None else float("inf")
        rows.append(row)
        snr_label = f"{snr_db}dB" if snr_db is not None else "clean"
        print(
            f"  snr={snr_label}, "
            f"mean={row['mean_time_per_window_s']:.4f}s/window"
        )
    return rows


def main() -> None:
    """Run the benchmark sweep and generate outputs."""
    out_dir = ROOT / "results" / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)

    N = 10000
    fs = 1000.0
    signal = chirp_plus_sinusoid(
        N=N, f_sin=50.0, f_start=10.0, f_end=150.0, fs=fs,
    )

    window_lens = [100, 200, 400, 800, 1600, 3200, 6400]
    engine_configs = [
        ("ssd", "SSD", {}),
        ("ssd_incremental", "IncrementalSSD", {}),
        ("ssd_incremental", "rSVD-IncrementalSSD", {"use_rsvd": True}),
        ("ssd_rank1", "RankOneIncrementalSSD", {}),
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

    # Save window-length sweep CSV
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

    # Generate window-length plots
    _plot_results(results, out_dir)

    # ------------------------------------------------------------------
    # Sweep 2: varying number of components (SSD engine, fixed window)
    # ------------------------------------------------------------------
    print("\n--- Component count sweep ---")
    n_components_list = [1, 2, 3, 4, 5, 6]
    comp_results: list[dict[str, object]] = []
    for snr_db in [None, 20.0, 10.0]:
        snr_label = f"{snr_db}dB" if snr_db is not None else "clean"
        print(f"  SNR={snr_label}")
        rows = _benchmark_components(
            n_components_list,
            engine_name="ssd",
            window_len=300,
            stride=150,
            fs=fs,
            N=N,
            snr_db=snr_db,
        )
        for r in rows:
            r["snr_label"] = snr_label
        comp_results.extend(rows)

    comp_csv = out_dir / "complexity_vs_components.csv"
    comp_fieldnames = [
        "engine", "n_components", "snr_db", "snr_label", "window_len",
        "stride", "n_windows", "mean_time_per_window_s",
        "std_time_per_window_s", "total_runtime_s", "peak_memory_mib",
    ]
    with open(comp_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=comp_fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(comp_results)
    print(f"Component sweep saved to {comp_csv}")
    _plot_component_sweep(comp_results, out_dir)

    # ------------------------------------------------------------------
    # Sweep 3: varying noise level (SSD engine, fixed components)
    # ------------------------------------------------------------------
    print("\n--- Noise level sweep ---")
    snr_db_list: list[float | None] = [None, 40.0, 30.0, 20.0, 10.0, 5.0]
    noise_results: list[dict[str, object]] = []
    for n_comp in [2, 4]:
        print(f"  n_components={n_comp}")
        rows = _benchmark_noise(
            snr_db_list,
            engine_name="ssd",
            window_len=300,
            stride=150,
            fs=fs,
            N=N,
            n_components=n_comp,
        )
        for r in rows:
            r["n_comp_label"] = str(n_comp)
        noise_results.extend(rows)

    noise_csv = out_dir / "complexity_vs_noise.csv"
    noise_fieldnames = [
        "engine", "n_components", "n_comp_label", "snr_db", "window_len",
        "stride", "n_windows", "mean_time_per_window_s",
        "std_time_per_window_s", "total_runtime_s", "peak_memory_mib",
    ]
    with open(noise_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=noise_fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(noise_results)
    print(f"Noise sweep saved to {noise_csv}")
    _plot_noise_sweep(noise_results, out_dir)


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


def _plot_component_sweep(
    results: list[dict[str, object]],
    out_dir: Path,
) -> None:
    """Plot mean time per window vs number of components, grouped by SNR."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    snr_labels: list[str] = []
    for r in results:
        lbl = str(r.get("snr_label", ""))
        if lbl not in snr_labels:
            snr_labels.append(lbl)

    fig, ax = plt.subplots(figsize=(8, 5))
    for lbl in snr_labels:
        subset = [r for r in results if str(r.get("snr_label", "")) == lbl]
        subset.sort(key=lambda r: int(r["n_components"]))
        n_comps = np.array([r["n_components"] for r in subset], dtype=float)
        times = np.array([r["mean_time_per_window_s"] for r in subset], dtype=float)
        ax.plot(n_comps, times * 1000, "o-", label=f"SNR={lbl}")

    ax.set_xlabel("Number of sinusoidal components")
    ax.set_ylabel("Mean time per window (ms)")
    ax.set_title("Decomposition Time vs Component Count")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_dir / "time_vs_n_components.png", dpi=150)
    plt.close(fig)
    print(f"Component sweep plot saved to {out_dir / 'time_vs_n_components.png'}")


def _plot_noise_sweep(
    results: list[dict[str, object]],
    out_dir: Path,
) -> None:
    """Plot mean time per window vs SNR, grouped by component count."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_comp_labels: list[str] = []
    for r in results:
        lbl = str(r.get("n_comp_label", ""))
        if lbl not in n_comp_labels:
            n_comp_labels.append(lbl)

    fig, ax = plt.subplots(figsize=(8, 5))
    for lbl in n_comp_labels:
        subset = [r for r in results if str(r.get("n_comp_label", "")) == lbl]
        subset.sort(key=lambda r: float(r["snr_db"]))
        snrs = np.array([float(r["snr_db"]) for r in subset])
        times = np.array([r["mean_time_per_window_s"] for r in subset], dtype=float)
        x_labels = [
            "clean" if s == float("inf") else f"{s:.0f}"
            for s in snrs
        ]
        ax.plot(range(len(x_labels)), times * 1000, "o-", label=f"{lbl} components")
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels)

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Mean time per window (ms)")
    ax.set_title("Decomposition Time vs Noise Level")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_dir / "time_vs_noise.png", dpi=150)
    plt.close(fig)
    print(f"Noise sweep plot saved to {out_dir / 'time_vs_noise.png'}")


if __name__ == "__main__":
    main()