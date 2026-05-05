"""Cross-window benchmark grid for OptimizedSSD vs baseline engines.

Sweeps (engine, window_len, seed), records per-window timing and peak
memory, and writes complexity_grid.csv + run_summary.json.

Usage
-----
    python experiments/benchmark_optimized_grid.py \\
        --seeds 5 \\
        --out results/benchmarks_optimized \\
        --signal chirp_plus_sinusoid
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
import tracemalloc
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.engines.ssd_optimized as _opt_mod
from experiments.synthetic.generators import chirp_plus_sinusoid, two_sinusoids
from src.engines import get_engine
from src.streaming.component_matcher import ComponentMatcher
from src.streaming.trajectory_store import TrajectoryStore
from src.streaming.window_manager import WindowManager


# ---------------------------------------------------------------------------
# Engine configurations
# (engine_label, registry_key, extra_kwargs)
# ---------------------------------------------------------------------------
_ENGINE_CONFIGS = [
    ("ssd",                   "ssd",           {}),
    ("ssd_optimized_fwhm",    "ssd_optimized", {"spectral_method": "fwhm"}),
    ("ssd_optimized_moment",  "ssd_optimized", {"spectral_method": "moment"}),
    ("ssd_incremental",       "ssd_incremental", {}),
    ("ssd_rank1",             "ssd_rank1",     {}),
]

WINDOW_LENGTHS = [100, 200, 400, 800, 1600, 3200, 6400]

# Budget (seconds) per single (engine, window_len, seed) cell
_CELL_BUDGET_S = 60.0


def _get_env_info() -> dict:
    import scipy
    info = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "cpu_model": platform.processor() or "unknown",
        "os": platform.platform(),
    }
    return info


def _generate_signal(signal_type: str, N: int, fs: float, seed: int) -> np.ndarray:
    if signal_type == "chirp_plus_sinusoid":
        return chirp_plus_sinusoid(
            N=N, f_sin=50.0, f_start=10.0, f_end=150.0, fs=fs, seed=seed,
        )
    elif signal_type == "two_sinusoids":
        return two_sinusoids(
            N=N, f1=50.0, f2=120.0, fs=fs, seed=seed,
        )
    else:
        raise ValueError(f"Unknown signal type '{signal_type}'")


def _benchmark_one(
    signal: np.ndarray,
    engine_label: str,
    registry_key: str,
    engine_kwargs: dict,
    window_len: int,
    stride: int,
    fs: float = 1000.0,
    max_components: int = 10,
) -> dict:
    """Run one (engine, window_len) cell; return metrics dict."""
    N = len(signal)

    # Detect moment guard substitution
    if (
        registry_key == "ssd_optimized"
        and engine_kwargs.get("spectral_method") == "moment"
        and window_len < 256  # OptimizedSSD.min_window_length_for_moment
    ):
        actual_label = "ssd_optimized_moment_substituted_fwhm"
    else:
        actual_label = engine_label

    _opt_mod._MOMENT_GUARD_WARNED = False

    wm = WindowManager(window_len=window_len, stride=stride, fs=fs)
    kwargs = dict(engine_kwargs)
    if registry_key == "ssd_rank1" and "stride" not in kwargs:
        kwargs["stride"] = stride
    engine = get_engine(registry_key, fs=fs, **kwargs)
    matcher = ComponentMatcher(
        distance="d_freq",freq_weight=1.0, fs=fs, lookback=10,
        max_cost=0.1, max_trajectories=max_components,
    )
    store = TrajectoryStore(max_components=max_components, max_len=N)

    window_times: list[float] = []
    tracemalloc.start()

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        for sample_idx in range(N):
            window = wm.push(float(signal[sample_idx]))
            if window is None:
                continue

            t0 = time.perf_counter()
            components = engine.fit(window)
            components_no_res = components[:-1]
            matching = dict(matcher.match_stateful(components_no_res, wm.overlap))
            window_start = sample_idx - wm.window_len + 1
            store.update(window_start, components_no_res, matching, wm.overlap)
            t1 = time.perf_counter()
            window_times.append(t1 - t0)

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    times_ms = np.array(window_times) * 1000.0
    return {
        "engine": actual_label,
        "window_len": window_len,
        "stride": stride,
        "n_windows": len(window_times),
        "mean_time_ms": float(np.mean(times_ms)) if window_times else 0.0,
        "std_time_ms": float(np.std(times_ms)) if window_times else 0.0,
        "p95_time_ms": float(np.percentile(times_ms, 95)) if window_times else 0.0,
        "peak_memory_mib": peak / (1024 * 1024),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark OptimizedSSD grid")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--out", type=str, default="results/benchmarks_optimized")
    parser.add_argument(
        "--signal",
        choices=["chirp_plus_sinusoid", "two_sinusoids"],
        default="chirp_plus_sinusoid",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    fs = 1000.0
    n_seeds = args.seeds
    seeds = list(range(n_seeds))
    signal_type = args.signal

    env_info = _get_env_info()
    wall_start = time.perf_counter()

    rows: list[dict] = []
    total_cells = len(_ENGINE_CONFIGS) * len(WINDOW_LENGTHS) * n_seeds

    print(
        f"Benchmark grid: {len(_ENGINE_CONFIGS)} engines × "
        f"{len(WINDOW_LENGTHS)} window lengths × {n_seeds} seeds "
        f"= {total_cells} cells"
    )

    for eng_label, reg_key, eng_kwargs in _ENGINE_CONFIGS:
        for wl in WINDOW_LENGTHS:
            stride = wl // 2
            N = max(10_000, 4 * wl)
            for seed in seeds:
                signal = _generate_signal(signal_type, N, fs, seed)

                print(
                    f"  {eng_label:35s}  wl={wl:5d}  seed={seed}  N={N} ... ",
                    end="", flush=True,
                )
                t0 = time.perf_counter()
                row = _benchmark_one(
                    signal, eng_label, reg_key, eng_kwargs,
                    wl, stride, fs=fs,
                )
                elapsed = time.perf_counter() - t0
                row["seed"] = seed
                rows.append(row)
                print(
                    f"mean={row['mean_time_ms']:.2f}ms  "
                    f"p95={row['p95_time_ms']:.2f}ms  "
                    f"mem={row['peak_memory_mib']:.2f}MiB  "
                    f"({elapsed:.1f}s wall)"
                )

    # Save CSV
    csv_path = out_dir / "complexity_grid.csv"
    fieldnames = [
        "engine", "window_len", "stride", "seed", "n_windows",
        "mean_time_ms", "std_time_ms", "p95_time_ms", "peak_memory_mib",
    ]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved to {csv_path}  ({len(rows)} rows)")

    # Run summary
    wall_total = time.perf_counter() - wall_start
    summary = {
        "seeds": seeds,
        "n_seeds": n_seeds,
        "signal_type": signal_type,
        "window_lengths": WINDOW_LENGTHS,
        "engines": [e[0] for e in _ENGINE_CONFIGS],
        "total_rows": len(rows),
        "total_wall_time_s": wall_total,
        "environment": env_info,
    }
    summary_path = out_dir / "run_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Summary saved to {summary_path}")

    # Quick summary table
    print("\n--- Speedup summary (vs ssd baseline) at selected window lengths ---")
    by_engine_wl: dict[tuple, list[float]] = {}
    for row in rows:
        key = (row["engine"], row["window_len"])
        by_engine_wl.setdefault(key, []).append(row["mean_time_ms"])
    avg: dict[tuple, float] = {
        k: float(np.mean(v)) for k, v in by_engine_wl.items()
    }

    for wl_check in [200, 400, 1600]:
        baseline = avg.get(("ssd", wl_check))
        if baseline is None or baseline == 0:
            continue
        for eng_label, _, _ in _ENGINE_CONFIGS:
            if eng_label == "ssd":
                continue
            opt = avg.get((eng_label, wl_check))
            if opt is None or opt == 0:
                continue
            print(
                f"  {eng_label:35s}  wl={wl_check:5d}: "
                f"speedup = {baseline/opt:.2f}× "
                f"(baseline {baseline:.2f}ms → {opt:.2f}ms)"
            )

    # Scaling exponents
    print("\n--- Scaling exponents (α from log-log fit) ---")
    for eng_label, _, _ in _ENGINE_CONFIGS:
        wls_eng = sorted(
            set(row["window_len"] for row in rows if row["engine"] == eng_label)
        )
        times_eng = [avg.get((eng_label, wl), 0.0) for wl in wls_eng]
        valid = [(w, t) for w, t in zip(wls_eng, times_eng) if t > 0]
        if len(valid) >= 2:
            log_w = np.log10([v[0] for v in valid])
            log_t = np.log10([v[1] for v in valid])
            alpha = float(np.polyfit(log_w, log_t, 1)[0])
            print(f"  {eng_label:35s}  α = {alpha:.3f}")

    print(f"\nTotal wall time: {wall_total:.1f}s")


if __name__ == "__main__":
    main()