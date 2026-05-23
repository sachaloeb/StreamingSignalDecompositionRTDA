"""Long-stream stress test for streaming SSD pipeline.

Validates the real-time claim on a 60-second stream (N=60000 at fs=1000).
Records per-window processing time, memory, QRF, matching confidence,
and active trajectory count.

Usage
-----
    # Baseline engine
    python experiments/long_stream_test.py \\
        --engine ssd \\
        --out results/long_stream/baseline

    # Optimized engine
    python experiments/long_stream_test.py \\
        --engine ssd_optimized_fwhm \\
        --out results/long_stream/optimized_fwhm
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

from experiments.synthetic.generators import chirp_plus_sinusoid
from src.engines import get_engine
from src.metrics.similarity import d_corr
from src.metrics.stability import qrf
from src.streaming.component_matcher import ComponentMatcher
from src.streaming.trajectory_store import TrajectoryStore
from src.streaming.window_manager import WindowManager

# Real-time budget: stride / fs seconds
# At stride=150, fs=1000: T_w = 0.150 s = 150 ms
RT_BUDGET_MS = 150.0

FIELDNAMES = [
    "window_index",
    "time_ms",
    "peak_memory_mib",
    "qrf_db",
    "matching_confidence",
    "active_trajectories",
]


def _get_env() -> dict:
    import scipy
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "cpu_model": platform.processor() or "unknown",
        "os": platform.platform(),
    }


def _resolve_engine(engine_label: str) -> tuple[str, dict]:
    """Map engine label to (registry_key, kwargs)."""
    mapping = {
        "ssd":                  ("ssd",           {}),
        "ssd_optimized_fwhm":   ("ssd_optimized", {"spectral_method": "fwhm"}),
        "ssd_optimized_moment": ("ssd_optimized", {"spectral_method": "moment"}),
        "ssd_optimized_gaussian":("ssd_optimized",{"spectral_method": "gaussian"}),
        "ssd_rsvd":             ("ssd_rsvd",      {}),
        "ssd_rank1":            ("ssd_rank1",     {}),
        "ssd_shsvd":            ("ssd_shsvd",     {}),
        "ssd_grouse":           ("ssd_grouse",    {}),
    }
    if engine_label not in mapping:
        raise ValueError(
            f"Unknown engine label '{engine_label}'. "
            f"Available: {sorted(mapping)}"
        )
    return mapping[engine_label]


def main() -> None:
    parser = argparse.ArgumentParser(description="Long-stream stress test")
    parser.add_argument("--N", type=int, default=60_000)
    parser.add_argument("--fs", type=float, default=1000.0)
    parser.add_argument("--window-len", type=int, default=300)
    parser.add_argument("--stride", type=int, default=150)
    parser.add_argument(
        "--engine",
        type=str,
        default="ssd_optimized_fwhm",
        choices=["ssd", "ssd_optimized_fwhm", "ssd_optimized_moment",
                 "ssd_optimized_gaussian", "ssd_rsvd", "ssd_rank1",
                 "ssd_shsvd", "ssd_grouse"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="results/long_stream")
    parser.add_argument("--snr-db", type=float, default=20.0)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    N = args.N
    fs = args.fs
    window_len = args.window_len
    stride = args.stride
    engine_label = args.engine
    seed = args.seed
    snr_db = args.snr_db

    budget_ms = stride / fs * 1000.0  # inter-window arrival period
    print(
        f"Long-stream test: N={N}, fs={fs}, window={window_len}, stride={stride}, "
        f"engine={engine_label}, seed={seed}, SNR={snr_db}dB"
    )
    print(f"Real-time budget: {budget_ms:.1f} ms  ({stride}/{int(fs)} s)")
    print(f"Expected windows: {(N - window_len) // stride + 1}")

    # Generate signal
    signal = chirp_plus_sinusoid(
        N=N, f_sin=50.0, f_start=10.0, f_end=150.0,
        fs=fs, snr_db=snr_db, seed=seed,
    )

    # Build pipeline
    registry_key, engine_kwargs = _resolve_engine(engine_label)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        engine = get_engine(registry_key, fs=fs, **engine_kwargs)
    wm = WindowManager(window_len=window_len, stride=stride, fs=fs)
    matcher = ComponentMatcher(
        distance="d_freq",freq_weight=1.0, fs=fs, lookback=10,
        max_cost=0.1, max_trajectories=12,
    )
    store = TrajectoryStore(max_components=12, max_len=N)
    overlap = wm.overlap

    rows: list[dict] = []
    prev_components: list[np.ndarray] | None = None
    window_idx = 0

    print("Running... ", flush=True)
    wall_start = time.perf_counter()

    for sample_idx in range(N):
        window = wm.push(float(signal[sample_idx]))
        if window is None:
            continue

        tracemalloc.start()
        t0 = time.perf_counter()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            components = engine.fit(window)

        components_no_res = components[:-1]
        matching = dict(matcher.match_stateful(components_no_res, overlap))
        window_start = sample_idx - window_len + 1
        store.update(window_start, components_no_res, matching, overlap)

        t1 = time.perf_counter()
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        time_ms = (t1 - t0) * 1000.0
        peak_mib = peak_mem / (1024 * 1024)

        # QRF
        recon = (
            np.sum(components_no_res, axis=0)
            if components_no_res else np.zeros_like(window)
        )
        qrf_val = qrf(window, recon)

        # Matching confidence
        if prev_components is not None and components_no_res:
            prev_match = matcher.previous_window_mapping()
            if prev_match:
                cost = matcher.build_cost_matrix(prev_components, components_no_res, overlap)
                confs = [
                    1.0 - cost[ci, pj]
                    for ci, pj in prev_match.items()
                    if pj is not None
                    and ci < cost.shape[0] and pj < cost.shape[1]
                ]
                mc_val = float(np.mean(confs)) if confs else float("nan")
            else:
                mc_val = float("nan")
        else:
            mc_val = float("nan")

        # Active trajectories
        active_traj = len(set(matching.values()) - {None})

        rows.append({
            "window_index": window_idx,
            "time_ms": round(time_ms, 3),
            "peak_memory_mib": round(peak_mib, 4),
            "qrf_db": round(qrf_val, 3) if np.isfinite(qrf_val) else float("nan"),
            "matching_confidence": round(mc_val, 4) if np.isfinite(mc_val) else float("nan"),
            "active_trajectories": active_traj,
        })

        prev_components = components_no_res
        window_idx += 1

        if window_idx % 50 == 0:
            print(
                f"  Window {window_idx:4d}: {time_ms:.1f}ms, "
                f"mem={peak_mib:.2f}MiB, QRF={qrf_val:.1f}dB",
                flush=True,
            )

    wall_total = time.perf_counter() - wall_start
    print(f"\nCompleted {window_idx} windows in {wall_total:.1f}s")

    # Save CSV
    csv_path = out_dir / "long_stream_metrics.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                k: ("nan" if isinstance(v, float) and not np.isfinite(v) else v)
                for k, v in row.items()
            })
    print(f"Metrics saved to {csv_path}  ({len(rows)} rows)")

    # Compute summary statistics
    times = [r["time_ms"] for r in rows]
    mems = [r["peak_memory_mib"] for r in rows]
    active = [r["active_trajectories"] for r in rows]

    p50 = float(np.percentile(times, 50))
    p95 = float(np.percentile(times, 95))
    p99 = float(np.percentile(times, 99))
    max_t = float(np.max(times))
    mean_t = float(np.mean(times))
    med_t = float(np.median(times))

    rt_pass = p95 <= budget_ms

    summary = {
        "engine": engine_label,
        "seed": seed,
        "N": N,
        "fs": fs,
        "window_len": window_len,
        "stride": stride,
        "snr_db": snr_db,
        "n_windows": window_idx,
        "budget_ms": budget_ms,
        "mean_time_ms": round(mean_t, 3),
        "median_time_ms": round(med_t, 3),
        "p50_time_ms": round(p50, 3),
        "p95_time_ms": round(p95, 3),
        "p99_time_ms": round(p99, 3),
        "max_time_ms": round(max_t, 3),
        "rt_pass_p95": rt_pass,
        "peak_memory_mib_max": round(float(np.max(mems)), 4),
        "peak_memory_mib_mean": round(float(np.mean(mems)), 4),
        "active_traj_min": int(np.min(active)),
        "active_traj_max": int(np.max(active)),
        "total_wall_time_s": round(wall_total, 2),
        "environment": _get_env(),
    }
    summary_path = out_dir / "run_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Summary saved to {summary_path}")

    # Real-time status
    status = "PASS" if rt_pass else "FAIL"
    print(
        f"\nReal-time status: {status}  "
        f"(p95={p95:.1f}ms vs budget={budget_ms:.1f}ms)"
    )
    if not rt_pass:
        print(
            "[STOP] p95 exceeds real-time budget — this is a thesis-level finding "
            "that needs human attention. Do not adjust parameters; report this to the author."
        )

    # Memory bound check: linear growth warning
    if len(mems) > 50:
        first_50 = float(np.mean(mems[:50]))
        last_50 = float(np.mean(mems[-50:]))
        growth_ratio = last_50 / first_50 if first_50 > 0 else 0.0
        if growth_ratio > 2.0:
            print(
                f"\n[WARNING] Peak memory grew {growth_ratio:.1f}× from first to last "
                f"50 windows ({first_50:.2f} → {last_50:.2f} MiB). "
                "This may indicate a streaming-pipeline memory leak — "
                "investigate before submitting thesis."
            )

    print(
        f"\nSummary: {engine_label}: p95={p95:.1f}ms, max={max_t:.1f}ms, "
        f"peak_mem={float(np.max(mems)):.2f}MiB, "
        f"active_trajectories=[{int(np.min(active))}, {int(np.max(active))}]"
    )


if __name__ == "__main__":
    main()
