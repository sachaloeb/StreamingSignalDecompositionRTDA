"""Sensitivity sweep: nmse_threshold × engines × signals × seeds.

Records median QRF, mean per-window time, mean NMSE, and mean component
count for each cell.  Output CSV and run_summary.json.

Usage
-----
    python experiments/sensitivity_nmse_threshold.py [--seeds 5] \
        [--out results/sensitivity/nmse_threshold]
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.synthetic.generators import chirp_plus_sinusoid, n_sinusoids
from src.engines import get_engine
from src.metrics.stability import qrf
from src.streaming.component_matcher import ComponentMatcher
from src.streaming.trajectory_store import TrajectoryStore
from src.streaming.window_manager import WindowManager


# ── Grid ──────────────────────────────────────────────────────────
THRESHOLDS = [0.005, 0.01, 0.02, 0.05, 0.1]
ENGINES = {
    "ssd":                  ("ssd",            {}),
    "ssd_optimized_fwhm":   ("ssd_optimized",  {"spectral_method": "fwhm"}),
}
SIGNAL_FACTORIES = {
    "chirp_plus_sinusoid": lambda seed: chirp_plus_sinusoid(
        N=3000, f_sin=50.0, f_start=10.0, f_end=150.0,
        fs=1000.0, snr_db=20.0, seed=seed,
    ),
    "n_sinusoids": lambda seed: n_sinusoids(
        N=3000, frequencies=[20.0, 50.0, 120.0],
        fs=1000.0, snr_db=20.0, seed=seed,
    ),
}

FS = 1000.0
WINDOW_LEN = 300
STRIDE = 150
MAX_COMPONENTS = 100


def _get_env() -> dict:
    import scipy
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "cpu_model": platform.processor() or "unknown",
        "os": platform.platform(),
    }


def run_one_cell(
    signal: np.ndarray,
    engine_key: str,
    engine_kwargs: dict,
    nmse_threshold: float,
) -> dict:
    """Run one (signal, engine, threshold) cell and return metrics."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        engine = get_engine(
            engine_key, fs=FS,
            nmse_threshold=nmse_threshold,
            **engine_kwargs,
        )

    wm = WindowManager(window_len=WINDOW_LEN, stride=STRIDE, fs=FS)
    matcher = ComponentMatcher(
        distance="d_freq", freq_weight=1.0, fs=FS,
        lookback=10, max_cost=0.1, max_trajectories=MAX_COMPONENTS,
    )
    store = TrajectoryStore(max_components=MAX_COMPONENTS, max_len=len(signal))
    overlap = wm.overlap

    window_times: list[float] = []
    qrf_vals: list[float] = []
    nmse_vals: list[float] = []
    ncomp_vals: list[int] = []

    prev_components = None
    for sample_idx in range(len(signal)):
        window = wm.push(float(signal[sample_idx]))
        if window is None:
            continue

        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            components = engine.fit(window)
        components_no_res = components[:-1]
        matching = dict(matcher.match_stateful(components_no_res, overlap))
        window_start = sample_idx - WINDOW_LEN + 1
        store.update(window_start, components_no_res, matching, overlap)
        t1 = time.perf_counter()

        window_times.append((t1 - t0) * 1000.0)
        recon = (
            np.sum(components_no_res, axis=0)
            if components_no_res else np.zeros_like(window)
        )
        qrf_vals.append(qrf(window, recon))

        # NMSE = ||x - x_hat||^2 / ||x||^2
        sig_power = float(np.dot(window, window))
        err = window - recon
        nmse_val = float(np.dot(err, err)) / sig_power if sig_power > 0 else 0.0
        nmse_vals.append(nmse_val)
        ncomp_vals.append(len(components_no_res))

        prev_components = components_no_res

    return {
        "median_qrf_db": float(np.median(qrf_vals)) if qrf_vals else float("nan"),
        "mean_time_ms": float(np.mean(window_times)) if window_times else float("nan"),
        "mean_nmse": float(np.mean(nmse_vals)) if nmse_vals else float("nan"),
        "mean_ncomp": float(np.mean(ncomp_vals)) if ncomp_vals else float("nan"),
        "n_windows": len(window_times),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sensitivity sweep: nmse_threshold",
    )
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument(
        "--out", type=str, default="results/sensitivity/nmse_threshold",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)

    seeds = list(range(args.seeds))
    total_cells = (
        len(THRESHOLDS) * len(ENGINES) * len(SIGNAL_FACTORIES) * len(seeds)
    )
    print(
        f"Sensitivity sweep: {len(THRESHOLDS)} thresholds × "
        f"{len(ENGINES)} engines × {len(SIGNAL_FACTORIES)} signals × "
        f"{len(seeds)} seeds = {total_cells} cells"
    )

    # Time-guard: run a single cell and check
    sig0 = SIGNAL_FACTORIES["chirp_plus_sinusoid"](0)
    t_check = time.perf_counter()
    run_one_cell(sig0, "ssd", {}, 0.01)
    t_check = time.perf_counter() - t_check
    print(f"Single-cell time: {t_check:.1f}s")
    if t_check > 10.0:
        seeds = seeds[:3]
        print(f"[WARN] Cell > 10s — reducing to {len(seeds)} seeds")
        total_cells = (
            len(THRESHOLDS) * len(ENGINES) * len(SIGNAL_FACTORIES) * len(seeds)
        )

    rows: list[dict] = []
    wall_start = time.perf_counter()
    done = 0

    for threshold in THRESHOLDS:
        for engine_label, (engine_key, engine_kwargs) in ENGINES.items():
            for signal_name, signal_factory in SIGNAL_FACTORIES.items():
                for seed in seeds:
                    signal = signal_factory(seed)
                    result = run_one_cell(
                        signal, engine_key, engine_kwargs, threshold,
                    )
                    rows.append({
                        "threshold": threshold,
                        "engine": engine_label,
                        "signal": signal_name,
                        "seed": seed,
                        **result,
                    })
                    done += 1
                    if done % 10 == 0:
                        print(f"  {done}/{total_cells} cells done", flush=True)

    wall_total = time.perf_counter() - wall_start

    # Save CSV
    fieldnames = [
        "threshold", "engine", "signal", "seed",
        "median_qrf_db", "mean_time_ms", "mean_nmse", "mean_ncomp",
        "n_windows",
    ]
    csv_path = out_dir / "nmse_threshold_grid.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved: {csv_path} ({len(rows)} rows)")

    # Save run_summary.json
    summary = {
        "experiment": "sensitivity_nmse_threshold",
        "thresholds": THRESHOLDS,
        "engines": list(ENGINES.keys()),
        "signals": list(SIGNAL_FACTORIES.keys()),
        "seeds": seeds,
        "total_cells": len(rows),
        "total_wall_time_s": round(wall_total, 2),
        "environment": _get_env(),
    }
    summary_path = out_dir / "run_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Summary saved: {summary_path}")
    print(f"Wall time: {wall_total:.1f}s ({wall_total/60:.1f}min)")

    # Print sample row
    sample = [r for r in rows if r["threshold"] == 0.01 and r["seed"] == 0]
    if sample:
        r = sample[0]
        print(
            f"\nSample (threshold=0.01, seed=0, {r['engine']}, {r['signal']}): "
            f"QRF={r['median_qrf_db']:.1f}dB, time={r['mean_time_ms']:.1f}ms, "
            f"NMSE={r['mean_nmse']:.4f}, ncomp={r['mean_ncomp']:.1f}"
        )


if __name__ == "__main__":
    main()
