"""Sensitivity sweep: max_components × engines × seeds.

Sweeps max_components on chirp_plus_sinusoid only.  Records median QRF,
mean per-window time, mean NMSE, and mean component count.

Usage
-----
    python experiments/sensitivity_max_components.py [--seeds 5] \
        [--out results/sensitivity/max_components]
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

from experiments.synthetic.generators import chirp_plus_sinusoid
from src.engines import get_engine
from src.metrics.stability import qrf
from src.streaming.component_matcher import ComponentMatcher
from src.streaming.trajectory_store import TrajectoryStore
from src.streaming.window_manager import WindowManager

# ── Grid ──────────────────────────────────────────────────────────
MAX_COMPONENTS_VALUES = [3, 5, 10, 20, 50]
ENGINES = {
    "ssd":                  ("ssd",            {}),
    "ssd_optimized_fwhm":   ("ssd_optimized",  {"spectral_method": "fwhm"}),
}

FS = 1000.0
WINDOW_LEN = 300
STRIDE = 150
N_SIGNAL = 3000
SNR_DB = 20.0
NMSE_THRESHOLD = 0.01


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
    max_components: int,
) -> dict:
    """Run one (signal, engine, max_components) cell and return metrics."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        engine = get_engine(
            engine_key, fs=FS,
            nmse_threshold=NMSE_THRESHOLD,
            max_iter=max_components,
            **engine_kwargs,
        )

    wm = WindowManager(window_len=WINDOW_LEN, stride=STRIDE, fs=FS)
    matcher = ComponentMatcher(
        distance="d_freq", freq_weight=1.0, fs=FS,
        lookback=10, max_cost=0.1, max_trajectories=max_components,
    )
    store = TrajectoryStore(max_components=max_components, max_len=len(signal))
    overlap = wm.overlap

    window_times: list[float] = []
    qrf_vals: list[float] = []
    nmse_vals: list[float] = []
    ncomp_vals: list[int] = []

    for sample_idx in range(len(signal)):
        window = wm.push(float(signal[sample_idx]))
        if window is None:
            continue

        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            components = engine.fit(window)
        components_no_res = components[:-1]
        # Cap to max_components
        if len(components_no_res) > max_components:
            components_no_res = components_no_res[:max_components]
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

        sig_power = float(np.dot(window, window))
        err = window - recon
        nmse_val = float(np.dot(err, err)) / sig_power if sig_power > 0 else 0.0
        nmse_vals.append(nmse_val)
        ncomp_vals.append(len(components_no_res))

    return {
        "median_qrf_db": float(np.median(qrf_vals)) if qrf_vals else float("nan"),
        "mean_time_ms": float(np.mean(window_times)) if window_times else float("nan"),
        "mean_nmse": float(np.mean(nmse_vals)) if nmse_vals else float("nan"),
        "mean_ncomp": float(np.mean(ncomp_vals)) if ncomp_vals else float("nan"),
        "n_windows": len(window_times),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sensitivity sweep: max_components",
    )
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument(
        "--out", type=str, default="results/sensitivity/max_components",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)

    seeds = list(range(args.seeds))
    total_cells = len(MAX_COMPONENTS_VALUES) * len(ENGINES) * len(seeds)
    print(
        f"Sensitivity sweep: {len(MAX_COMPONENTS_VALUES)} max_components × "
        f"{len(ENGINES)} engines × {len(seeds)} seeds = {total_cells} cells"
    )

    # Time-guard
    sig0 = chirp_plus_sinusoid(
        N=N_SIGNAL, f_sin=50.0, f_start=10.0, f_end=150.0,
        fs=FS, snr_db=SNR_DB, seed=0,
    )
    t_check = time.perf_counter()
    run_one_cell(sig0, "ssd", {}, 10)
    t_check = time.perf_counter() - t_check
    print(f"Single-cell time: {t_check:.1f}s")
    if t_check > 10.0:
        seeds = seeds[:3]
        print(f"[WARN] Cell > 10s — reducing to {len(seeds)} seeds")
        total_cells = len(MAX_COMPONENTS_VALUES) * len(ENGINES) * len(seeds)

    rows: list[dict] = []
    wall_start = time.perf_counter()
    done = 0

    for mc in MAX_COMPONENTS_VALUES:
        for engine_label, (engine_key, engine_kwargs) in ENGINES.items():
            for seed in seeds:
                signal = chirp_plus_sinusoid(
                    N=N_SIGNAL, f_sin=50.0, f_start=10.0, f_end=150.0,
                    fs=FS, snr_db=SNR_DB, seed=seed,
                )
                result = run_one_cell(signal, engine_key, engine_kwargs, mc)
                rows.append({
                    "max_components": mc,
                    "engine": engine_label,
                    "signal": "chirp_plus_sinusoid",
                    "seed": seed,
                    **result,
                })
                done += 1
                if done % 5 == 0:
                    print(f"  {done}/{total_cells} cells done", flush=True)

    wall_total = time.perf_counter() - wall_start

    # Save CSV
    fieldnames = [
        "max_components", "engine", "signal", "seed",
        "median_qrf_db", "mean_time_ms", "mean_nmse", "mean_ncomp",
        "n_windows",
    ]
    csv_path = out_dir / "max_components_grid.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved: {csv_path} ({len(rows)} rows)")

    # Save run_summary.json
    summary = {
        "experiment": "sensitivity_max_components",
        "max_components_values": MAX_COMPONENTS_VALUES,
        "engines": list(ENGINES.keys()),
        "signals": ["chirp_plus_sinusoid"],
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


if __name__ == "__main__":
    main()
