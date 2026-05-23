"""Sensitivity sweep: nmse_threshold OR max_components × engines × signals × seeds.

Merges the former sensitivity_nmse_threshold.py and sensitivity_max_components.py.
Records median QRF, mean per-window time, mean NMSE, and mean component count
for each cell.  Output CSV name and column name match what the downstream
plot_nmse_threshold_grid.py / plot_max_components_grid.py scripts expect.

Usage
-----
    python experiments/sensitivity_sweep.py --param nmse_threshold [--seeds 5] \
        [--out results/sensitivity/nmse_threshold]

    python experiments/sensitivity_sweep.py --param max_components [--seeds 5] \
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

from experiments.synthetic.generators import chirp_plus_sinusoid, n_sinusoids
from src.engines import get_engine
from src.metrics.stability import qrf
from src.streaming.component_matcher import ComponentMatcher
from src.streaming.trajectory_store import TrajectoryStore
from src.streaming.window_manager import WindowManager


# ── Shared pipeline constants ──────────────────────────────────────
ENGINES = {
    "ssd":                ("ssd",           {}),
    "ssd_optimized_fwhm": ("ssd_optimized", {"spectral_method": "fwhm"}),
}

FS = 1000.0
WINDOW_LEN = 300
STRIDE = 150

# ── Parameter-specific configuration ──────────────────────────────
_SIGNAL_FACTORIES_BOTH = {
    "chirp_plus_sinusoid": lambda seed: chirp_plus_sinusoid(
        N=3000, f_sin=50.0, f_start=10.0, f_end=150.0,
        fs=1000.0, snr_db=20.0, seed=seed,
    ),
    "n_sinusoids": lambda seed: n_sinusoids(
        N=3000, frequencies=[20.0, 50.0, 120.0],
        fs=1000.0, snr_db=20.0, seed=seed,
    ),
}

_SIGNAL_FACTORIES_CHIRP = {
    "chirp_plus_sinusoid": _SIGNAL_FACTORIES_BOTH["chirp_plus_sinusoid"],
}

PARAM_CONFIGS: dict[str, dict] = {
    "nmse_threshold": {
        "values": [0.005, 0.01, 0.02, 0.05, 0.1],
        "csv_name": "nmse_threshold_grid.csv",
        "col_name": "threshold",
        "signal_factories": _SIGNAL_FACTORIES_BOTH,
        "fixed_max_components": 100,
        "fixed_nmse_threshold": None,   # swept
    },
    "max_components": {
        "values": [3, 5, 10, 20, 50],
        "csv_name": "max_components_grid.csv",
        "col_name": "max_components",
        "signal_factories": _SIGNAL_FACTORIES_CHIRP,
        "fixed_max_components": None,   # swept
        "fixed_nmse_threshold": 0.01,
    },
}


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
    param_name: str,
    param_value: float | int,
) -> dict:
    """Run one (signal, engine, param_value) cell and return metrics."""
    cfg = PARAM_CONFIGS[param_name]
    nmse_threshold: float = (
        param_value if param_name == "nmse_threshold"
        else cfg["fixed_nmse_threshold"]
    )
    max_components: int = (
        param_value if param_name == "max_components"
        else cfg["fixed_max_components"]
    )

    extra_engine_kwargs: dict = (
        {"max_iter": max_components} if param_name == "max_components" else {}
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        engine = get_engine(
            engine_key, fs=FS,
            nmse_threshold=nmse_threshold,
            **extra_engine_kwargs,
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
        # When sweeping max_components, cap extracted components to the ceiling
        if param_name == "max_components" and len(components_no_res) > max_components:
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
        description="Sensitivity sweep over a single hyperparameter.",
    )
    parser.add_argument(
        "--param",
        choices=list(PARAM_CONFIGS.keys()),
        required=True,
        help="Hyperparameter to sweep.",
    )
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    cfg = PARAM_CONFIGS[args.param]
    out_dir = Path(args.out or f"results/sensitivity/{args.param}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)

    values = cfg["values"]
    signal_factories = cfg["signal_factories"]
    seeds = list(range(args.seeds))
    total_cells = len(values) * len(ENGINES) * len(signal_factories) * len(seeds)
    print(
        f"Sensitivity sweep ({args.param}): {len(values)} values × "
        f"{len(ENGINES)} engines × {len(signal_factories)} signals × "
        f"{len(seeds)} seeds = {total_cells} cells"
    )

    # Time-guard: run a single cell to estimate cost
    sig0 = next(iter(signal_factories.values()))(0)
    mid_value = values[len(values) // 2]
    t_check = time.perf_counter()
    run_one_cell(sig0, "ssd", {}, args.param, mid_value)
    t_check = time.perf_counter() - t_check
    print(f"Single-cell time: {t_check:.1f}s")
    if t_check > 10.0:
        seeds = seeds[:3]
        print(f"[WARN] Cell > 10s — reducing to {len(seeds)} seeds")
        total_cells = len(values) * len(ENGINES) * len(signal_factories) * len(seeds)

    rows: list[dict] = []
    wall_start = time.perf_counter()
    done = 0

    for value in values:
        for engine_label, (engine_key, engine_kwargs) in ENGINES.items():
            for signal_name, signal_factory in signal_factories.items():
                for seed in seeds:
                    signal = signal_factory(seed)
                    result = run_one_cell(
                        signal, engine_key, engine_kwargs, args.param, value,
                    )
                    rows.append({
                        cfg["col_name"]: value,
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
        cfg["col_name"], "engine", "signal", "seed",
        "median_qrf_db", "mean_time_ms", "mean_nmse", "mean_ncomp",
        "n_windows",
    ]
    csv_path = out_dir / cfg["csv_name"]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved: {csv_path} ({len(rows)} rows)")

    # Save run_summary.json
    summary = {
        "experiment": f"sensitivity_{args.param}",
        f"{args.param}_values": values,
        "engines": list(ENGINES.keys()),
        "signals": list(signal_factories.keys()),
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
    sample = [r for r in rows if r[cfg["col_name"]] == mid_value and r["seed"] == 0]
    if sample:
        r = sample[0]
        print(
            f"\nSample ({args.param}={mid_value}, seed=0, {r['engine']}, {r['signal']}): "
            f"QRF={r['median_qrf_db']:.1f}dB, time={r['mean_time_ms']:.1f}ms, "
            f"NMSE={r['mean_nmse']:.4f}, ncomp={r['mean_ncomp']:.1f}"
        )


if __name__ == "__main__":
    main()