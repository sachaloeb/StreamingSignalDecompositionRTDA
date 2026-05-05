"""Multi-seed SNR sweep for statistical QRF comparison across engines.

Generates snr_sweep.csv with one row per (signal, snr_db, engine, seed).
Used by snr_sweep_stats.py for paired Wilcoxon tests and BH correction.

Usage
-----
    python experiments/snr_sweep_multiseed.py --seeds 10 --out results/snr_sweep_multiseed
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

from experiments.synthetic.generators import (
    chirp_plus_sinusoid,
    component_onset,
    n_sinusoids,
    rossler,
    two_sinusoids,
)
from src.engines import get_engine
from src.metrics.stability import nmse, qrf
from src.streaming.component_matcher import ComponentMatcher
from src.streaming.trajectory_store import TrajectoryStore
from src.streaming.window_manager import WindowManager


# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------

SNR_LEVELS_DB = [5.0, 10.0, 15.0, 20.0, 30.0, 40.0]

ENGINES = [
    ("ssd",                "ssd",           {}),
    ("ssd_optimized_fwhm", "ssd_optimized", {"spectral_method": "fwhm"}),
    ("ssd_optimized_moment","ssd_optimized", {"spectral_method": "moment"}),
    ("ssd_optimized_gaussian","ssd_optimized",{"spectral_method": "gaussian"}),
]

FS = 1000.0
N = 3000
WINDOW_LEN = 300
STRIDE = 150

FIELDNAMES = [
    "signal", "snr_db", "engine", "seed",
    "median_qrf_db", "mean_nmse", "mean_ncomp",
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


def _generate(signal_name: str, snr_db: float, seed: int) -> np.ndarray:
    """Generate one signal realization."""
    rng = np.random.default_rng(seed)
    # Use seed+1000 offset so signal shape varies across seeds
    s = seed
    if signal_name == "two_sinusoids":
        return two_sinusoids(
            N=N, f1=50.0, f2=120.0, fs=FS, snr_db=snr_db, seed=s,
        )
    elif signal_name == "chirp_plus_sinusoid":
        return chirp_plus_sinusoid(
            N=N, f_sin=50.0, f_start=10.0, f_end=150.0,
            fs=FS, snr_db=snr_db, seed=s,
        )
    elif signal_name == "rossler":
        # Rossler has no snr_db param; add AWGN manually
        x = rossler(N=N, seed=s)
        if snr_db is not None:
            power_s = float(np.dot(x, x) / len(x))
            power_n = power_s / (10.0 ** (snr_db / 10.0))
            x = x + rng.normal(0.0, float(np.sqrt(max(power_n, 1e-30))), size=len(x))
        return x
    elif signal_name == "component_onset":
        x = component_onset(
            N=N, f_steady=50.0, f_onset=120.0,
            onset_sample=N // 2, fs=FS, seed=s,
        )
        if snr_db is not None:
            power_s = float(np.dot(x, x) / len(x))
            power_n = power_s / (10.0 ** (snr_db / 10.0))
            x = x + rng.normal(0.0, float(np.sqrt(max(power_n, 1e-30))), size=len(x))
        return x
    elif signal_name == "n_sinusoids":
        return n_sinusoids(
            N=N,
            frequencies=[20.0, 50.0, 80.0, 120.0, 200.0],
            fs=FS,
            snr_db=snr_db,
            seed=s,
        )
    else:
        raise ValueError(f"Unknown signal: {signal_name}")


def _run_one(
    signal: np.ndarray,
    engine_label: str,
    registry_key: str,
    engine_kwargs: dict,
) -> dict:
    """Run the streaming pipeline and return summary metrics."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        engine = get_engine(registry_key, fs=FS, **engine_kwargs)
        wm = WindowManager(window_len=WINDOW_LEN, stride=STRIDE, fs=FS)
        matcher = ComponentMatcher(
            distance="d_freq",freq_weight=1.0, fs=FS, lookback=5,
            max_cost=0.1, max_trajectories=12,
        )
        store = TrajectoryStore(max_components=12, max_len=N)

        qrf_vals: list[float] = []
        nmse_vals: list[float] = []
        ncomp_vals: list[int] = []
        overlap = wm.overlap

        for sample in signal:
            window = wm.push(float(sample))
            if window is None:
                continue
            components = engine.fit(window)
            components_no_res = components[:-1]
            residual = components[-1]

            matching = dict(matcher.match_stateful(components_no_res, overlap))
            window_start = len(signal) - WINDOW_LEN  # simplified; store handles bounds
            store.update(window_start, components_no_res, matching, overlap)

            ncomp_vals.append(len(components_no_res))
            recon = (
                np.sum(components_no_res, axis=0)
                if components_no_res else np.zeros_like(window)
            )
            q = qrf(window, recon)
            if np.isfinite(q):
                qrf_vals.append(q)
            nm = nmse(residual - float(np.mean(window)), window)
            if np.isfinite(nm):
                nmse_vals.append(nm)

    return {
        "median_qrf_db": float(np.median(qrf_vals)) if qrf_vals else float("nan"),
        "mean_nmse": float(np.mean(nmse_vals)) if nmse_vals else float("nan"),
        "mean_ncomp": float(np.mean(ncomp_vals)) if ncomp_vals else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed SNR sweep")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--out", type=str, default="results/snr_sweep_multiseed")
    parser.add_argument(
        "--signals",
        nargs="+",
        default=["two_sinusoids", "chirp_plus_sinusoid", "rossler",
                 "component_onset", "n_sinusoids"],
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = list(range(args.seeds))
    signal_names = args.signals

    total = len(signal_names) * len(SNR_LEVELS_DB) * len(ENGINES) * len(seeds)
    print(
        f"SNR sweep: {len(signal_names)} signals × {len(SNR_LEVELS_DB)} SNRs × "
        f"{len(ENGINES)} engines × {len(seeds)} seeds = {total} runs"
    )

    wall_start = time.perf_counter()
    rows: list[dict] = []

    for sig_name in signal_names:
        for snr_db in SNR_LEVELS_DB:
            for eng_label, reg_key, eng_kwargs in ENGINES:
                for seed in seeds:
                    signal = _generate(sig_name, snr_db, seed)
                    metrics = _run_one(signal, eng_label, reg_key, eng_kwargs)
                    rows.append({
                        "signal": sig_name,
                        "snr_db": snr_db,
                        "engine": eng_label,
                        "seed": seed,
                        "median_qrf_db": metrics["median_qrf_db"],
                        "mean_nmse": metrics["mean_nmse"],
                        "mean_ncomp": metrics["mean_ncomp"],
                    })

                # Print progress for each (signal, snr, engine) group
                group_qrfs = [
                    r["median_qrf_db"] for r in rows
                    if r["signal"] == sig_name
                    and r["snr_db"] == snr_db
                    and r["engine"] == eng_label
                ]
                valid_qrfs = [q for q in group_qrfs if np.isfinite(q)]
                mean_q = float(np.mean(valid_qrfs)) if valid_qrfs else float("nan")
                elapsed = time.perf_counter() - wall_start
                print(
                    f"  {sig_name:20s} SNR={snr_db:5.1f}dB {eng_label:25s} "
                    f"mean_QRF={mean_q:.2f}dB  ({elapsed:.0f}s elapsed)"
                )

                # Wall time guard
                if elapsed > 55 * 60:  # 55 minutes
                    print(
                        "\n[WARNING] Wall time approaching 60 min — saving what we have."
                    )
                    break
                else:
                    continue
                break
            else:
                continue
            break
        else:
            continue
        break

    wall_total = time.perf_counter() - wall_start
    csv_path = out_dir / "snr_sweep.csv"

    # Save CSV (raw data first, per brief rule 8)
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved to {csv_path}  ({len(rows)} rows)")

    # Run summary
    summary = {
        "seeds": seeds,
        "signal_names": signal_names,
        "snr_levels_db": SNR_LEVELS_DB,
        "engines": [e[0] for e in ENGINES],
        "N": N,
        "window_len": WINDOW_LEN,
        "stride": STRIDE,
        "fs": FS,
        "total_rows": len(rows),
        "total_wall_time_s": wall_total,
        "wall_time_note": (
            "Completed full sweep." if wall_total < 55 * 60
            else "Interrupted at 55-min wall-time guard; partial results saved."
        ),
        "environment": _get_env(),
    }
    summary_path = out_dir / "run_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Summary saved to {summary_path}")
    print(f"Total wall time: {wall_total:.0f}s")


if __name__ == "__main__":
    main()
