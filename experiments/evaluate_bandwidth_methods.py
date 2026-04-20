"""Evaluation suite for the three δf bandwidth estimation methods in OptimizedSSD.

Runs four evaluation levels:
  Level 1 — Unit δf accuracy table (12 regime × 3 method × 20 seeds)
  Level 2 — End-to-end reconstruction sweep (4 signals × 3 methods)
  Level 3 — Sparse-PSD stress sweep (7 window sizes × 3 methods)
  Level 4 — Latency micro-benchmark (2000 PSD evaluations per method)

Usage
-----
    python experiments/evaluate_bandwidth_methods.py
    python experiments/evaluate_bandwidth_methods.py --output-dir /tmp/bw_eval
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy.signal import welch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.engines.ssd_optimized import OptimizedSSD
from src.metrics.stability import qrf, nmse
from src.streaming.window_manager import WindowManager
from experiments.synthetic.generators import (
    two_sinusoids,
    chirp_plus_sinusoid,
    rossler,
    component_onset,
)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

METHODS: list[str] = ["fwhm", "moment", "gaussian"]


def _compute_psd(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute PSD using the same parameters as SSD._compute_psd."""
    N = len(x)
    nperseg = min(N, 256)
    freqs, psd_vals = welch(x, fs=fs, nperseg=nperseg, nfft=4096)
    return freqs, psd_vals


def _make_sinusoid_noisy(
    f0: float,
    fs: float,
    N: int,
    snr_db: float,
    seed: int,
) -> np.ndarray:
    """Generate a sinusoid with AWGN at the given SNR."""
    rng = np.random.default_rng(seed)
    t = np.arange(N) / fs
    x = np.sin(2.0 * np.pi * f0 * t)
    power_signal = float(np.dot(x, x) / N)
    power_noise = power_signal / (10.0 ** (snr_db / 10.0))
    noise_std = float(np.sqrt(max(power_noise, 1e-30)))
    return x + rng.normal(0.0, noise_std, size=N)


def _call_estimator_safe(
    method: str,
    psd: np.ndarray,
    freqs: np.ndarray,
) -> float:
    """Call estimator and return NaN on any exception (never abort the eval)."""
    try:
        if method == "fwhm":
            return float(OptimizedSSD._estimate_bandwidth_fwhm(psd, freqs))
        elif method == "moment":
            return float(OptimizedSSD._estimate_bandwidth_moment(psd, freqs))
        elif method == "gaussian":
            return float(OptimizedSSD._fit_gaussian_with_jacobian(psd, freqs))
    except Exception as exc:  # noqa: BLE001
        print(f"  [WARN] {method} raised: {exc}", file=sys.stderr)
        return float("nan")
    return float("nan")


def _run_streaming_pipeline(
    signal: np.ndarray,
    method: str,
    fs: float,
    window_len: int,
    stride: int,
) -> dict[str, object]:
    """Run the streaming pipeline; return per-window QRF and NMSE lists."""
    engine = OptimizedSSD(fs=fs, spectral_method=method)
    wm = WindowManager(window_len=window_len, stride=stride, fs=fs)
    qrf_vals: list[float] = []
    nmse_vals: list[float] = []
    n_windows = 0

    for sample in signal:
        window = wm.push(float(sample))
        if window is None:
            continue
        components = engine.fit(window)
        components_no_res = components[:-1]
        residual_comp = components[-1]

        recon = (
            np.sum(components_no_res, axis=0)
            if components_no_res
            else np.zeros_like(window)
        )
        qrf_val = qrf(window, recon)
        if np.isfinite(qrf_val):
            qrf_vals.append(qrf_val)
        nmse_val = nmse(residual_comp - float(np.mean(window)), window)
        if np.isfinite(nmse_val):
            nmse_vals.append(nmse_val)
        n_windows += 1

    return {
        "qrf_vals": qrf_vals,
        "nmse_vals": nmse_vals,
        "n_windows": n_windows,
    }


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    """Write rows to a CSV file; raise RuntimeError if write fails."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    except OSError as exc:
        raise RuntimeError(f"Failed to write CSV to {path}: {exc}") from exc


def _pivot_table(
    rows: list[dict],
    row_key: str,
    col_key: str,
    val_key: str,
    fmt: str = ".2f",
) -> list[str]:
    """Return lines of a simple ASCII pivot table."""
    row_vals = list(dict.fromkeys(r[row_key] for r in rows))
    col_vals = list(dict.fromkeys(r[col_key] for r in rows))
    lookup: dict[tuple, float] = {
        (r[row_key], r[col_key]): r[val_key] for r in rows
    }
    col_w = max(10, max(len(str(c)) for c in col_vals) + 2)
    row_w = max(20, max(len(str(r)) for r in row_vals) + 2)
    header = f"{'':>{row_w}}" + "".join(f"{str(c):>{col_w}}" for c in col_vals)
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for rv in row_vals:
        row_str = f"{str(rv):>{row_w}}"
        for cv in col_vals:
            val = lookup.get((rv, cv), float("nan"))
            if np.isfinite(val):
                row_str += f"{format(val, fmt):>{col_w}}"
            else:
                row_str += f"{'nan':>{col_w}}"
        lines.append(row_str)
    lines.append(sep)
    return lines


# ---------------------------------------------------------------------------
# Level 1 — Unit δf accuracy table
# ---------------------------------------------------------------------------

def run_level1(out_dir: Path) -> list[dict]:
    """Level 1: unit δf accuracy over 12 (fs, N, f0) regimes × 20 seeds."""
    print("\n" + "=" * 65)
    print("LEVEL 1 — Unit δf accuracy table")
    print("=" * 65)

    combos = [
        (250.0, 128, 25.0),
        (250.0, 512, 25.0),
        (250.0, 2048, 25.0),
        (500.0, 128, 50.0),
        (500.0, 512, 50.0),
        (500.0, 2048, 50.0),
        (1000.0, 128, 100.0),
        (1000.0, 512, 100.0),
        (1000.0, 2048, 100.0),
        # Additional f0 = fs/10 variants
        (250.0, 256, 25.0),
        (500.0, 256, 50.0),
        (1000.0, 256, 100.0),
    ]
    n_seeds = 20
    snr_db = 20.0
    rows: list[dict] = []

    for fs, N, f0 in combos:
        for method in METHODS:
            dfs: list[float] = []
            for seed in range(n_seeds):
                x = _make_sinusoid_noisy(f0, fs, N, snr_db, seed)
                freqs, psd = _compute_psd(x, fs)
                df = _call_estimator_safe(method, psd, freqs)
                dfs.append(df)

            finite_dfs = [v for v in dfs if np.isfinite(v)]
            nan_frac = 1.0 - len(finite_dfs) / len(dfs) if dfs else 1.0
            mean_df = float(np.mean(finite_dfs)) if finite_dfs else float("nan")
            std_df = float(np.std(finite_dfs)) if finite_dfs else float("nan")
            cv = (std_df / mean_df) if mean_df > 0 and np.isfinite(mean_df) else float("nan")
            floor = fs / N
            above_floor = (
                sum(1 for v in finite_dfs if v >= floor) / len(finite_dfs)
                if finite_dfs else float("nan")
            )
            rows.append({
                "fs": fs,
                "N": N,
                "f0": f0,
                "method": method,
                "mean_df": round(mean_df, 4) if np.isfinite(mean_df) else float("nan"),
                "std_df": round(std_df, 4) if np.isfinite(std_df) else float("nan"),
                "cv": round(cv, 4) if np.isfinite(cv) else float("nan"),
                "floor": round(floor, 4),
                "above_floor_frac": round(above_floor, 4) if np.isfinite(above_floor) else float("nan"),
                "nan_frac": round(nan_frac, 4),
            })

    fieldnames = ["fs", "N", "f0", "method", "mean_df", "std_df", "cv", "floor", "above_floor_frac", "nan_frac"]
    csv_path = out_dir / "level1_unit_accuracy.csv"
    _write_csv(csv_path, fieldnames, rows)
    print(f"  Saved {csv_path}")

    # Pivot: method vs regime (fs_N), showing mean_df
    pivot_rows = [
        {"regime": f"fs={int(r['fs'])}_N={r['N']}", "method": r["method"], "val": r["mean_df"]}
        for r in rows
    ]
    print("\n  Pivot: mean δf (Hz) by method × regime")
    for line in _pivot_table(pivot_rows, "regime", "method", "val", ".3f"):
        print("  " + line)

    return rows


# ---------------------------------------------------------------------------
# Level 2 — End-to-end reconstruction sweep
# ---------------------------------------------------------------------------

def _build_signal_configs(fs: float, N: int, seed: int) -> list[dict]:
    """Build signal configurations for Level 2."""
    snr_sweep_signals = ["two_sinusoids", "chirp_plus_sinusoid"]
    snr_levels = [5.0, 10.0, 20.0, 40.0]
    configs: list[dict] = []

    # Base signals (no explicit SNR override)
    configs.append({
        "name": "two_sinusoids",
        "snr_db": None,
        "signal": two_sinusoids(N=N, f1=50.0, f2=120.0, fs=fs, seed=seed),
    })
    configs.append({
        "name": "chirp_plus_sinusoid",
        "snr_db": None,
        "signal": chirp_plus_sinusoid(
            N=N, f_sin=50.0, f_start=10.0, f_end=150.0, fs=fs, seed=seed,
        ),
    })
    configs.append({
        "name": "rossler",
        "snr_db": None,
        "signal": rossler(N=N, seed=seed),
    })
    configs.append({
        "name": "component_onset",
        "snr_db": None,
        "signal": component_onset(
            N=N, f_steady=50.0, f_onset=120.0,
            onset_sample=N // 2, fs=fs, seed=seed,
        ),
    })

    # SNR sweep for two_sinusoids and chirp_plus_sinusoid
    for snr in snr_levels:
        rng = np.random.default_rng(seed + int(snr * 10))
        sig_ts = two_sinusoids(N=N, f1=50.0, f2=120.0, fs=fs, seed=seed, snr_db=snr)
        configs.append({"name": "two_sinusoids", "snr_db": snr, "signal": sig_ts})
        sig_cp = chirp_plus_sinusoid(
            N=N, f_sin=50.0, f_start=10.0, f_end=150.0, fs=fs, seed=seed, snr_db=snr,
        )
        configs.append({"name": "chirp_plus_sinusoid", "snr_db": snr, "signal": sig_cp})

    return configs


def run_level2(out_dir: Path) -> tuple[list[dict], dict[str, float]]:
    """Level 2: end-to-end reconstruction sweep; return rows and per-method median QRF."""
    print("\n" + "=" * 65)
    print("LEVEL 2 — End-to-end reconstruction sweep")
    print("=" * 65)

    fs = 500.0
    N = 3000
    seed = 42
    window_len = 300
    stride = 150
    configs = _build_signal_configs(fs, N, seed)
    rows: list[dict] = []

    for cfg in configs:
        sig_name = cfg["name"]
        snr_db = cfg["snr_db"]
        signal = cfg["signal"]
        snr_label = f"{snr_db:.0f}" if snr_db is not None else "clean"
        print(f"  {sig_name} (snr={snr_label})")

        for method in METHODS:
            result = _run_streaming_pipeline(signal, method, fs, window_len, stride)
            qrfs = result["qrf_vals"]
            nmses = result["nmse_vals"]
            median_qrf = float(np.median(qrfs)) if qrfs else float("nan")
            p10_qrf = float(np.percentile(qrfs, 10)) if qrfs else float("nan")
            mean_nmse = float(np.mean(nmses)) if nmses else float("nan")

            rows.append({
                "signal": sig_name,
                "snr_db": snr_db if snr_db is not None else "clean",
                "method": method,
                "median_qrf_db": round(median_qrf, 3) if np.isfinite(median_qrf) else float("nan"),
                "p10_qrf_db": round(p10_qrf, 3) if np.isfinite(p10_qrf) else float("nan"),
                "mean_nmse": round(mean_nmse, 6) if np.isfinite(mean_nmse) else float("nan"),
                "n_windows": result["n_windows"],
            })
            print(f"    {method:>8}: median QRF={median_qrf:.2f} dB, mean NMSE={mean_nmse:.4f}")

    fieldnames = ["signal", "snr_db", "method", "median_qrf_db", "p10_qrf_db", "mean_nmse", "n_windows"]
    csv_path = out_dir / "level2_system_quality.csv"
    _write_csv(csv_path, fieldnames, rows)
    print(f"  Saved {csv_path}")

    # Pivot: signal × method for clean signals only
    clean_rows = [r for r in rows if r["snr_db"] == "clean"]
    pivot_rows = [
        {"signal": r["signal"], "method": r["method"], "val": r["median_qrf_db"]}
        for r in clean_rows
    ]
    print("\n  Pivot: median QRF (dB) by signal × method (clean signals)")

    # Find best method per row for ★ marker
    best_per_signal: dict[str, str] = {}
    for r in clean_rows:
        sig = r["signal"]
        prev = best_per_signal.get(sig)
        if (prev is None or
                (np.isfinite(r["median_qrf_db"]) and
                 (not np.isfinite(
                     next(
                         (x["median_qrf_db"] for x in clean_rows
                          if x["signal"] == sig and x["method"] == prev),
                         float("nan"),
                     )
                 ) or r["median_qrf_db"] > next(
                     (x["median_qrf_db"] for x in clean_rows
                      if x["signal"] == sig and x["method"] == prev),
                     float("-inf"),
                 )))):
            best_per_signal[sig] = r["method"]

    for line in _pivot_table(pivot_rows, "signal", "method", "val", ".2f"):
        print("  " + line)

    # Mark best per signal
    print("\n  Best method per signal (clean):")
    for sig, meth in best_per_signal.items():
        best_qrf = next(
            (r["median_qrf_db"] for r in clean_rows if r["signal"] == sig and r["method"] == meth),
            float("nan"),
        )
        print(f"    ★ {sig}: {meth} ({best_qrf:.2f} dB)")

    # Compute per-method mean QRF across clean signals for verdict
    method_qrf: dict[str, list[float]] = {m: [] for m in METHODS}
    for r in clean_rows:
        if np.isfinite(r["median_qrf_db"]):
            method_qrf[r["method"]].append(r["median_qrf_db"])
    method_mean_qrf = {
        m: float(np.mean(vals)) if vals else float("nan")
        for m, vals in method_qrf.items()
    }
    return rows, method_mean_qrf


# ---------------------------------------------------------------------------
# Level 3 — Sparse-PSD stress sweep
# ---------------------------------------------------------------------------

def run_level3(out_dir: Path) -> list[dict]:
    """Level 3: sparse-PSD stress sweep over window sizes N=32..2048."""
    print("\n" + "=" * 65)
    print("LEVEL 3 — Sparse-PSD stress sweep")
    print("=" * 65)

    ns = [32, 64, 128, 256, 512, 1024, 2048]
    fs = 500.0
    f0 = 50.0
    snr_db = 20.0
    n_seeds = 20
    rows: list[dict] = []

    for N in ns:
        for method in METHODS:
            dfs: list[float] = []
            for seed in range(n_seeds):
                x = _make_sinusoid_noisy(f0, fs, N, snr_db, seed)
                freqs, psd = _compute_psd(x, fs)
                df = _call_estimator_safe(method, psd, freqs)
                dfs.append(df)

            finite_dfs = [v for v in dfs if np.isfinite(v)]
            nan_frac = 1.0 - len(finite_dfs) / len(dfs) if dfs else 1.0
            mean_df = float(np.mean(finite_dfs)) if finite_dfs else float("nan")
            floor = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0  # Welch freq_resolution
            floor_display = fs / N  # natural bin floor for display
            ratio = (mean_df / floor_display) if np.isfinite(mean_df) and floor_display > 0 else float("nan")
            above_floor = (
                sum(1 for v in finite_dfs if v >= floor_display) / len(finite_dfs)
                if finite_dfs else float("nan")
            )
            rows.append({
                "N": N,
                "method": method,
                "mean_df": round(mean_df, 4) if np.isfinite(mean_df) else float("nan"),
                "floor": round(floor_display, 4),
                "ratio": round(ratio, 3) if np.isfinite(ratio) else float("nan"),
                "above_floor_frac": round(above_floor, 4) if np.isfinite(above_floor) else float("nan"),
                "nan_frac": round(nan_frac, 4),
            })

    fieldnames = ["N", "method", "mean_df", "floor", "ratio", "above_floor_frac", "nan_frac"]
    csv_path = out_dir / "level3_sparse_psd.csv"
    _write_csv(csv_path, fieldnames, rows)
    print(f"  Saved {csv_path}")

    # Pivot: ratio by N × method
    pivot_rows = [
        {"N": str(r["N"]), "method": r["method"], "val": r["ratio"]}
        for r in rows
    ]
    print("\n  Pivot: mean_df / (fs/N) ratio by N × method")
    print("  (ratio ≈ 1 = graceful floor degradation; large ratio = tail inflation)")
    for line in _pivot_table(pivot_rows, "N", "method", "val", ".2f"):
        print("  " + line)

    return rows


# ---------------------------------------------------------------------------
# Level 4 — Latency micro-benchmark
# ---------------------------------------------------------------------------

def run_level4(out_dir: Path) -> tuple[list[dict], dict[str, float]]:
    """Level 4: latency micro-benchmark; return rows and mean µs/call per method."""
    print("\n" + "=" * 65)
    print("LEVEL 4 — Latency micro-benchmark")
    print("=" * 65)

    fs = 500.0
    N_psd = 512
    f0 = 50.0
    snr_db = 20.0
    seed_bm = 0
    n_calls = 2000
    n_warmup = 100

    # Pre-compute a fixed PSD for repeated calls
    x_bm = _make_sinusoid_noisy(f0, fs, N_psd, snr_db, seed_bm)
    freqs_bm, psd_bm = _compute_psd(x_bm, fs)

    rows: list[dict] = []
    method_mean_us: dict[str, float] = {}

    for method in METHODS:
        all_times: list[float] = []
        for i in range(n_calls):
            t0 = time.perf_counter()
            _call_estimator_safe(method, psd_bm, freqs_bm)
            t1 = time.perf_counter()
            if i >= n_warmup:
                all_times.append((t1 - t0) * 1e6)  # µs

        mean_us = float(np.mean(all_times))
        p95_us = float(np.percentile(all_times, 95))
        p99_us = float(np.percentile(all_times, 99))
        method_mean_us[method] = mean_us
        print(f"  {method:>8}: mean={mean_us:.2f} µs, p95={p95_us:.2f} µs, p99={p99_us:.2f} µs")

        # Bandwidth fraction: run 30-window pipeline and compare per-window times
        N_pipe = 300
        stride_pipe = 150
        signal_pipe = _make_sinusoid_noisy(f0, fs, 5000, snr_db, seed=1)
        engine = OptimizedSSD(fs=fs, spectral_method=method)
        wm = WindowManager(window_len=N_pipe, stride=stride_pipe, fs=fs)

        # Instrument bandwidth estimation at instance level
        bw_times_pipe: list[float] = []
        window_times: list[float] = []

        if method == "fwhm":
            _orig = OptimizedSSD._estimate_bandwidth_fwhm

            def _timed_bw(*a):  # type: ignore[no-untyped-def]
                """Timed wrapper for fwhm bandwidth estimation."""
                _t0 = time.perf_counter()
                r = _orig(*a)
                bw_times_pipe.append(time.perf_counter() - _t0)
                return r

            engine._estimate_bandwidth_fwhm = _timed_bw  # type: ignore[assignment]
        elif method == "moment":
            _orig = OptimizedSSD._estimate_bandwidth_moment

            def _timed_bw(*a):  # type: ignore[no-untyped-def]
                """Timed wrapper for moment bandwidth estimation."""
                _t0 = time.perf_counter()
                r = _orig(*a)
                bw_times_pipe.append(time.perf_counter() - _t0)
                return r

            engine._estimate_bandwidth_moment = _timed_bw  # type: ignore[assignment]
        else:  # gaussian
            _orig = OptimizedSSD._fit_gaussian_with_jacobian

            def _timed_bw(*a):  # type: ignore[no-untyped-def]
                """Timed wrapper for gaussian bandwidth estimation."""
                _t0 = time.perf_counter()
                r = _orig(*a)
                bw_times_pipe.append(time.perf_counter() - _t0)
                return r

            engine._fit_gaussian_with_jacobian = _timed_bw  # type: ignore[assignment]

        for sample in signal_pipe:
            window = wm.push(float(sample))
            if window is None:
                continue
            _tw0 = time.perf_counter()
            engine.fit(window)
            window_times.append(time.perf_counter() - _tw0)

        mean_bw_s = float(np.mean(bw_times_pipe)) if bw_times_pipe else float("nan")
        mean_win_s = float(np.mean(window_times)) if window_times else float("nan")
        bw_frac = (mean_bw_s / mean_win_s) if (
            np.isfinite(mean_bw_s) and np.isfinite(mean_win_s) and mean_win_s > 0
        ) else float("nan")

        rows.append({
            "method": method,
            "mean_us": round(mean_us, 3),
            "p95_us": round(p95_us, 3),
            "p99_us": round(p99_us, 3),
            "bw_frac_of_window": round(bw_frac, 4) if np.isfinite(bw_frac) else float("nan"),
        })

    fieldnames = ["method", "mean_us", "p95_us", "p99_us", "bw_frac_of_window"]
    csv_path = out_dir / "level4_latency.csv"
    _write_csv(csv_path, fieldnames, rows)
    print(f"  Saved {csv_path}")

    print("\n  Latency summary:")
    col_w = 14
    header = (
        f"  {'method':>10}"
        f"{'mean µs':>{col_w}}"
        f"{'p95 µs':>{col_w}}"
        f"{'p99 µs':>{col_w}}"
        f"{'bw_frac':>{col_w}}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        bwf = r["bw_frac_of_window"]
        bwf_str = f"{bwf:.3f}" if np.isfinite(bwf) else "nan"
        print(
            f"  {r['method']:>10}"
            f"{r['mean_us']:>{col_w}.2f}"
            f"{r['p95_us']:>{col_w}.2f}"
            f"{r['p99_us']:>{col_w}.2f}"
            f"{bwf_str:>{col_w}}"
        )

    return rows, method_mean_us


# ---------------------------------------------------------------------------
# Final verdict
# ---------------------------------------------------------------------------

def _print_verdict(
    method_mean_qrf: dict[str, float],
    method_mean_us: dict[str, float],
    level3_rows: list[dict],
) -> None:
    """Print the structured verdict section to stdout."""
    # Floor compliance: average above_floor_frac across all N for each method
    method_floor: dict[str, list[float]] = {m: [] for m in METHODS}
    for r in level3_rows:
        v = r["above_floor_frac"]
        if np.isfinite(v):
            method_floor[r["method"]].append(v)
    method_mean_floor = {
        m: float(np.mean(vals)) if vals else float("nan")
        for m, vals in method_floor.items()
    }

    # Determine recommended method: best combination of QRF and latency.
    # Score = QRF_normalised - latency_normalised (higher QRF better, lower latency better)
    max_qrf = max((v for v in method_mean_qrf.values() if np.isfinite(v)), default=0.0)
    min_us = min((v for v in method_mean_us.values() if np.isfinite(v)), default=1.0)
    max_us = max((v for v in method_mean_us.values() if np.isfinite(v)), default=1.0)
    scores: dict[str, float] = {}
    for m in METHODS:
        qrf_v = method_mean_qrf.get(m, float("nan"))
        us_v = method_mean_us.get(m, float("nan"))
        if not (np.isfinite(qrf_v) and np.isfinite(us_v)):
            scores[m] = float("-inf")
            continue
        qrf_norm = qrf_v / max_qrf if max_qrf > 0 else 0.0
        us_range = max_us - min_us
        us_norm = 1.0 - (us_v - min_us) / us_range if us_range > 0 else 1.0
        scores[m] = qrf_norm + us_norm

    best_method = max(scores, key=lambda m: scores[m])
    best_qrf = method_mean_qrf.get(best_method, float("nan"))
    best_us = method_mean_us.get(best_method, float("nan"))

    reason = (
        f"highest combined score: median QRF {best_qrf:.1f} dB "
        f"at mean latency {best_us:.1f} µs/call"
        if np.isfinite(best_qrf) and np.isfinite(best_us)
        else "insufficient data to determine"
    )

    print()
    print("  ══════════════════════════════════════════════════════════")
    print("  BANDWIDTH ESTIMATION METHOD EVALUATION — SUMMARY VERDICT")
    print("  ══════════════════════════════════════════════════════════")
    print()
    print("  Reconstruction quality (median QRF, averaged across signals):")
    for m in METHODS:
        v = method_mean_qrf.get(m, float("nan"))
        vs = f"{v:.1f}" if np.isfinite(v) else "nan"
        print(f"    {m:<9}: {vs} dB")
    print()
    print("  Latency (mean µs/call):")
    for m in METHODS:
        v = method_mean_us.get(m, float("nan"))
        vs = f"{v:.1f}" if np.isfinite(v) else "nan"
        print(f"    {m:<9}: {vs} µs")
    print()
    print("  Floor compliance (fraction of seeds above fs/N, avg across N):")
    for m in METHODS:
        v = method_mean_floor.get(m, float("nan"))
        vs = f"{v:.2f}" if np.isfinite(v) else "nan"
        print(f"    {m:<9}: {vs}")
    print()
    print(f"  Recommended default for streaming SSD: {best_method}")
    print(f"  Reason: {reason}")
    print("  ══════════════════════════════════════════════════════════")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

# Consistent colours and display names for the three methods.
_METHOD_COLOR = {"fwhm": "#2166ac", "moment": "#d6604d", "gaussian": "#4dac26"}
_METHOD_LABEL = {"fwhm": "FWHM", "moment": "Moment", "gaussian": "Gaussian+Jac"}


def _fig_save(fig: plt.Figure, path: Path) -> None:
    """Save figure and close it."""
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_level1_mean_df(rows: list[dict], plot_dir: Path) -> None:
    """L1 Plot A — mean δf by method across spectral-resolution regimes.

    Groups regimes by fs so you can see how δf scales with N for each fs.
    One subplot per fs value; bars grouped by N; colours = methods.
    """
    fss = sorted({r["fs"] for r in rows})
    fig, axes = plt.subplots(1, len(fss), figsize=(5 * len(fss), 4), sharey=False)
    if len(fss) == 1:
        axes = [axes]

    for ax, fs in zip(axes, fss):
        sub = [r for r in rows if r["fs"] == fs]
        ns = sorted({r["N"] for r in sub})
        x = np.arange(len(ns))
        width = 0.25
        offsets = np.linspace(-(len(METHODS) - 1) * width / 2,
                               (len(METHODS) - 1) * width / 2, len(METHODS))
        for i, m in enumerate(METHODS):
            vals = [next((r["mean_df"] for r in sub if r["N"] == n and r["method"] == m),
                         float("nan")) for n in ns]
            errs = [next((r["std_df"] for r in sub if r["N"] == n and r["method"] == m),
                         0.0) for n in ns]
            ax.bar(x + offsets[i], vals, width, yerr=errs, capsize=3,
                   label=_METHOD_LABEL[m], color=_METHOD_COLOR[m], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in ns])
        ax.set_xlabel("Window length N")
        ax.set_ylabel("Mean δf (Hz)")
        ax.set_title(f"fs = {int(fs)} Hz")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("L1 — Mean estimated δf by regime (±1 std, SNR = 20 dB, 20 seeds)",
                 fontsize=11)
    fig.tight_layout()
    _fig_save(fig, plot_dir / "L1_mean_df_by_regime.png")


def plot_level1_cv(rows: list[dict], plot_dir: Path) -> None:
    """L1 Plot B — coefficient of variation (stability) across all regimes.

    Lower CV = more consistent estimates across noise seeds.
    Each dot is one (fs, N) regime; x-axis = regime label; y-axis = CV.
    """
    regimes = list(dict.fromkeys(
        f"fs={int(r['fs'])}\nN={r['N']}" for r in rows
    ))
    fig, ax = plt.subplots(figsize=(max(8, len(regimes) * 0.7), 4))
    x = np.arange(len(regimes))
    width = 0.25
    offsets = np.linspace(-(len(METHODS) - 1) * width / 2,
                           (len(METHODS) - 1) * width / 2, len(METHODS))
    for i, m in enumerate(METHODS):
        vals = []
        for reg in regimes:
            fs_str, n_str = reg.replace("\n", " ").split()
            fs_v = float(fs_str.split("=")[1])
            n_v = int(n_str.split("=")[1])
            cv = next((r["cv"] for r in rows
                       if r["fs"] == fs_v and r["N"] == n_v and r["method"] == m),
                      float("nan"))
            vals.append(cv)
        ax.bar(x + offsets[i], vals, width, label=_METHOD_LABEL[m],
               color=_METHOD_COLOR[m], alpha=0.85)

    ax.axhline(0.30, color="grey", linestyle="--", linewidth=1,
               label="CV = 0.30 threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(regimes, fontsize=7)
    ax.set_ylabel("CV  (std / mean)")
    ax.set_title("L1 — Estimator stability: coefficient of variation across noise seeds")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _fig_save(fig, plot_dir / "L1_cv_stability.png")


def plot_level2_clean_qrf(rows: list[dict], plot_dir: Path) -> None:
    """L2 Plot C — median QRF on clean signals (grouped bars + p10 error bar).

    The error bar drops to the p10 QRF (10th-percentile worst window) so you
    can see both the typical quality and the worst-case window quality.
    """
    clean = [r for r in rows if r["snr_db"] == "clean"]
    signals = list(dict.fromkeys(r["signal"] for r in clean))
    x = np.arange(len(signals))
    width = 0.25
    offsets = np.linspace(-(len(METHODS) - 1) * width / 2,
                           (len(METHODS) - 1) * width / 2, len(METHODS))

    fig, ax = plt.subplots(figsize=(max(7, len(signals) * 2), 5))
    for i, m in enumerate(METHODS):
        medians = []
        yerr_lower = []  # drop from median to p10
        for sig in signals:
            row = next((r for r in clean if r["signal"] == sig and r["method"] == m), None)
            med = row["median_qrf_db"] if row else float("nan")
            p10 = row["p10_qrf_db"] if row else float("nan")
            medians.append(med)
            yerr_lower.append(max(0.0, med - p10) if np.isfinite(med) and np.isfinite(p10) else 0.0)

        bars = ax.bar(x + offsets[i], medians, width,
                      label=_METHOD_LABEL[m], color=_METHOD_COLOR[m], alpha=0.85)
        # Error bar: asymmetric — only downward to p10
        ax.errorbar(x + offsets[i], medians,
                    yerr=[yerr_lower, [0.0] * len(medians)],
                    fmt="none", color="black", capsize=4, linewidth=1)

    sig_labels = [s.replace("_", "\n") for s in signals]
    ax.set_xticks(x)
    ax.set_xticklabels(sig_labels, fontsize=9)
    ax.set_ylabel("Median QRF (dB)")
    ax.set_title("L2 — Reconstruction quality on clean signals\n"
                 "(bar = median, error bar drops to p10 worst window)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _fig_save(fig, plot_dir / "L2_qrf_clean_signals.png")


def plot_level2_snr_sweep(rows: list[dict], plot_dir: Path) -> None:
    """L2 Plot D — median QRF vs SNR for noisy signals.

    One subplot per swept signal (two_sinusoids, chirp_plus_sinusoid).
    Lines = methods; x-axis = SNR dB; includes the clean condition at SNR=∞.
    """
    swept = ["two_sinusoids", "chirp_plus_sinusoid"]
    snr_numeric = sorted({r["snr_db"] for r in rows if r["snr_db"] != "clean"})

    fig, axes = plt.subplots(1, len(swept), figsize=(6 * len(swept), 4), sharey=False)
    if len(swept) == 1:
        axes = [axes]

    for ax, sig in zip(axes, swept):
        for m in METHODS:
            xs, ys = [], []
            for snr in snr_numeric:
                row = next((r for r in rows
                            if r["signal"] == sig and r["snr_db"] == snr and r["method"] == m),
                           None)
                if row and np.isfinite(row["median_qrf_db"]):
                    xs.append(float(snr))
                    ys.append(row["median_qrf_db"])
            if xs:
                ax.plot(xs, ys, marker="o", label=_METHOD_LABEL[m],
                        color=_METHOD_COLOR[m], linewidth=2)

        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("Median QRF (dB)")
        ax.set_title(sig.replace("_", " "))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.invert_xaxis()  # left = noisiest (0 dB), right = cleanest (40 dB)
        ax.set_xticks(snr_numeric)

    fig.suptitle("L2 — QRF vs SNR (left = noisiest)", fontsize=11)
    fig.tight_layout()
    _fig_save(fig, plot_dir / "L2_snr_sweep.png")


def plot_level3_ratio(rows: list[dict], plot_dir: Path) -> None:
    """L3 Plot E — mean δf / (fs/N) ratio vs window length N.

    ratio ≈ 1.0 means the estimator returns exactly one natural bin width
    (graceful floor degradation).  Large ratio = over-estimation.
    ratio < 1.0 (moment at small N) = under-estimation below the natural floor.
    """
    ns = sorted({r["N"] for r in rows})
    fig, ax = plt.subplots(figsize=(7, 4))

    for m in METHODS:
        ratios = [next((r["ratio"] for r in rows if r["N"] == n and r["method"] == m),
                       float("nan")) for n in ns]
        ax.plot(ns, ratios, marker="o", label=_METHOD_LABEL[m],
                color=_METHOD_COLOR[m], linewidth=2)

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="ratio = 1 (floor)")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xticks(ns)
    ax.set_xlabel("Window length N  (log₂ scale)")
    ax.set_ylabel("mean δf / (fs/N)")
    ax.set_title("L3 — Sparse-PSD stress: δf floor compliance vs N\n"
                 "(dashed = ideal ratio 1; below = under-estimates floor)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _fig_save(fig, plot_dir / "L3_floor_ratio_vs_N.png")


def plot_level4_latency(rows: list[dict], plot_dir: Path) -> None:
    """L4 Plot F — latency and window-time fraction per method.

    Two panels: (left) mean/p95/p99 call time on a log scale — gaussian's
    curve_fit overhead is visible at a glance; (right) fraction of total
    window-processing time spent inside the bandwidth estimator.
    """
    methods = [r["method"] for r in rows]
    x = np.arange(len(methods))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left panel: latency (log scale)
    means = [r["mean_us"] for r in rows]
    p95s  = [r["p95_us"] for r in rows]
    p99s  = [r["p99_us"] for r in rows]
    colors = [_METHOD_COLOR[m] for m in methods]

    ax1.bar(x, means, color=colors, alpha=0.85, label="mean")
    ax1.errorbar(x, means,
                 yerr=[np.zeros(len(means)), [p - m for p, m in zip(p99s, means)]],
                 fmt="none", color="black", capsize=5, linewidth=1.5,
                 label="p99 whisker")
    ax1.scatter(x, p95s, marker="_", s=300, color="black", zorder=5, label="p95")
    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels([_METHOD_LABEL[m] for m in methods])
    ax1.set_ylabel("µs / call  (log scale)")
    ax1.set_title("Call latency\n(bar = mean, tick = p95, whisker = p99)")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3, which="both")

    # Annotate bar tops with the actual mean value
    for xi, v in zip(x, means):
        ax1.text(xi, v * 1.5, f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    # Right panel: bw_frac as percentage
    fracs = [r["bw_frac_of_window"] * 100 for r in rows]
    bars = ax2.bar(x, fracs, color=colors, alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels([_METHOD_LABEL[m] for m in methods])
    ax2.set_ylabel("% of window-processing time")
    ax2.set_title("Bandwidth estimation fraction\nof total per-window SSD time")
    ax2.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, fracs):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{v:.1f}%", ha="center", va="bottom", fontsize=9)

    fig.suptitle("L4 — Estimator latency", fontsize=11)
    fig.tight_layout()
    _fig_save(fig, plot_dir / "L4_latency.png")


def generate_plots(
    out_dir: Path,
    level1_rows: list[dict],
    level2_rows: list[dict],
    level3_rows: list[dict],
    level4_rows: list[dict],
) -> None:
    """Generate and save all six evaluation plots to out_dir/plots/."""
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 65)
    print("PLOTS")
    print("=" * 65)
    if level1_rows:
        plot_level1_mean_df(level1_rows, plot_dir)
        plot_level1_cv(level1_rows, plot_dir)
    if level2_rows:
        plot_level2_clean_qrf(level2_rows, plot_dir)
        plot_level2_snr_sweep(level2_rows, plot_dir)
    if level3_rows:
        plot_level3_ratio(level3_rows, plot_dir)
    if level4_rows:
        plot_level4_latency(level4_rows, plot_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """Run all four evaluation levels and print a structured verdict."""
    parser = argparse.ArgumentParser(
        description="Evaluate bandwidth estimation methods for OptimizedSSD."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "bandwidth_eval",
        help="Directory for CSV output files (default: results/bandwidth_eval/)",
    )
    args = parser.parse_args()
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    exit_code = 0

    try:
        level1_rows = run_level1(out_dir)
    except RuntimeError as exc:
        print(f"[ERROR] Level 1 CSV write failed: {exc}", file=sys.stderr)
        exit_code = 1
        level1_rows = []

    try:
        level2_rows, method_mean_qrf = run_level2(out_dir)
    except RuntimeError as exc:
        print(f"[ERROR] Level 2 CSV write failed: {exc}", file=sys.stderr)
        exit_code = 1
        level2_rows, method_mean_qrf = [], {}

    try:
        level3_rows = run_level3(out_dir)
    except RuntimeError as exc:
        print(f"[ERROR] Level 3 CSV write failed: {exc}", file=sys.stderr)
        exit_code = 1
        level3_rows = []

    try:
        level4_rows, method_mean_us = run_level4(out_dir)
    except RuntimeError as exc:
        print(f"[ERROR] Level 4 CSV write failed: {exc}", file=sys.stderr)
        exit_code = 1
        level4_rows, method_mean_us = [], {}

    # Hard-failure check: any method returning NaN on > 50% of seeds (per spec).
    # Floor compliance (above_floor_frac) is NOT a hard failure — it documents
    # the moment estimator's known narrow-window behaviour at small N.
    for r in level1_rows:
        nan_f = r.get("nan_frac", 0.0)
        if np.isfinite(nan_f) and nan_f > 0.5:
            print(
                f"[ERROR] Hard failure: method={r['method']} fs={r['fs']} N={r['N']} "
                f"returned NaN/inf on {nan_f*100:.0f}% of seeds (> 50%)",
                file=sys.stderr,
            )
            exit_code = 1

    generate_plots(out_dir, level1_rows, level2_rows, level3_rows, level4_rows)

    _print_verdict(method_mean_qrf, method_mean_us, level3_rows)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())