"""Evaluate bandwidth estimation methods on the n_sinusoids signal.

Evaluates all four bandwidth methods (baseline, fwhm, moment, gaussian)
on a 5-sinusoid mixture across SNRs [clean, 5, 10, 20, 40] dB and writes
results to results/bandwidth_eval/level2_n_sinusoids.csv with the same
column schema as level2_system_quality.csv.

Usage
-----
    python experiments/eval_n_sinusoids.py [--out results/bandwidth_eval]
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

from experiments.synthetic.generators import n_sinusoids
from src.engines.ssd import SSD
from src.engines.ssd_optimized import OptimizedSSD
from src.metrics.stability import nmse, qrf
from src.streaming.window_manager import WindowManager

# 5 frequencies as specified in the thesis plan.
# Note: n_sinusoids uses np.sin() with zero initial phase (no random phase offset).
N_SINUSOID_FREQS = [20.0, 50.0, 80.0, 120.0, 200.0]

METHODS = ["baseline", "fwhm", "moment", "gaussian"]
SNR_LEVELS: list[float | None] = [None, 5.0, 10.0, 20.0, 40.0]  # None = clean

FS = 500.0
N_SAMPLES = 3000
WINDOW_LEN = 300
STRIDE = 150
SEED = 42

FIELDNAMES = [
    "signal", "snr_db", "method",
    "median_qrf_db", "p10_qrf_db", "mean_nmse",
    "mean_ncomp", "std_ncomp", "min_ncomp", "max_ncomp",
    "n_windows",
]


def _run_streaming(
    signal: np.ndarray,
    method: str,
    fs: float,
    window_len: int,
    stride: int,
) -> dict:
    """Run the streaming pipeline; return per-window QRF and NMSE lists."""
    if method == "baseline":
        engine: SSD = SSD(fs=fs)
    else:
        engine = OptimizedSSD(fs=fs, spectral_method=method)

    wm = WindowManager(window_len=window_len, stride=stride, fs=fs)
    qrf_vals: list[float] = []
    nmse_vals: list[float] = []
    n_components_vals: list[int] = []
    n_windows = 0

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # suppress moment guard warnings
        for sample in signal:
            window = wm.push(float(sample))
            if window is None:
                continue
            components = engine.fit(window)
            components_no_res = components[:-1]
            residual_comp = components[-1]

            n_components_vals.append(len(components_no_res))

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
        "n_components_vals": n_components_vals,
        "n_windows": n_windows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate bandwidth methods on n_sinusoids signal"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results/bandwidth_eval",
        help="Output directory (default: results/bandwidth_eval)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "level2_n_sinusoids.csv"

    rows: list[dict] = []

    print(
        f"Evaluating n_sinusoids ({N_SINUSOID_FREQS} Hz) "
        f"at fs={FS} Hz, N={N_SAMPLES}, window={WINDOW_LEN}, stride={STRIDE}"
    )
    print(f"Methods: {METHODS}")
    print(f"SNRs: {['clean' if s is None else f'{s}dB' for s in SNR_LEVELS]}")
    print()

    for snr_db in SNR_LEVELS:
        snr_label = "clean" if snr_db is None else f"{snr_db:.0f} dB"
        signal = n_sinusoids(
            N=N_SAMPLES,
            frequencies=N_SINUSOID_FREQS,
            fs=FS,
            snr_db=snr_db,
            seed=SEED,
        )
        print(f"  SNR = {snr_label}")

        for method in METHODS:
            result = _run_streaming(signal, method, FS, WINDOW_LEN, STRIDE)
            qrfs = result["qrf_vals"]
            nmses = result["nmse_vals"]
            ncomps = result["n_components_vals"]

            median_qrf = float(np.median(qrfs)) if qrfs else float("nan")
            p10_qrf = float(np.percentile(qrfs, 10)) if qrfs else float("nan")
            mean_nmse = float(np.mean(nmses)) if nmses else float("nan")
            mean_ncomp = float(np.mean(ncomps)) if ncomps else float("nan")
            std_ncomp = float(np.std(ncomps)) if ncomps else float("nan")
            min_ncomp = int(np.min(ncomps)) if ncomps else 0
            max_ncomp = int(np.max(ncomps)) if ncomps else 0

            rows.append({
                "signal": "n_sinusoids",
                "snr_db": snr_db if snr_db is not None else "clean",
                "method": method,
                "median_qrf_db": round(median_qrf, 3) if np.isfinite(median_qrf) else float("nan"),
                "p10_qrf_db": round(p10_qrf, 3) if np.isfinite(p10_qrf) else float("nan"),
                "mean_nmse": round(mean_nmse, 6) if np.isfinite(mean_nmse) else float("nan"),
                "mean_ncomp": round(mean_ncomp, 2) if np.isfinite(mean_ncomp) else float("nan"),
                "std_ncomp": round(std_ncomp, 2) if np.isfinite(std_ncomp) else float("nan"),
                "min_ncomp": min_ncomp,
                "max_ncomp": max_ncomp,
                "n_windows": result["n_windows"],
            })
            print(
                f"    {method:>8}: median QRF={median_qrf:.2f} dB, "
                f"mean NMSE={mean_nmse:.4f}, "
                f"components={mean_ncomp:.1f}±{std_ncomp:.1f} [{min_ncomp}–{max_ncomp}]"
            )

    # Save CSV
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {csv_path}  ({len(rows)} rows)")

    # Acceptance criterion summary
    clean_rows = [r for r in rows if r["snr_db"] == "clean"]
    snr20_rows = [r for r in rows if r["snr_db"] == 20.0]

    def _qrf(method: str, subset: list[dict]) -> float:
        for r in subset:
            if r["method"] == method:
                return float(r["median_qrf_db"])
        return float("nan")

    clean_fwhm = _qrf("fwhm", clean_rows)
    snr20_fwhm = _qrf("fwhm", snr20_rows)
    snr20_baseline = _qrf("baseline", snr20_rows)
    diff = snr20_fwhm - snr20_baseline

    print(
        f"\nn_sinusoids median QRF: "
        f"clean = {clean_fwhm:.2f} dB, "
        f"20 dB SNR = {snr20_fwhm:.2f} dB. "
        f"FWHM vs baseline diff at 20 dB: Δ = {diff:+.2f} dB."
    )


if __name__ == "__main__":
    main()