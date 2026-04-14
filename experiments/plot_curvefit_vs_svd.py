"""Profile curve_fit vs SVD cost across window sizes.

Instruments the SSD engine's ``_fit_gaussian_model`` and
``_decompose_trajectory`` methods with timing wrappers and sweeps
over a range of window lengths to quantify the relative cost of
each stage.

Usage
-----
    python experiments/plot_curvefit_vs_svd.py
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.synthetic.generators import chirp_plus_sinusoid
from src.engines.ssd import SSD
from src.streaming.component_matcher import ComponentMatcher
from src.streaming.trajectory_store import TrajectoryStore
from src.streaming.window_manager import WindowManager


def _run_instrumented(
    signal: np.ndarray,
    fs: float,
    window_len: int,
    stride: int,
) -> dict[str, float]:
    """Run the pipeline with timing wrappers on SSD internals.

    Returns per-window mean times (in seconds) for curve_fit, SVD,
    and everything else.
    """
    N = len(signal)
    max_components = 50

    wm = WindowManager(window_len=window_len, stride=stride, fs=fs)
    ssd = SSD(fs=fs)
    matcher = ComponentMatcher(
        distance="hybrid", freq_weight=0.5, fs=fs, lookback=10,
        max_cost=0.6, max_trajectories=max_components,
    )
    store = TrajectoryStore(max_components=max_components, max_len=N)

    # ---- monkey-patch timing wrappers ----
    gauss_times: list[float] = []
    svd_times: list[float] = []

    original_gauss = SSD._fit_gaussian_model  # staticmethod — already a plain function

    def timed_gauss(*args, **kwargs):  # noqa: ANN002
        t0 = time.perf_counter()
        result = original_gauss(*args, **kwargs)
        gauss_times.append(time.perf_counter() - t0)
        return result

    original_svd = ssd._decompose_trajectory

    def timed_svd(*args, **kwargs):  # noqa: ANN002
        t0 = time.perf_counter()
        result = original_svd(*args, **kwargs)
        svd_times.append(time.perf_counter() - t0)
        return result

    ssd._fit_gaussian_model = timed_gauss  # type: ignore[assignment]
    ssd._decompose_trajectory = timed_svd  # type: ignore[assignment]

    # ---- run pipeline, collecting per-window times ----
    per_window_gauss: list[float] = []
    per_window_svd: list[float] = []
    per_window_other: list[float] = []
    n_windows = 0

    for sample_idx in range(N):
        window = wm.push(float(signal[sample_idx]))
        if window is None:
            continue

        gauss_times.clear()
        svd_times.clear()

        t_start = time.perf_counter()
        components = ssd.fit(window)
        t_end = time.perf_counter()

        total_window = t_end - t_start
        g_total = sum(gauss_times)
        s_total = sum(svd_times)

        per_window_gauss.append(g_total)
        per_window_svd.append(s_total)
        per_window_other.append(max(0.0, total_window - g_total - s_total))

        components_no_res = components[:-1]
        matching = dict(
            matcher.match_stateful(components_no_res, wm.overlap)
        )
        window_start = sample_idx - wm.window_len + 1
        store.update(window_start, components_no_res, matching, wm.overlap)
        n_windows += 1

    if n_windows == 0:
        return {
            "n_windows": 0,
            "mean_curvefit_s": 0.0,
            "mean_svd_s": 0.0,
            "mean_other_s": 0.0,
        }

    return {
        "n_windows": n_windows,
        "mean_curvefit_s": float(np.mean(per_window_gauss)),
        "mean_svd_s": float(np.mean(per_window_svd)),
        "mean_other_s": float(np.mean(per_window_other)),
    }


def main() -> None:
    """Sweep window sizes, save CSV and plots."""
    out_dir = ROOT / "results" / "curvefit_vs_svd"
    out_dir.mkdir(parents=True, exist_ok=True)

    N = 30000
    fs = 1000.0
    signal = chirp_plus_sinusoid(
        N=N, f_sin=50.0, f_start=10.0, f_end=150.0,
        fs=fs, snr_db=20.0,
    )

    window_sizes = [100, 200, 400, 800, 1200, 1600, 2000, 2500, 3000, 4000, 5000, 6000, 8000, 10000]
    rows: list[dict[str, object]] = []

    for wl in window_sizes:
        stride = wl // 2
        print(f"Window length = {wl}, stride = {stride} ... ", end="", flush=True)
        result = _run_instrumented(signal, fs, wl, stride)
        total = result["mean_curvefit_s"] + result["mean_svd_s"] + result["mean_other_s"]
        cf_pct = 100.0 * result["mean_curvefit_s"] / total if total > 0 else 0.0
        svd_pct = 100.0 * result["mean_svd_s"] / total if total > 0 else 0.0

        row = {
            "window_len": wl,
            "mean_curvefit_ms": result["mean_curvefit_s"] * 1000.0,
            "mean_svd_ms": result["mean_svd_s"] * 1000.0,
            "mean_other_ms": result["mean_other_s"] * 1000.0,
            "curvefit_pct": cf_pct,
            "svd_pct": svd_pct,
            "n_windows": result["n_windows"],
        }
        rows.append(row)
        print(
            f"done  (curvefit={row['mean_curvefit_ms']:.2f}ms, "
            f"svd={row['mean_svd_ms']:.2f}ms, "
            f"curvefit%={cf_pct:.1f}%)"
        )

    # ---- save CSV ----
    csv_path = out_dir / "timing_data.csv"
    fieldnames = [
        "window_len", "mean_curvefit_ms", "mean_svd_ms",
        "mean_other_ms", "curvefit_pct", "svd_pct", "n_windows",
    ]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved to {csv_path}")

    # ---- print summary table ----
    print(f"\n{'Window':>8}  {'CurveFit(ms)':>13}  {'SVD(ms)':>9}  "
          f"{'Other(ms)':>10}  {'CF%':>6}  {'SVD%':>6}  {'#Win':>5}")
    print("-" * 72)
    for r in rows:
        print(
            f"{r['window_len']:>8}  {r['mean_curvefit_ms']:>13.2f}  "
            f"{r['mean_svd_ms']:>9.2f}  {r['mean_other_ms']:>10.2f}  "
            f"{r['curvefit_pct']:>6.1f}  {r['svd_pct']:>6.1f}  "
            f"{r['n_windows']:>5}"
        )

    # ---- Plot A: line plot (log-log) ----
    wl_arr = np.array([r["window_len"] for r in rows])
    cf_arr = np.array([r["mean_curvefit_ms"] for r in rows])
    svd_arr = np.array([r["mean_svd_ms"] for r in rows])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(wl_arr, cf_arr, "o-", color="#E69F00", linewidth=2,
            markersize=7, label="Gaussian curve_fit")
    ax.plot(wl_arr, svd_arr, "s-", color="#0072B2", linewidth=2,
            markersize=7, label="SVD")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Window length (samples)", fontsize=12)
    ax.set_ylabel("Mean time per window (ms)", fontsize=12)
    ax.set_title("Per-Window Time: curve_fit vs SVD", fontsize=14)
    ax.legend(fontsize=11)

    # Annotation for dominance zone
    y_mid = np.sqrt(cf_arr.max() * svd_arr.min())
    ax.annotate(
        "curve_fit dominates",
        xy=(wl_arr[len(wl_arr) // 2], y_mid),
        fontsize=10, fontstyle="italic", color="#E69F00",
        ha="center",
    )

    ax.tick_params(which="both", direction="in")
    fig.tight_layout()
    fig.savefig(out_dir / "curvefit_vs_svd_time.png", dpi=150)
    plt.close(fig)
    print(f"Plot A saved to {out_dir / 'curvefit_vs_svd_time.png'}")

    # ---- Plot B: stacked bar chart ----
    cf_pct_arr = np.array([r["curvefit_pct"] for r in rows])
    svd_pct_arr = np.array([r["svd_pct"] for r in rows])
    other_pct_arr = 100.0 - cf_pct_arr - svd_pct_arr

    x_labels = [str(int(w)) for w in wl_arr]
    x_pos = np.arange(len(x_labels))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x_pos, cf_pct_arr, color="#E69F00", label="curve_fit")
    ax.bar(x_pos, svd_pct_arr, bottom=cf_pct_arr, color="#0072B2",
           label="SVD")
    ax.bar(x_pos, other_pct_arr, bottom=cf_pct_arr + svd_pct_arr,
           color="#999999", label="Other")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Window length (samples)", fontsize=12)
    ax.set_ylabel("Percentage of total window time", fontsize=12)
    ax.set_title("Cost Breakdown by Window Length", fontsize=14)
    ax.legend(fontsize=11, loc="upper right")
    ax.set_ylim(0, 105)

    fig.tight_layout()
    fig.savefig(out_dir / "curvefit_vs_svd_pct.png", dpi=150)
    plt.close(fig)
    print(f"Plot B saved to {out_dir / 'curvefit_vs_svd_pct.png'}")


if __name__ == "__main__":
    main()