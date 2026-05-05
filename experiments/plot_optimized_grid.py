"""Plot the OptimizedSSD benchmark grid results.

Reads complexity_grid.csv and produces:
1. time_vs_window_len_optimized.png/.pdf  — log-log mean time ± 1 std
2. memory_vs_window_len_optimized.png/.pdf — peak memory vs window len
3. speedup_vs_window_len.png/.pdf         — baseline/optimized ratio

Usage
-----
    python experiments/plot_optimized_grid.py \\
        --csv results/benchmarks_optimized/complexity_grid.csv \\
        --out results/benchmarks_optimized/plots
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Apply consistent style
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    pass  # fall back to default

# ---------------------------------------------------------------------------
# Colour map per engine
# ---------------------------------------------------------------------------
ENGINE_COLORS = {
    "ssd":                                  "#666666",
    "ssd_optimized_fwhm":                   "#1f77b4",
    "ssd_optimized_moment":                 "#d62728",
    "ssd_optimized_moment_substituted_fwhm": "#d62728",  # same as moment
    "ssd_optimized_gaussian":               "#2ca02c",
    "ssd_incremental":                      "#ff7f0e",
    "ssd_rank1":                            "#9467bd",
}
ENGINE_LABELS = {
    "ssd":                                  "Baseline SSD",
    "ssd_optimized_fwhm":                   "OptimizedSSD-fwhm",
    "ssd_optimized_moment":                 "OptimizedSSD-moment",
    "ssd_optimized_moment_substituted_fwhm": "OptimizedSSD-moment (→FWHM)",
    "ssd_optimized_gaussian":               "OptimizedSSD-gaussian+jac",
    "ssd_incremental":                      "IncrementalSSD",
    "ssd_rank1":                            "RankOneIncrementalSSD",
}


def _read_csv(csv_path: Path) -> list[dict]:
    rows = []
    with open(csv_path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append({
                "engine": row["engine"],
                "window_len": int(row["window_len"]),
                "seed": int(row["seed"]),
                "mean_time_ms": float(row["mean_time_ms"]),
                "std_time_ms": float(row["std_time_ms"]),
                "p95_time_ms": float(row["p95_time_ms"]),
                "peak_memory_mib": float(row["peak_memory_mib"]),
            })
    return rows


def _aggregate(rows: list[dict]) -> dict:
    """Aggregate over seeds: mean/std of mean_time_ms and peak_memory_mib."""
    by_eng_wl: dict[tuple, list] = {}
    for row in rows:
        key = (row["engine"], row["window_len"])
        by_eng_wl.setdefault(key, []).append(row)

    engines = sorted(set(r["engine"] for r in rows))
    window_lens = sorted(set(r["window_len"] for r in rows))

    result: dict = {"engines": engines, "window_lens": window_lens}
    for eng in engines:
        result[eng] = {
            "wls": [], "mean_t": [], "std_t": [], "mean_mem": [], "std_mem": [],
        }
        for wl in window_lens:
            subset = by_eng_wl.get((eng, wl), [])
            if not subset:
                continue
            mean_ts = [r["mean_time_ms"] for r in subset]
            mems = [r["peak_memory_mib"] for r in subset]
            result[eng]["wls"].append(wl)
            result[eng]["mean_t"].append(float(np.mean(mean_ts)))
            result[eng]["std_t"].append(float(np.std(mean_ts)))
            result[eng]["mean_mem"].append(float(np.mean(mems)))
            result[eng]["std_mem"].append(float(np.std(mems)))
    return result


def _polyfit_alpha(wls: list[int], times: list[float]) -> float | None:
    valid = [(w, t) for w, t in zip(wls, times) if t > 0]
    if len(valid) < 2:
        return None
    log_w = np.log10([v[0] for v in valid])
    log_t = np.log10([v[1] for v in valid])
    return float(np.polyfit(log_w, log_t, 1)[0])


def _save_csv(data: dict, engines: list[str], out_csv: Path) -> None:
    """Save aggregated stats to a sibling CSV before plotting."""
    with open(out_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["engine", "window_len", "mean_time_ms", "std_time_ms",
                         "mean_mem_mib", "std_mem_mib"])
        for eng in engines:
            d = data[eng]
            for wl, mt, st, mm, sm in zip(
                d["wls"], d["mean_t"], d["std_t"], d["mean_mem"], d["std_mem"]
            ):
                writer.writerow([eng, wl, mt, st, mm, sm])


def plot_time_vs_window_len(data: dict, out_dir: Path) -> None:
    """Log-log mean time per window vs window length, ±1 std across seeds."""
    # Save raw data first
    _save_csv(
        data, data["engines"],
        out_dir / "time_vs_window_len_optimized.csv",
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    for eng in data["engines"]:
        d = data[eng]
        if not d["wls"]:
            continue
        wls = np.array(d["wls"], dtype=float)
        mt = np.array(d["mean_t"])
        st = np.array(d["std_t"])
        color = ENGINE_COLORS.get(eng, "#333333")
        label = ENGINE_LABELS.get(eng, eng)

        ax.loglog(wls, mt, "o-", color=color, label=label)
        ax.fill_between(wls, np.maximum(mt - st, 1e-3), mt + st,
                        color=color, alpha=0.15)

        alpha = _polyfit_alpha(d["wls"], d["mean_t"])
        if alpha is not None:
            ax.annotate(
                f"α={alpha:.2f}",
                xy=(wls[-1], mt[-1]),
                xytext=(4, 0), textcoords="offset points",
                fontsize=8, color=color, va="center",
            )

    ax.set_xlabel("Window length (samples)", fontsize=12)
    ax.set_ylabel("Mean time per window (ms)", fontsize=12)
    ax.set_title("Time per Window vs Window Length (log-log)", fontsize=13)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    fig.tight_layout()

    for ext in (".png", ".pdf"):
        path = out_dir / f"time_vs_window_len_optimized{ext}"
        dpi = 300 if ext == ".png" else None
        fig.savefig(path, dpi=dpi)
        print(f"Saved: {path}")
    plt.close(fig)


def plot_memory_vs_window_len(data: dict, out_dir: Path) -> None:
    """Peak memory vs window length."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for eng in data["engines"]:
        d = data[eng]
        if not d["wls"]:
            continue
        wls = np.array(d["wls"], dtype=float)
        mm = np.array(d["mean_mem"])
        sm = np.array(d["std_mem"])
        color = ENGINE_COLORS.get(eng, "#333333")
        label = ENGINE_LABELS.get(eng, eng)

        ax.loglog(wls, mm, "s-", color=color, label=label)
        ax.fill_between(wls, np.maximum(mm - sm, 1e-3), mm + sm,
                        color=color, alpha=0.15)

    ax.set_xlabel("Window length (samples)", fontsize=12)
    ax.set_ylabel("Peak memory (MiB)", fontsize=12)
    ax.set_title("Peak Memory vs Window Length", fontsize=13)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    fig.tight_layout()

    for ext in (".png", ".pdf"):
        path = out_dir / f"memory_vs_window_len_optimized{ext}"
        dpi = 300 if ext == ".png" else None
        fig.savefig(path, dpi=dpi)
        print(f"Saved: {path}")
    plt.close(fig)


def plot_speedup_vs_window_len(data: dict, out_dir: Path) -> None:
    """speedup = baseline_mean / engine_mean, log-y."""
    baseline_d = data.get("ssd")
    if baseline_d is None or not baseline_d["wls"]:
        print("No baseline 'ssd' data; skipping speedup plot.")
        return

    baseline_by_wl = dict(zip(baseline_d["wls"], baseline_d["mean_t"]))

    # Save speedup CSV
    speedup_csv = out_dir / "speedup_vs_window_len.csv"
    with open(speedup_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["engine", "window_len", "speedup"])
        for eng in data["engines"]:
            if eng == "ssd":
                continue
            for wl, mt in zip(data[eng]["wls"], data[eng]["mean_t"]):
                baseline_t = baseline_by_wl.get(wl)
                if baseline_t and mt > 0:
                    writer.writerow([eng, wl, baseline_t / mt])

    fig, ax = plt.subplots(figsize=(10, 6))

    for eng in data["engines"]:
        if eng == "ssd":
            continue
        d = data[eng]
        speedups = []
        wls_plot = []
        for wl, mt in zip(d["wls"], d["mean_t"]):
            baseline_t = baseline_by_wl.get(wl)
            if baseline_t and mt > 0:
                speedups.append(baseline_t / mt)
                wls_plot.append(wl)
        if not wls_plot:
            continue
        color = ENGINE_COLORS.get(eng, "#333333")
        label = ENGINE_LABELS.get(eng, eng)
        ax.semilogy(wls_plot, speedups, "o-", color=color, label=label)

    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1, alpha=0.7,
               label="Baseline (1×)")
    ax.set_xlabel("Window length (samples)", fontsize=12)
    ax.set_ylabel("Speedup over baseline SSD (×)", fontsize=12)
    ax.set_title("Speedup vs Window Length (log-y)", fontsize=13)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", ls="--", alpha=0.5)
    fig.tight_layout()

    for ext in (".png", ".pdf"):
        path = out_dir / f"speedup_vs_window_len{ext}"
        dpi = 300 if ext == ".png" else None
        fig.savefig(path, dpi=dpi)
        print(f"Saved: {path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot OptimizedSSD benchmark grid")
    parser.add_argument(
        "--csv",
        default="results/benchmarks_optimized/complexity_grid.csv",
    )
    parser.add_argument(
        "--out",
        default="results/benchmarks_optimized/plots",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_csv(csv_path)
    print(f"Loaded {len(rows)} rows from {csv_path}")

    data = _aggregate(rows)

    plot_time_vs_window_len(data, out_dir)
    plot_memory_vs_window_len(data, out_dir)
    plot_speedup_vs_window_len(data, out_dir)

    # Print one-line summary
    engines_present = data["engines"]
    baseline_d = data.get("ssd", {})
    baseline_by_wl = dict(zip(baseline_d.get("wls", []), baseline_d.get("mean_t", [])))

    def _speedup_at(eng: str, wl: int) -> str:
        mt = dict(zip(data.get(eng, {}).get("wls", []), data.get(eng, {}).get("mean_t", [])))
        b = baseline_by_wl.get(wl)
        o = mt.get(wl)
        if b and o and o > 0:
            return f"{b/o:.1f}×"
        return "N/A"

    def _alpha(eng: str) -> str:
        d = data.get(eng, {})
        a = _polyfit_alpha(d.get("wls", []), d.get("mean_t", []))
        return f"{a:.2f}" if a is not None else "N/A"

    print(
        f"\nOptimizedSSD-fwhm vs baseline speedup: "
        f"at window_len=400 = {_speedup_at('ssd_optimized_fwhm', 400)}; "
        f"at window_len=1600 = {_speedup_at('ssd_optimized_fwhm', 1600)}; "
        f"scaling exponent α_baseline = {_alpha('ssd')}, "
        f"α_fwhm = {_alpha('ssd_optimized_fwhm')}"
    )


if __name__ == "__main__":
    main()
