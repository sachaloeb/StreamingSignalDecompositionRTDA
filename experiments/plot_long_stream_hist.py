#!/usr/bin/env python
"""Plot per-window latency histograms for the long-stream stress test."""

import sys, os, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except Exception:
    pass

# ── paths ──────────────────────────────────────────────────────────────
ROOT = os.path.join(os.path.dirname(__file__), "..", "results", "long_stream")
baseline_csv = os.path.join(ROOT, "baseline", "long_stream_metrics.csv")
optimized_csv = os.path.join(ROOT, "optimized_fwhm", "long_stream_metrics.csv")
baseline_json = os.path.join(ROOT, "baseline", "run_summary.json")
optimized_json = os.path.join(ROOT, "optimized_fwhm", "run_summary.json")
out_dir = os.path.join(ROOT, "plots")
os.makedirs(out_dir, exist_ok=True)

# ── load data ──────────────────────────────────────────────────────────
df_base = pd.read_csv(baseline_csv)
df_opt = pd.read_csv(optimized_csv)

with open(baseline_json) as f:
    p95_base = json.load(f)["p95_time_ms"]
with open(optimized_json) as f:
    p95_opt = json.load(f)["p95_time_ms"]

BUDGET_MS = 150.0

# ── plot ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

configs = [
    (axes[0], df_base["time_ms"], p95_base, "Baseline SSD", "#666666"),
    (axes[1], df_opt["time_ms"], p95_opt, "OptimizedSSD-FWHM", "#1f77b4"),
]

for ax, times, p95, label, color in configs:
    ax.hist(times.values, bins=60, color=color, edgecolor="white", linewidth=0.4,
            alpha=0.85, label=label)
    ax.axvline(p95, color="blue", linestyle="--", linewidth=1.2,
               label=f"p95 = {p95:.1f} ms")
    ax.axvline(BUDGET_MS, color="red", linestyle="--", linewidth=1.2,
               label=f"RT budget = {BUDGET_MS:.0f} ms")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_ylabel("Count")
    ax.legend(fontsize=8, loc="upper right")

axes[0].set_title("Per-window latency distribution (long stream, N=60000)")
axes[1].set_xlabel("Per-window time (ms, symlog scale)")
fig.tight_layout()

png_path = os.path.join(out_dir, "latency_hist.png")
pdf_path = os.path.join(out_dir, "latency_hist.pdf")
fig.savefig(png_path, dpi=300)
fig.savefig(pdf_path)
plt.close(fig)

print(f"Saved {png_path}")
print(f"Saved {pdf_path}")
