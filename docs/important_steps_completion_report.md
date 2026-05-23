# Important-but-Non-Blocking Tasks (I1–I7): Completion Report

## Executive Summary

All seven important-but-non-blocking tasks (I1–I7) have been completed. Four documentation files, one latency histogram, one improved dashboard render, and two sensitivity sweep experiments with plots were produced. No existing files were modified. Total experiment wall time: I2a = 50.8 s, I2b = 33.9 s. All outputs are additive and reside in `docs/`, `experiments/`, and `results/`.

## Per-Phase Status Table

| Phase | Task | Status | Notes |
|-------|------|--------|-------|
| I1 | Latency histogram | ✓ | Two stacked histograms (baseline + optimized), symlog x-axis, p95 and RT budget lines |
| I2a | NMSE threshold sweep | ✓ | 100 cells (5 thresholds × 2 engines × 2 signals × 5 seeds), 50.8 s wall time |
| I2b | max_components sweep | ✓ | 50 cells (5 values × 2 engines × 1 signal × 5 seeds), 33.9 s wall time |
| I3 | Latency vs throughput doc | ✓ | Distinguishes throughput RT from end-to-end algorithmic latency |
| I4 | Memory-bound proof doc | ✓ | O(window_len + max_components × history) argument with empirical data |
| I5 | BMSC 2025 comparison doc | ✓ | Crossover at W ≈ 5519 computed by interpolation; BMSC claims marked [verify against PDF] |
| I6 | Dashboard re-render | ✓ | constrained_layout, max 8 components, original untouched |
| I7 | Compute environment doc | ✓ | Consolidated env from all run_summary.json files, BLAS backend captured |

## Files Created

### Phase I1
- `experiments/plot_long_stream_hist.py`
- `results/long_stream/plots/latency_hist.png`
- `results/long_stream/plots/latency_hist.pdf`

### Phase I2
- `experiments/sensitivity_nmse_threshold.py`
- `experiments/sensitivity_max_components.py`
- `experiments/plot_nmse_threshold_grid.py`
- `experiments/plot_max_components_grid.py`
- `results/sensitivity/nmse_threshold/nmse_threshold_grid.csv` (100 rows)
- `results/sensitivity/nmse_threshold/run_summary.json`
- `results/sensitivity/nmse_threshold/plots/nmse_threshold_qrf_vs_time.png`
- `results/sensitivity/nmse_threshold/plots/nmse_threshold_qrf_vs_time.pdf`
- `results/sensitivity/max_components/max_components_grid.csv` (50 rows)
- `results/sensitivity/max_components/run_summary.json`
- `results/sensitivity/max_components/plots/max_components_qrf_vs_time.png`
- `results/sensitivity/max_components/plots/max_components_qrf_vs_time.pdf`

### Phase I3
- `docs/latency_vs_throughput.md`

### Phase I4
- `docs/memory_bound.md`

### Phase I5
- `docs/related_work_bmsc2025.md`

### Phase I6
- `experiments/replot_pipeline_dashboard.py`
- `results/demo_run/07_pipeline_dashboard_v2.png`
- `results/demo_run/07_pipeline_dashboard_v2.pdf`

### Phase I7
- `docs/compute_environment.md`

## Files Modified

**None.** All work was additive.

## Anomalies

1. **BMSC 2025 PDF not found in repo.** `docs/related_work_bmsc2025.md` was written based on the brief's abstract-level claims. All statements attributed to the BMSC paper are marked with `[verify against PDF]`.
2. **Dashboard v2 uses reconstructed signal.** The original `plot_pipeline_dashboard` function requires a `TrajectoryStore` object and the original signal array. Since only `trajectories.npz` was saved, the replot script reconstructs the signal by summing all trajectories (NaN → 0). This is an approximation — the residual component is lost.
3. **Sensitivity sweep max_components uses `max_iter` as proxy.** The SSD `max_iter` parameter caps the number of extraction iterations; setting it to `max_components` effectively limits the component count, but the semantics are "maximum iterations" not "maximum components." This is functionally equivalent for SSD.

## Outstanding Human Actions

1. **Verify BMSC 2025 claims** — locate `bmsc2025_422.pdf` and remove `[verify against PDF]` markers from `docs/related_work_bmsc2025.md` after verification.
2. **Review sensitivity plots** — the nmse_threshold sweep shows large QRF variance at low thresholds for n_sinusoids; confirm this is expected given the signal structure.
3. **Consider dashboard signal source** — if the original raw signal used for `07_pipeline_dashboard.png` is available, re-run `replot_pipeline_dashboard.py` with the real signal instead of the trajectory sum.

## Suggested Follow-ups

1. Add per-engine histogram overlay (both engines on one subplot) as a companion to the stacked view.
2. Extend the NMSE threshold sweep to include `ssd_optimized_moment` and `ssd_optimized_gaussian` engines.
3. Add a crossover-point visualization to `docs/related_work_bmsc2025.md` (plot curvefit% vs svd% with crossover annotation).
4. Run the sensitivity sweeps on a longer signal (N=30,000) to check if the trends hold at more windows.
5. Add a `max_iter` column to the sensitivity CSV to distinguish it from `max_components` semantically.
