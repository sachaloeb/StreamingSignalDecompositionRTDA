# Canonical Numbers — Streaming SSD Thesis

*Generated from experiment results files. Every number is cited with its source.*
*Last updated: 2026-05-03*

---

## 1. Pipeline-Level Latency

**Config**: window_len=300, stride=150, fs=10 000 Hz, signal=chirp_plus_sinusoid, N=60 000, SNR=5 dB
**Source**: `results/profiling/optimized_report.txt`

> **Important**: This profiling run used **fs=10 000 Hz**, giving a real-time budget of
> stride/fs = 150/10000 = **15 ms** — not the 150 ms budget at fs=1000 Hz.
> All per-window times and budget comparisons in this section apply to that high-frequency config.

| Engine | mean ± std | p95 | Peak memory | Budget status |
|---|---|---|---|---|
| Baseline SSD | 48.59 ± 29.62 ms | 73.91 ms | 13.16 MiB | OVER BUDGET (>15 ms) |
| OptimizedSSD-fwhm | 5.11 ± 1.86 ms | 8.49 ms | 2.34 MiB | OK |
| OptimizedSSD-moment | 5.89 ± 2.22 ms | 9.91 ms | 2.41 MiB | OK |
| OptimizedSSD-gaussian+jac | 29.56 ± 10.81 ms | 46.33 ms | 12.00 MiB | OVER BUDGET |

**Derived ratios** (baseline ÷ optimized, same config):
- Speedup (mean): baseline 48.59 ms / fwhm 5.11 ms = **9.51×**
- Memory ratio: 13.16 MiB / 2.34 MiB = **5.62×**

**Decomposition breakdown** — Baseline SSD (source: `optimized_report.txt`, "Engine: Baseline SSD"):
- Bandwidth estimation: mean=37.73 ms, p95=57.15 ms — **77.6% of decomp**
- SVD: mean=2.84 ms, p95=5.22 ms — 5.8%
- Eigentriple selection: mean=6.26 ms, p95=11.11 ms — 12.9%

**Decomposition breakdown** — OptimizedSSD-fwhm (source: `optimized_report.txt`, "Engine: OptimizedSSD (fwhm)"):
- Bandwidth estimation: mean=0.06 ms, p95=0.08 ms — **1.2% of decomp**
- SVD: mean=2.69 ms, p95=5.04 ms — 52.5%
- Eigentriple selection: mean=0.78 ms, p95=1.32 ms — 15.3%

---

## 2. Bandwidth Estimator Microbenchmark

**Config**: 2000 calls on fixed PSD, N_psd=512, fs=500 Hz, f0=50 Hz, SNR=20 dB, 100-call warmup discarded
**Source**: `results/bandwidth_eval/level4_latency.csv`

| Method | mean (µs) | p95 (µs) | p99 (µs) | bw_frac_of_window |
|---|---|---|---|---|
| baseline (Gaussian 3-model) | 6 100.511 | 6 618.966 | 7 256.238 | 80.26% |
| fwhm | 6.475 | 6.625 | 8.168 | **1.60%** |
| moment | 9.197 | 10.335 | 13.126 | 2.43% |
| gaussian (with Jacobian) | 4 258.681 | 4 748.096 | 5 059.957 | 82.20% |

Speedup (baseline / fwhm): 6100.5 / 6.5 = **940×** per call.

---

## 3. Cross-Window Scaling — OptimizedSSD Benchmark Grid

**Config**: fs=1000 Hz, 5 seeds, chirp_plus_sinusoid, stride=window_len/2
**Source**: `results/benchmarks_optimized/complexity_grid.csv`

### Mean per-window time (ms), averaged over 5 seeds

| window_len | Baseline SSD | OptimizedSSD-fwhm | Speedup (fwhm/baseline) | OptimizedSSD-moment |
|---|---|---|---|---|
| 100 | 49.03 | 7.42 | **6.61×** | N/A (guard→fwhm) |
| 200 | 47.59 | 9.40 | **5.06×** | N/A (guard→fwhm) |
| 400 | 39.36 | 11.96 | **3.29×** | 11.90 |
| 800 | 49.58 | 22.93 | **2.16×** | 23.16 |
| 1 600 | 74.92 | 41.01 | **1.83×** | 46.36 |
| 3 200 | 144.21 | 89.50 | **1.61×** | 110.18 |
| 6 400 | 201.83 | 151.27 | **1.33×** | 220.53 |

### Scaling exponents (α, log-log fit over window_len 100–6400)

| Engine | α |
|---|---|
| Baseline SSD | 0.366 |
| OptimizedSSD-fwhm | 0.762 |
| OptimizedSSD-moment | 1.067 |
| IncrementalSSD | 0.362 |
| RankOneIncrementalSSD | 0.893 |

Note: OptimizedSSD-fwhm has a **steeper** slope (α=0.76) than baseline (α=0.37) because the
SVD now dominates; the speedup diminishes at larger window sizes as SVD share grows.

---

## 4. Reconstruction Quality Across SNRs

**Config**: fs=1000 Hz, N=3000, window_len=300, stride=150, 5 seeds per cell
**Source**: `results/snr_sweep_multiseed/snr_sweep_stats.csv`
**Test**: Wilcoxon paired (per-seed QRF difference, zero_method="wilcox"), BH FDR α=0.05

### Sample cells (signal=two_sinusoids, SNR=20 dB)

| Engine | median QRF (dB) | 95% CI | Wilcoxon p vs baseline | BH q-val | Reject null? |
|---|---|---|---|---|---|
| Baseline SSD | 20.99 | [20.98, 21.20] | — | — | — |
| OptimizedSSD-fwhm | 20.99 | [20.98, 21.20] | 1.000 | n/a | No |
| OptimizedSSD-moment | 20.99 | [20.98, 21.20] | 1.000 | n/a | No |
| OptimizedSSD-gaussian | 20.99 | [20.98, 21.20] | 1.000 | n/a | No |

**Global rejection summary** (all signals × all SNRs × all optimized engines, BH-corrected):
- OptimizedSSD-fwhm: 0 / 30 cells rejected → equivalence holds
- OptimizedSSD-moment: 0 / 30 cells rejected → equivalence holds
- OptimizedSSD-gaussian: 0 / 30 cells rejected → equivalence holds

**Conclusion**: OptimizedSSD is statistically indistinguishable from the baseline in QRF
across all tested signal types and SNR levels. (source: `snr_sweep_stats.csv`)

---

## 5. Component Matching

**Source**: `results/matcher_comparison.csv`

| Strategy | freq_weight | mean_confidence | mean_QRF (dB) |
|---|---|---|---|
| d_corr | 0.0 | 0.9864 | 18.68 |
| d_freq | 1.0 | 0.9957 | 18.68 |
| hybrid (w=0.3) | 0.3 | 0.9827 | 18.68 |
| hybrid (w=0.5) | 0.5 | 0.9871 | 18.68 |

All strategies produce identical QRF; confidence is highest for d_freq.

---

## 6. Bottleneck Composition (curve_fit vs SVD)

**Source**: `results/curvefit_vs_svd/timing_data.csv`

| window_len | curve_fit % | SVD % | other % |
|---|---|---|---|
| 100 | 86.5% | 2.2% | 11.3% |
| 1 600 | 61.8% | 16.1% | 22.1% |
| 5 000 | 40.6% | 30.9% | 28.5% |
| 10 000 | 23.1% | 41.7% | 35.2% |

**Crossover** (SVD share overtakes curve_fit share): linear interpolation between
window_len=5000 (curvefit_pct=40.6, svd_pct=30.9) and window_len=6000 (curvefit_pct=30.0,
svd_pct=39.0) gives crossover at **window_len ≈ 5 519**
(source: `results/curvefit_vs_svd/timing_data.csv`, rows at wl=5000 and wl=6000)

---

## 7. Long-Stream Behaviour (N=60 000, fs=1000 Hz)

**Config**: window_len=300, stride=150, fs=1000 Hz, SNR=20 dB, chirp_plus_sinusoid
**Budget**: T_w = 150 ms (stride/fs = 150/1000)
**Source**: `results/long_stream/*/run_summary.json`

| Engine | n_windows | p95 (ms) | max (ms) | peak mem (MiB) | RT PASS? |
|---|---|---|---|---|---|
| Baseline SSD | 399 | **140.1** | 387.4 | 6.80 | ✓ Yes (margin: 9.9 ms) |
| OptimizedSSD-fwhm | 399 | **27.3** | 78.8 | 1.43 | ✓ Yes |

Active trajectories: baseline [1, 12], fwhm [1, 11].

**Memory bounded check**: both engines show bounded (non-monotonically-increasing) peak
memory after the first ~50 windows; no evidence of memory leak was observed in the
long-stream run. (Verify by inspecting `long_stream_metrics.csv` memory column.)

**Note on baseline margin**: The baseline p95=140.1 ms is only 9.9 ms below the 150 ms
budget. Any additional overhead (e.g., from more complex signals or slower hardware)
could push the baseline above budget. The OptimizedSSD-fwhm margin (122.7 ms) is
considerably safer.

---

## 8. Crossover Window Length

SVD share overtakes curve_fit share at approximately **window_len ≈ 5 519 samples**
(linear interpolation between adjacent rows in `results/curvefit_vs_svd/timing_data.csv`).

At this window length the SVD becomes the dominant bottleneck, and further bandwidth
estimation speedups will yield diminishing returns. This aligns with the visual
observation that the speedup-vs-window-length plot flattens above wl ≈ 3 200.

---

## 9. Slide Deck Corrections

Numbers on checkpoint slides 7 and 8 that disagree with the canonical values above:

### Slide 7 Corrections

| Slide claim | Canonical value | Source |
|---|---|---|
| "8× faster (baseline: 68.56 ms → OptimizedSSD-fwhm: 8.36 ms)" | **9.51× (48.59 ms → 5.11 ms)** at fs=10 000 Hz profiling config | `results/profiling/optimized_report.txt` |
| "5× memory reduction (11.6 → 2.1 MiB)" | **5.62× (13.16 → 2.34 MiB)** | `results/profiling/optimized_report.txt`, "Peak memory" lines |

> **Discrepancy note**: The brief expected canonical values of "~4×, ~63.8 → ~15.0 ms",
> but the actual profiling report shows 9.51× and 48.59 → 5.11 ms. The profiling config
> (fs=10 000 Hz, budget=15 ms) differs from the long-stream config (fs=1000 Hz, budget=150 ms).
> At fs=1000 Hz (benchmark grid, wl≈300): baseline≈47–49 ms, fwhm≈9–12 ms, speedup≈4–6×.
> The slide's baseline value "68.56 ms" does not appear in any current result file;
> it may come from an older experimental run or a different config. **Human verification needed.**

### Slide 8 Corrections

| Slide claim | Canonical value | Source |
|---|---|---|
| "FWHM bandwidth share 1.7%" | **1.6%** (bw_frac_of_window=0.016 → 1.60%) | `results/bandwidth_eval/level4_latency.csv`, fwhm row |
| "p95 = 25 ms (OptimizedSSD-fwhm)" | **8.49 ms** (at fs=10 000 Hz profiling) OR **27.3 ms** (long-stream at fs=1000 Hz) | `results/profiling/optimized_report.txt` vs `results/long_stream/optimized_fwhm/run_summary.json` |
| "baseline p95 = 91 ms" | **73.91 ms** (profiling at fs=10 000 Hz) OR **140.1 ms** (long-stream at fs=1000 Hz) | `results/profiling/optimized_report.txt` vs `results/long_stream/baseline/run_summary.json` |

> **Config discrepancy**: The two sets of p95 values (profiling vs long-stream) use different
> sampling frequencies and thus different real-time budgets. Slides should specify the config
> clearly. Recommended: use the long-stream values (fs=1000 Hz, budget=150 ms) as they
> represent the primary thesis evaluation configuration.