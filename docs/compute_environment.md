# Compute Environment

This appendix documents the hardware and software environment used to produce all experimental results reported in this thesis. All experiments were executed on a single machine; no cloud or cluster resources were used.

## Hardware

| Component | Value |
|-----------|-------|
| CPU | Apple Silicon (ARM, aarch64) |
| Architecture | arm64 |
| OS | macOS 26.3 (Darwin), arm64 |
| RAM | System default (not constrained) |

## Software stack

| Package | Version |
|---------|---------|
| Python | 3.12.4 (Anaconda) |
| NumPy | 2.4.2 |
| SciPy | 1.17.1 |
| Matplotlib | 3.10.8 |

## BLAS backend

NumPy is linked against **Apple Accelerate** (not OpenBLAS or MKL). This is the default BLAS/LAPACK backend on macOS ARM systems and provides hardware-optimized linear algebra routines for Apple Silicon.

```
BLAS: accelerate (system detection)
LAPACK: accelerate (system detection)
Compiler: clang 15.0.0
SIMD baseline: NEON, NEON_FP16, NEON_VFPV4, ASIMD
SIMD extensions found: ASIMDHP, ASIMDDP
```

All SVD computations (the core of the SSD algorithm) are dispatched through Accelerate's LAPACK `dgesdd` routine.

## Wall-time totals

The following table summarizes the total wall-clock time for each major experiment. All times are from the `total_wall_time_s` field in each experiment's `run_summary.json`.

| Experiment | Source | Wall time (s) |
|------------|--------|---------------|
| Complexity grid benchmark | `results/benchmarks_optimized/run_summary.json` | 608.08 |
| SNR sweep (multi-seed) | `results/snr_sweep_multiseed/run_summary.json` | 347.63 |
| NMSE threshold sensitivity | `results/sensitivity/nmse_threshold/run_summary.json` | 50.79 |
| Max components sensitivity | `results/sensitivity/max_components/run_summary.json` | 33.92 |
| Long-stream baseline | `results/long_stream/baseline/run_summary.json` | 27.91 |
| Long-stream optimized FWHM | `results/long_stream/optimized_fwhm/run_summary.json` | 4.27 |
| **Total** | | **1072.60 s (17.9 min)** |

Note: Several smaller experiments (matcher comparisons, demo runs, metrics temporal test) do not record `total_wall_time_s` and are excluded from this total. Their individual runtimes are negligible (< 5 s each).

## Environment fields from run_summary.json

All run_summary.json files that include an `environment` block report identical values:

```json
{
  "python_version": "3.12.4",
  "numpy_version": "2.4.2",
  "scipy_version": "1.17.1",
  "cpu_model": "arm",
  "os": "macOS-26.3-arm64-arm-64bit"
}
```

## Reproducibility note

All experiments use explicit random seeds (typically seeds 0-4 for multi-seed experiments, seed 42 for single-seed runs). The full configuration for each experiment is saved as `config_used.yaml` in the respective results directory. Given the same hardware, OS, and package versions listed above, results should be numerically reproducible. Differences in BLAS backend (e.g., MKL vs Accelerate) may cause minor floating-point discrepancies in SVD results, which could propagate to small differences in decomposition quality metrics and timing.
