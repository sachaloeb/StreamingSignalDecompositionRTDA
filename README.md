# Streaming Signal Decomposition for Real-Time Data Analysis

A Python framework that adapts Singular Spectrum Decomposition (SSD) — an
iterative, fully-automated SSA-based method — to operate on streaming time
series via a sliding-window framework with principled component matching
across windows. Developed as part of a bachelor thesis at DACS, Maastricht
University (Spring 2026).

## Installation

```bash
pip install -r requirements.txt
```

## Quickstart

Run the baseline experiment (chirp + sinusoid, 30 000 samples at 1 kHz):

```bash
python experiments/run_experiment.py \
    --config experiments/configs/baseline.yaml \
    --output-dir results/baseline
```

Run the test suite:

```bash
pytest tests/ -v
```

## Module Overview

| Module | Purpose |
|---|---|
| `src/engines/base.py` | `DecompositionEngine` Strategy interface + factory |
| `src/engines/ssd.py` | Reference SSD algorithm (Bonizzi et al. 2014), MATLAB-aligned |
| `src/engines/ssa.py` | Base SSA, autoSSA with hierarchical grouping |
| `src/engines/ssd_optimized.py` | `OptimizedSSD`: algorithmic SSD with FWHM/moment/Jacobian bandwidth |
| `src/engines/ssd_incremental.py` | `IncrementalSSD`: warm-start SVD caching across windows |
| `src/engines/ssd_rank1.py` | `RankOneIncrementalSSD`: Brand (2003) rank-1 SVD update |
| `src/engines/svd_update.py` | `RankOneUpdater`: rank-1 Hankel SVD update (Brand 2003 / USSA) |
| `src/engines/rsvd.py` | Randomized SVD (Halko, Martinsson & Tropp 2011) |
| `src/streaming/window_manager.py` | Circular buffer + stride logic |
| `src/streaming/component_matcher.py` | Hungarian matching across windows |
| `src/streaming/trajectory_store.py` | Rolling component trajectory management |
| `src/metrics/similarity.py` | d_corr, d_freq, subspace_angle, w_correlation |
| `src/metrics/stability.py` | QRF, frequency drift, energy continuity, NMSE |
| `experiments/synthetic/generators.py` | Synthetic signal generators |
| `experiments/run_experiment.py` | CLI entry point for streaming experiments |
| `experiments/profile_pipeline.py` | cProfile + tracemalloc pipeline profiler |
| `experiments/profile_optimized.py` | OptimizedSSD variant profiling comparison |
| `experiments/benchmark_complexity.py` | Complexity sweep: window length, components, SNR |
| `experiments/plot_curvefit_vs_svd.py` | curve_fit vs SVD cost breakdown by window size |

## Algorithm References

Bonizzi, P., Karel, J. M. H., Meste, O., & Peeters, R. L. M. (2014).
Singular spectrum decomposition: A new method for time series
decomposition. *Advances in Adaptive Data Analysis*, 6(04), 1450011.

Harmouche, J., Fourer, D., Auger, F., Borgnat, P., & Flandrin, P.
(2017). The sliding singular spectrum analysis: A data-driven
nonstationary signal decomposition tool. *IEEE Transactions on Signal
Processing*, 66(1), 251–263.

Saeed, M., & Alty, S. R. (2020). USSA: A unified singular spectrum
analysis framework with application to real-time data. In *Proc. IEEE
ICASSP 2020* (pp. 4837–4841).

Kotala, S., Bonizzi, P., Boussé, M., Karel, J., Peeters, R., & Dreesen, P.
(2025). Randomized singular spectrum decomposition for data-driven signal
decomposition. In *Proc. BMSC 2025*.

Brand, M. (2003). Fast online SVD revisions for lightweight recommender
systems. In *Proc. SIAM International Conference on Data Mining* (pp. 37–46).
