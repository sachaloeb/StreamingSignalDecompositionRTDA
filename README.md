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
| `src/ssd/core.py` | Full SSD algorithm (Bonizzi et al. 2014) |
| `src/ssd/ssa.py` | Base SSA, autoSSA with hierarchical grouping |
| `src/ssd/svd_update.py` | Rank-1 USSA update skeleton (stub) |
| `src/streaming/window_manager.py` | Circular buffer + stride logic |
| `src/streaming/component_matcher.py` | Hungarian matching across windows |
| `src/streaming/trajectory_store.py` | Rolling component trajectory management |
| `src/metrics/similarity.py` | d_corr, d_freq, subspace_angle, w_correlation |
| `src/metrics/stability.py` | QRF, frequency drift, energy continuity, NMSE |
| `experiments/synthetic/generators.py` | Synthetic signal generators |
| `experiments/run_experiment.py` | CLI entry point for experiments |

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
