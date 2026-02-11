# Streaming Signal Decomposition (StreamSSD)

A reproducible Python framework for streaming signal decomposition with **SSD (Singular Spectrum Decomposition)** as the main algorithm, based on Bonizzi et al. 2014. The framework supports sliding-window decomposition with component tracking and consistency metrics across consecutive windows.

## Features

- **SSD Engine**: Iterative extraction of narrowband oscillatory components using SSA-like embedding
- **SSA Baseline**: Classical batch SSA for comparison
- **Component Tracking**: Hungarian assignment for matching components across windows
- **Consistency Metrics**: Cross-window correlation, L2 difference, energy delta, frequency delta
- **Reproducible**: Fixed seeds, config-driven runs, deterministic outputs
- **End-to-End Demo**: One command runs the full pipeline

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package in editable mode
pip install -e .

# Install development dependencies (optional, for tests)
pip install -e ".[dev]"
```

## Quick Start

Run the end-to-end demo:

```bash
python scripts/run_demo.py --config configs/demo.yaml
```

This will:
1. Generate a synthetic signal (trend + sinusoids + chirp + noise)
2. Extract sliding windows
3. Decompose each window using SSD (or SSA)
4. Track components across consecutive windows
5. Compute consistency metrics
6. Save results to `outputs/`:
   - `metrics.csv`: Per-window matched component metrics
   - `tracks.json`: Component track assignments over time
   - `signal_and_components.png`: Original signal and decomposition visualization
   - `stability_metrics.png`: Time-series of stability metrics

## Repository Structure

```
streamssd/
  __init__.py
  window.py                # Streaming buffer & window extraction
  embed.py                 # Hankel embedding utilities
  reconstruct.py           # Diagonal averaging (hankelization)
  tracking.py              # Component alignment + Hungarian assignment
  metrics.py               # Cross-window consistency metrics
  utils.py                 # Seed, logging, helpers
  engines/
    __init__.py
    base.py                # Engine interface and dataclasses
    ssa_batch.py           # SSA baseline engine
    ssd_bonizzi.py         # SSD engine (main focus)

configs/
  demo.yaml                # Demo configuration

experiments/
  synthetic.py             # Synthetic signal generators

scripts/
  run_demo.py              # End-to-end demo script
  run_benchmark.py         # Benchmark scaffold (placeholder)

tests/
  test_embed.py            # Tests for embedding
  test_reconstruct.py      # Tests for reconstruction
  test_tracking.py         # Tests for tracking
  test_metrics.py          # Tests for metrics
```

## Configuration

Edit `configs/demo.yaml` to customize:

- **Signal parameters**: Sampling frequency, duration, noise level
- **Window parameters**: Window length (W), stride (s)
- **Embedding**: Hankel dimension (L)
- **Engine**: Type (ssd/ssa), number of components, frequency bounds
- **Tracking**: Overlap length, similarity threshold
- **Output**: Output directory, random seed

## Running Tests

```bash
pytest tests/
```

## Engine Interface

All engines implement the `BaseEngine` interface:

```python
from streamssd.engines import SSDBonizziEngine

engine = SSDBonizziEngine(L=100, M=5, fmin=0.5, fmax=20.0)
result = engine.fit_window(x_window, fs=100.0)

# result.components: list of time-domain components
# result.residual: residual signal
# result.meta: metadata (frequencies, scores, etc.)
```

## SSD Algorithm

The SSD engine implements an iterative extraction procedure:

1. Start with `residual = x_window`
2. For each component (m = 1..M):
   - Embed residual into Hankel matrix
   - Compute SVD
   - Evaluate candidate components and select the one with highest narrowbandedness score
   - Reconstruct selected component via diagonal averaging
   - Subtract from residual (deflation)
3. Return M components and final residual

The narrowbandedness score is computed as `peak_power / total_power` in the frequency domain, favoring components with concentrated spectral energy.

## Component Tracking

Components are matched across consecutive windows using:
- **Similarity matrix**: Normalized correlation on overlap region
- **Hungarian assignment**: Optimal matching to maximize total similarity
- **Sign correction**: Automatic sign flipping to maintain consistency

## Metrics

For each matched component pair, the following metrics are computed:
- **corr**: Sign-invariant normalized correlation
- **overlap_l2**: Normalized L2 difference on overlap
- **energy_delta**: Relative energy change
- **freq_delta**: Difference in FFT peak frequency

## Requirements

- Python 3.11+
- numpy >= 1.24.0
- scipy >= 1.10.0
- pandas >= 2.0.0
- matplotlib >= 3.7.0
- pyyaml >= 6.0

## License

This project is provided as-is for research purposes.

## References

- Bonizzi, P., et al. (2014). "Singular Spectrum Decomposition: A new method for time series decomposition." *Advances in Adaptive Data Analysis*, 6(4).
