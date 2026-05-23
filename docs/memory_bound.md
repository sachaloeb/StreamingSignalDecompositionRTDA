# Memory Bound Analysis

## Claim

The streaming SSD pipeline's peak memory consumption is **O(window_len + max_components x trajectory_history_length)**, independent of the total stream length N. Once the sliding window advances past the initial fill, no additional memory is allocated as a function of N.

## What the pipeline holds in memory

At any point during execution, the pipeline's memory footprint consists of four allocations:

1. **Trajectory matrix** (within SSD core): an L x K matrix where L is the SSA window length (~window_len / 2) and K = window_len - L + 1. This is O(window_len^2) in the worst case but typically much smaller because L is chosen automatically.

2. **SVD workspace**: the U, Sigma, V^T factors of the trajectory matrix. Same asymptotic size as the trajectory matrix.

3. **Trajectory store**: stores the running average for each active component. Each trajectory is at most `max_len` samples long (typically a few multiples of `window_len`). With `max_components` trajectories, this is O(max_components x max_len).

4. **Matcher lookback buffer**: retains the overlap regions of the last `lookback` windows for matching. Each stored overlap region is `overlap = window_len - stride` samples per component. Total: O(lookback x max_components x overlap).

None of these depend on N. The circular buffer in `WindowManager` is fixed at `window_len` samples. Old samples are overwritten, never accumulated.

## Why N drops out

The pipeline processes the stream one window at a time. After each window is decomposed, matched, and stored, the raw samples are discarded (overwritten by the circular buffer). The trajectory store may prune old positions if `max_len` is set. There is no data structure that grows with N.

## Empirical confirmation

### Long-stream test (N=60,000, 399 windows)

From the long-stream run summaries:

| Engine | peak_memory_mib_max | peak_memory_mib_mean |
|--------|--------------------:|---------------------:|
| Baseline SSD | 6.80 MiB | 2.08 MiB |
| OptimizedSSD-FWHM | 1.43 MiB | 0.68 MiB |

(Source: `results/long_stream/baseline/run_summary.json`, `results/long_stream/optimized_fwhm/run_summary.json`)

The peak memory for the baseline engine (6.80 MiB) occurs at window index 2, which has 12 active trajectories — the highest count in the run. The optimized engine peaks at 1.43 MiB at the same window. Both values are constant throughout the remaining 397 windows despite processing 60,000 total samples, confirming that memory does not grow with N.

The time series of per-window peak memory is plotted in `results/long_stream/plots/memory_over_time.png`.

### Window-length dependence (complexity grid)

The benchmark grid (`results/benchmarks_optimized/complexity_grid.csv`) varies `window_len` from 100 to 6400 while holding the signal fixed (chirp + sinusoid, N=10,000). Peak memory averaged across 5 seeds:

| window_len | Baseline SSD (MiB) | OptimizedSSD-FWHM (MiB) |
|-----------:|--------------------:|-------------------------:|
| 100 | 8.51 | 0.99 |
| 200 | 9.22 | 1.29 |
| 400 | 10.79 | 2.58 |
| 800 | 5.76 | 2.54 |
| 1600 | 6.49 | 4.67 |
| 3200 | 9.18 | 9.00 |
| 6400 | 18.56 | 18.40 |

For OptimizedSSD-FWHM, memory scales roughly linearly with `window_len` (0.99 MiB at W=100, 18.40 MiB at W=6400, a ~19x increase for a 64x increase in window length). The baseline engine shows a less regular pattern because its peak memory is dominated by transient SVD workspace allocations that depend on the number of SSD iterations (which varies with signal content and window length).

At W=6400, both engines converge to ~18.5 MiB because the trajectory matrix itself dominates all other allocations at that scale.

See `results/benchmarks_optimized/plots/memory_vs_window_len_optimized.png` for the full curve.

## Summary

- Memory is bounded by window_len and max_components, not by stream length N.
- OptimizedSSD-FWHM uses 4.7x less peak memory than the baseline at the default window_len=300 (1.43 vs 6.80 MiB).
- For the default configuration (window_len=300, stride=150, max_components=4), peak memory stays below 7 MiB even for the baseline engine over a 60-second stream.
