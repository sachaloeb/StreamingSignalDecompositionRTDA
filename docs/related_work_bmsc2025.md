# Related Work: BMSC 2025 and This Thesis

## What BMSC 2025 contributes

The BMSC 2025 paper [Bonizzi2025BMSC] [verify against PDF] proposes replacing the full SVD in the SSD inner loop with a randomized SVD (rSVD). The rSVD computes only the top-k singular triplets in O(W x k^2) time rather than the full O(W^3) SVD, where W is the SSA window length (approximately half of the signal window length). This optimization targets the SVD step specifically, which becomes the dominant cost at large window lengths.

The key insight of [Bonizzi2025BMSC] is that the SSD algorithm only uses a small number of eigentriples per iteration (typically 1-5), so computing the full SVD is wasteful. By computing only the top-k singular triplets via randomized projection, the rSVD approach avoids the cubic scaling of the full SVD while preserving decomposition quality [verify against PDF].

## What this thesis contributes

This thesis takes a complementary approach: rather than accelerating the SVD, it accelerates the eigentriple selection step that follows the SVD. In the original SSD algorithm, eigentriple selection works by fitting a Gaussian spectral model to each eigentriple's power spectrum using nonlinear least-squares curve fitting (scipy.optimize.curve_fit). This thesis introduces two alternatives:

1. **FWHM-based selection**: estimates the spectral peak width directly from the discrete power spectrum using a full-width-at-half-maximum (FWHM) calculation, avoiding curve fitting entirely.
2. **Moment-based selection**: computes the spectral centroid and bandwidth from the first and second moments of the power spectrum, also avoiding curve fitting.

Both methods replace the iterative nonlinear optimization with closed-form O(W) computations.

## Why both are correct: the bottleneck depends on W

Profiling data from `results/curvefit_vs_svd/timing_data.csv` reveals that the relative cost of curve fitting vs. SVD shifts dramatically with window length:

| W | curve_fit % | SVD % |
|----:|------------:|------:|
| 200 | 78.75 | 5.08 |
| 1600 | 61.84 | 16.13 |
| 5000 | 40.64 | 30.87 |
| 6000 | 29.95 | 39.01 |

At small-to-moderate window lengths (W < 5000), curve fitting dominates: it consumes 60-80% of total SSD time, while the SVD accounts for only 5-16%. This is the regime where this thesis's FWHM/moment substitution yields the largest speedup — eliminating curve fitting removes the dominant bottleneck.

At large window lengths, the SVD's cubic scaling overtakes curve fitting. Linear interpolation between W=5000 (gap = 40.64 - 30.87 = 9.77 percentage points in favor of curve fitting) and W=6000 (gap = 29.95 - 39.01 = -9.06 percentage points, SVD now dominant) places the crossover at approximately:

W_crossover = 5000 + 1000 x 9.77 / (9.77 + 9.06) = 5519

Below W ~ 5500, curve fitting is the bottleneck and FWHM/moment substitution is the most effective optimization. Above W ~ 5500, SVD is the bottleneck and rSVD is the most effective optimization.

## Operational decision rule

For practitioners choosing between the two optimizations:

- **If W < 5000** (the common case for real-time physiological signal processing at fs = 1000 Hz with window lengths of 100-2000 ms): use FWHM or moment-based eigentriple selection. The SVD is already fast at these sizes, and curve fitting is the bottleneck.

- **If W > 6000** (offline analysis or very low-frequency applications requiring long windows): use rSVD [Bonizzi2025BMSC]. The SVD dominates at these sizes, and accelerating curve fitting alone provides diminishing returns.

- **If 5000 < W < 6000**: both optimizations contribute meaningfully. Combining rSVD with FWHM/moment selection would address both bottlenecks simultaneously, though this combination has not been empirically validated.

The default configuration in this thesis uses W=300 (window_len=300, so SSA window length L ~ 150), placing it firmly in the curve-fitting-dominant regime. At this operating point, the OptimizedSSD-FWHM engine achieves a 5.1x mean speedup over the baseline (10.56 ms vs 69.78 ms; source: `results/long_stream/optimized_fwhm/run_summary.json` and `results/long_stream/baseline/run_summary.json`), while the SVD step itself takes negligible time.

## Limitations

Several caveats apply to the comparison above:

1. **The crossover point is signal-dependent.** The profiling data was collected on a chirp-plus-sinusoid signal. Signals with more components per SSD iteration will shift the curve-fitting cost upward (more Gaussian fits per iteration), potentially pushing the crossover to larger W. Conversely, signals that converge in fewer iterations will reduce the curve-fitting burden.

2. **rSVD quality at small k is not evaluated here.** This thesis does not benchmark rSVD and cannot comment on whether the randomized approximation introduces decomposition errors at small window lengths where the signal-to-noise separation in the singular value spectrum may be less clear [verify against PDF].

3. **Platform dependence.** The profiling was performed on Apple Silicon (ARM, Accelerate BLAS). On platforms with different BLAS implementations (e.g., MKL on Intel), the SVD may be relatively faster or slower, shifting the crossover point.

4. **The two optimizations are not mutually exclusive.** A combined engine using rSVD for the decomposition step and FWHM for the selection step would address both bottlenecks. This is a natural direction for future work.

## References

- [Bonizzi2025BMSC] Bonizzi et al., "Randomized SVD for Singular Spectrum Decomposition," BMSC 2025 [verify against PDF].
- Bonizzi et al., "Singular spectrum decomposition: a new method for time series decomposition," Advances in Adaptive Data Analysis, 2014.
