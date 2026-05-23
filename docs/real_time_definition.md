# Operational Definition of "Real-Time" for Streaming SSD

*Streaming Signal Decomposition for Real-Time Data Analysis (RTDA)*
*Bachelor Thesis, DACS, Maastricht University, Spring 2026*

---

## 1. Definition

A **soft real-time, per-window deadline** definition:

> The streaming pipeline is **real-time-capable** on a given platform if the
> **95th-percentile per-window total processing time** does not exceed the
> **inter-window arrival period** T_w = stride / f_s.

Formally, let T_i denote the wall-clock time to process window i (from the
first `engine.fit()` call to the final `trajectory_store.update()` return).
The pipeline passes the real-time criterion if and only if:

```
Quantile_0.95 { T_i : i = 1, …, W } ≤ T_w = stride / f_s
```

where W is the total number of windows in the evaluation stream.

---

## 2. At Default Configuration

| Parameter | Value | Derived |
|---|---|---|
| `stride` | 150 samples | — |
| `f_s` | 1000 Hz | — |
| **T_w** | **150 ms** | 150 / 1000 = 0.150 s |
| `window_len` | 300 samples | 0.300 s of signal per window |
| Overlap | 150 samples | window_len − stride |

At these settings, the pipeline must process each 300-sample window within
150 ms to keep pace with incoming data.

---

## 3. What This Criterion Is

- **Soft real-time**: missing the deadline on ≤5% of windows is acceptable.
  Missing it on the 95th-percentile window would mean the pipeline falls behind
  on 1 in 20 windows, which causes bounded accumulating delay.
- **Deadline-based**: the deadline is derived from the signal acquisition rate,
  not from a physical danger threshold. No safety-critical consequences are
  implied.
- **Throughput real-time**: the criterion measures throughput (can the pipeline
  keep up on average, with bounded outliers?) rather than deterministic worst-
  case execution time (WCET).
- **Empirical at p95**: the criterion is evaluated empirically from a long
  streaming run (N = 60 000 samples, 399 windows) on the specific hardware
  described in Appendix [Y].  It is not derived analytically.

---

## 4. What This Criterion Is Not

- **Not hard real-time**: no formal worst-case execution time (WCET) bound is
  established; no interrupt-driven scheduling or real-time OS guarantees exist.
  Python's GIL, garbage collector, and OS scheduling jitter can cause sporadic
  outlier windows.
- **Not end-to-end algorithmic latency**: the trajectory output for a given
  sample is delayed by at least `window_len / f_s` additional seconds relative
  to acquisition (e.g., 0.300 s at default settings), plus any overlap
  averaging delay. The criterion addresses processing throughput, not sensing-
  to-output latency.
- **Not language- or hardware-portable**: the empirical p95 values are specific
  to the Python/NumPy/SciPy stack on the hardware enumerated in Appendix [Y].
  Porting to a compiled language (C, Rust) or a different microarchitecture
  would require a fresh evaluation.
- **Not a general streaming claim**: the criterion applies only to the specific
  decomposition engines, window configurations, and signal types listed in the
  evaluation tables.  Signals with many more components (e.g., EEG, high-DOF
  vibration) will produce longer per-window times and may fail this criterion.

---

## 5. Scope of Claims

All real-time claims in this thesis are bounded to:

1. **Signals**: the synthetic test signals listed in §[X] — specifically
   `chirp_plus_sinusoid`, `two_sinusoids`, `rossler`, `component_onset`, and
   `n_sinusoids` at the parameter settings defined in `experiments/configs/`.

2. **Hardware**: the workstation described in Appendix [Y] (CPU model, RAM,
   OS version), as recorded in each experiment's `run_summary.json` under the
   `"environment"` key.

3. **Configurations**: the window lengths, stride values, and engine parameters
   explicitly enumerated in the evaluation tables. Extrapolation to larger
   windows (e.g., window_len > 3 200) or different stride-to-window ratios is
   not warranted without fresh measurement.

4. **Engines**: only engines for which a complete long-stream run
   (`results/long_stream/*/long_stream_metrics.csv`) is available constitute
   verified real-time-capable configurations.

---

## References

- Bonizzi et al. (2014). *Singular Spectrum Decomposition.* — core algorithm.
- Harmouche et al. (2017). *Sliding SSA for Non-Stationary Signals.* —
  sliding-window framing for streaming decomposition.
- Saeed, Took & Alty (2020). *USSA.* — rank-1 streaming update framework.