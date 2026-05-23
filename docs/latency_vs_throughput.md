# Latency vs Throughput: Two Distinct Real-Time Notions

## Why two latencies, not one

Streaming decomposition systems operate under two distinct timing constraints that are often conflated:

1. **Throughput real-time** — Can the system process each window before the next one arrives?
2. **End-to-end algorithmic latency** — How long after a sample enters the input buffer does its contribution appear in an emitted trajectory?

These are independent quantities. A system can satisfy throughput real-time (processing each window within its deadline) while still imposing substantial end-to-end latency due to buffering and matching delays. Both must be reported separately.

## Throughput real-time (operational definition used in this thesis)

The operational definition of real-time used in this thesis is the throughput deadline (see `docs/real_time_definition.md`). The budget per window is:

T_w = stride / fs = 150 / 1000 = 150 ms

(`results/long_stream/baseline/run_summary.json`, `results/long_stream/optimized_fwhm/run_summary.json`)

Validated p95 latencies on the long-stream test (N=60,000, 399 windows):

| Engine | p95 (ms) | Budget (ms) | Status |
|--------|----------|-------------|--------|
| Baseline SSD | 140.14 | 150.0 | PASS (borderline) |
| OptimizedSSD-FWHM | 27.304 | 150.0 | PASS (comfortable) |

(Source: `results/long_stream/{baseline,optimized_fwhm}/run_summary.json`)

The baseline engine passes but with only 6.6% margin; a slightly heavier signal or background CPU load could push it over the deadline. OptimizedSSD-FWHM passes with 81.8% margin.

## End-to-end algorithmic latency

The end-to-end algorithmic latency is the wall-clock delay between a sample arriving at the input buffer and that sample's contribution being reflected in an emitted trajectory. It decomposes into three terms:

**(a) Window-fill delay.** The sliding window must accumulate `window_len` samples before the first decomposition can run:

t_fill = window_len / fs = 300 / 1000 = 300 ms

This is a fundamental lower bound — no streaming decomposition method can emit a trajectory before seeing at least one full window.

**(b) Per-window processing time.** Once a window is complete, it must be decomposed, matched, and stored. At p95:

- Baseline SSD: 140.14 ms (`results/long_stream/baseline/run_summary.json`)
- OptimizedSSD-FWHM: 27.304 ms (`results/long_stream/optimized_fwhm/run_summary.json`)

**(c) Matcher lookback delay.** The component matcher uses a lookback buffer of up to 10 previous windows to resolve ambiguous matches. In the worst case, a component's identity is not confirmed until all lookback windows have been processed:

t_lookback_worst = lookback × stride / fs = 10 × 150 / 1000 = 1500 ms

In practice, most matches resolve at lookback=1 (150 ms), since stable components have consistent spectral signatures.

## What this thesis claims and does not claim

**Claims.** This thesis claims throughput real-time on the tested platform (Apple Silicon, macOS) and signal types (chirp + sinusoid at SNR=20 dB). OptimizedSSD-FWHM comfortably meets the deadline; the baseline engine meets it but is borderline.

**Does not claim.** This thesis does *not* claim minimal end-to-end algorithmic latency. Reducing the end-to-end latency requires one or more of:

- **Smaller `window_len`**: reduces t_fill but fundamentally limits frequency resolution (the minimum resolvable frequency is approximately fs / window_len).
- **Smaller `stride`**: reduces the inter-window gap but increases compute load proportionally.
- **`lookback=0`**: eliminates matcher lookback delay but loses the cross-window robustness that lookback provides.

All three are explicit design trade-offs documented in the thesis.

## Numerical example

Using the default configuration (`window_len=300`, `stride=150`, `fs=1000`):

| Delay component | Value |
|-----------------|-------|
| Window-fill delay | 300 ms |
| OptimizedSSD-FWHM p95 processing | 27.304 ms (`results/long_stream/optimized_fwhm/run_summary.json`) |
| **First-match algorithmic latency** | **≈ 327 ms** (window-fill + processing) |
| Worst-case with lookback=10 | ≈ 300 + 10 × 150 + 27.304 = **1827 ms** |
| Typical with lookback=1 | ≈ 300 + 150 + 27.304 = **477 ms** |

For the baseline engine, replace 27.304 ms with 140.14 ms (`results/long_stream/baseline/run_summary.json`), giving a first-match latency of ≈ 440 ms.