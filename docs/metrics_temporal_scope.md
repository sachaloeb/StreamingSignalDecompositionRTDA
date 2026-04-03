# Metrics Temporal Scope Reference

## Intra-Window Metrics
Computed from a single window's data. No history required.

| Metric | Formula | Source |
|--------|---------|--------|
| `qrf` | 20·log10(‖x‖/‖x−x̂‖) | Harmouche et al., IEEE TSP 2017 |

## Cross-Window Metrics (Pairwise)
Require the previous window's state. NaN at t=0.

| Metric | Formula | State Required |
|--------|---------|----------------|
| `singular_value_drift` | ‖S_t − S_{t-1}‖_F | S_{t-1} singular values |
| `energy_continuity` | Σ_k(E_k(t)−E_k(t-1))² | E_k(t-1) component energies |

## Global Aggregate (Post-Hoc)
Computed over the full run after the streaming loop. Never logged per-row.

| Metric | Formula | How to Compute |
|--------|---------|----------------|
| `freq_drift` | Var_t[f_max(t)] | `freq_drift_aggregate(df["f_max_c0"])` |

## metrics.csv Column Contract
Each row = one streaming window.

```
window_index | qrf | singular_value_drift | energy_continuity
| matching_confidence | f_max_c0 | f_max_c1 | ...
```

`singular_value_drift` and `energy_continuity` are NaN in row 0.
`f_max_cK` stores the raw dominant frequency (Hz) — NOT variance.

## run_summary.json Keys
Post-hoc aggregates saved alongside metrics.csv:

```json
{
  "freq_drift_c0": 0.42,
  "freq_drift_c1": 1.13
}
```
