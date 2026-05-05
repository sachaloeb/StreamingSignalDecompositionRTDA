"""Statistical analysis of the multi-seed SNR sweep.

Reads snr_sweep.csv and writes snr_sweep_stats.csv containing:
- Per-(signal, SNR, engine): median QRF and bootstrap 95% CI
- Wilcoxon paired test (each optimized engine vs ssd baseline)
- Benjamini-Hochberg FDR-corrected q-values

Usage
-----
    python experiments/snr_sweep_stats.py \\
        --csv results/snr_sweep_multiseed/snr_sweep.csv \\
        --out results/snr_sweep_multiseed
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Benjamini-Hochberg FDR correction (manual, no pandas required)
# ---------------------------------------------------------------------------

def _bh_correction(pvals: list[float], alpha: float = 0.05) -> list[float]:
    """Benjamini-Hochberg FDR correction.

    Returns BH-adjusted q-values.  If scipy.stats.false_discovery_control
    is available (scipy >= 1.11), it is used; otherwise falls back to a
    manual implementation.

    Parameters
    ----------
    pvals : list[float]
        Raw p-values (NaN entries are passed through unchanged).
    alpha : float
        FDR level.  Default 0.05.

    Returns
    -------
    list[float]
        BH q-values, same length as pvals.
    """
    try:
        from scipy.stats import false_discovery_control
        finite_mask = [not (p != p) for p in pvals]  # not NaN
        all_finite = all(finite_mask)
        if all_finite:
            return list(false_discovery_control(pvals, method="bh"))
    except ImportError:
        pass

    # Manual BH
    n = len(pvals)
    qvals = [float("nan")] * n
    indexed = [(i, p) for i, p in enumerate(pvals) if p == p]  # drop NaN
    if not indexed:
        return qvals

    indexed.sort(key=lambda x: x[1])
    m = len(indexed)
    running_min = float("inf")
    ranks_qval: list[tuple[int, float]] = []
    for rank_1_based, (orig_i, pv) in enumerate(reversed(indexed), start=1):
        bh_q = min(running_min, pv * m / (m - rank_1_based + 1))
        running_min = bh_q
        ranks_qval.append((orig_i, bh_q))

    for orig_i, q in ranks_qval:
        qvals[orig_i] = q
    return qvals


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def _bootstrap_ci(
    values: list[float],
    n_boot: int = 1000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Bootstrap 95% CI for the median."""
    if rng is None:
        rng = np.random.default_rng(0)
    arr = np.array([v for v in values if np.isfinite(v)])
    if len(arr) == 0:
        return float("nan"), float("nan")
    medians = np.array([
        np.median(rng.choice(arr, size=len(arr), replace=True))
        for _ in range(n_boot)
    ])
    lo = float(np.percentile(medians, (1 - ci) / 2 * 100))
    hi = float(np.percentile(medians, (1 + ci) / 2 * 100))
    return lo, hi


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

FIELDNAMES_OUT = [
    "signal", "snr_db", "engine",
    "median_qrf_db", "ci_lo", "ci_hi",
    "wilcoxon_pval_vs_baseline", "bh_qval", "reject_null",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="SNR sweep statistical analysis")
    parser.add_argument(
        "--csv",
        default="results/snr_sweep_multiseed/snr_sweep.csv",
    )
    parser.add_argument(
        "--out",
        default="results/snr_sweep_multiseed",
    )
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    from scipy.stats import wilcoxon as _wilcoxon

    # Load CSV
    rows_raw: list[dict] = []
    with open(args.csv, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows_raw.append({
                "signal": row["signal"],
                "snr_db": float(row["snr_db"]),
                "engine": row["engine"],
                "seed": int(row["seed"]),
                "median_qrf_db": float(row["median_qrf_db"])
                    if row["median_qrf_db"] not in ("nan", "") else float("nan"),
            })
    print(f"Loaded {len(rows_raw)} rows from {args.csv}")

    signals = sorted(set(r["signal"] for r in rows_raw))
    snrs = sorted(set(r["snr_db"] for r in rows_raw))
    engines = sorted(set(r["engine"] for r in rows_raw))
    optimized_engines = [e for e in engines if e != "ssd"]

    rng = np.random.default_rng(42)
    n_boot = args.n_boot

    # Build lookup: (signal, snr, engine) -> {seed: qrf}
    lookup: dict[tuple, dict[int, float]] = {}
    for r in rows_raw:
        key = (r["signal"], r["snr_db"], r["engine"])
        lookup.setdefault(key, {})[r["seed"]] = r["median_qrf_db"]

    # Phase 1: compute per-cell stats + raw Wilcoxon p-values
    out_rows: list[dict] = []
    raw_pval_records: list[tuple[int, dict]] = []  # (row_index, partial_dict)

    for sig in signals:
        for snr in snrs:
            # Baseline values per seed
            baseline_by_seed = lookup.get(("ssd".replace("ssd", "ssd"), snr, "ssd"), {})
            # Rewrite to use actual key
            baseline_by_seed = lookup.get((sig, snr, "ssd"), {})

            for eng in engines:
                qrf_by_seed = lookup.get((sig, snr, eng), {})
                qrf_vals = [
                    v for v in qrf_by_seed.values() if np.isfinite(v)
                ]
                med = float(np.median(qrf_vals)) if qrf_vals else float("nan")
                ci_lo, ci_hi = _bootstrap_ci(qrf_vals, n_boot=n_boot, rng=rng)

                # Wilcoxon: only for optimized engines vs baseline
                pval = float("nan")
                if eng != "ssd" and baseline_by_seed:
                    # Pair by seed
                    seeds_common = sorted(
                        set(qrf_by_seed.keys()) & set(baseline_by_seed.keys())
                    )
                    diffs = [
                        qrf_by_seed[s] - baseline_by_seed[s]
                        for s in seeds_common
                        if np.isfinite(qrf_by_seed.get(s, float("nan")))
                        and np.isfinite(baseline_by_seed.get(s, float("nan")))
                    ]
                    if len(diffs) >= 2:
                        try:
                            stat, pval = _wilcoxon(
                                diffs,
                                zero_method="wilcox",
                                alternative="two-sided",
                            )
                            pval = float(pval)
                        except Exception:
                            pval = float("nan")

                row_idx = len(out_rows)
                partial = {
                    "signal": sig,
                    "snr_db": snr,
                    "engine": eng,
                    "median_qrf_db": round(med, 4) if np.isfinite(med) else float("nan"),
                    "ci_lo": round(ci_lo, 4) if np.isfinite(ci_lo) else float("nan"),
                    "ci_hi": round(ci_hi, 4) if np.isfinite(ci_hi) else float("nan"),
                    "wilcoxon_pval_vs_baseline": pval,
                    "bh_qval": float("nan"),   # filled in below
                    "reject_null": False,
                }
                out_rows.append(partial)
                if eng != "ssd" and np.isfinite(pval):
                    raw_pval_records.append((row_idx, partial))

    # Phase 2: BH correction across all optimized-engine cells
    pvals_for_bh = [partial["wilcoxon_pval_vs_baseline"] for _, partial in raw_pval_records]
    if pvals_for_bh:
        qvals = _bh_correction(pvals_for_bh, alpha=args.alpha)
        for (row_idx, partial), qv in zip(raw_pval_records, qvals):
            partial["bh_qval"] = round(qv, 6) if np.isfinite(qv) else float("nan")
            partial["reject_null"] = bool(np.isfinite(qv) and qv < args.alpha)
            out_rows[row_idx] = partial

    # Save stats CSV
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_path = out_dir / "snr_sweep_stats.csv"
    with open(stats_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES_OUT, extrasaction="ignore")
        writer.writeheader()
        for row in out_rows:
            writer.writerow({
                k: ("nan" if isinstance(v, float) and not np.isfinite(v) else v)
                for k, v in row.items()
            })
    print(f"Stats saved to {stats_path}  ({len(out_rows)} rows)")

    # Summary table: per (engine vs baseline), count of cells where BH null rejected
    print("\n--- Rejection summary (BH-corrected, α=0.05) ---")
    print(f"  {'engine':35s}  cells_tested  n_rejected")
    print("  " + "-" * 60)
    for eng in optimized_engines:
        eng_rows = [r for r in out_rows if r["engine"] == eng]
        tested = [r for r in eng_rows if np.isfinite(r["wilcoxon_pval_vs_baseline"])]
        rejected = [r for r in tested if r["reject_null"]]
        print(
            f"  {eng:35s}  {len(tested):12d}  {len(rejected):10d}  "
            f"({'NONE rejected — equivalence holds' if not rejected else 'SOME rejected — CHECK'})"
        )

    print(
        "\nNote: Most cells should NOT reject — that confirms OptimizedSSD ≈ baseline QRF."
    )


if __name__ == "__main__":
    main()
