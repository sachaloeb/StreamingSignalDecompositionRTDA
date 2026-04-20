"""Integration tests for metrics temporal-scope contracts.

Runs the full streaming pipeline and verifies that the output
``metrics.csv`` and ``run_summary.json`` respect the documented
temporal scopes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTPUT_DIR = "results/metrics_temporal_test"


@pytest.fixture(scope="module", autouse=True)
def _run_pipeline() -> None:
    """Execute the streaming experiment once for the module."""
    from experiments.run_experiment import run

    run(
        config_path="experiments/configs/baseline.yaml",
        output_dir=OUTPUT_DIR,
    )


def test_metrics_csv_temporal_consistency() -> None:
    """metrics.csv respects temporal scope contracts."""
    df = pd.read_csv(f"{OUTPUT_DIR}/metrics.csv")

    assert df["qrf"].isna().sum() == 0, (
        "QRF must be finite at every window — "
        "it is intra-window"
    )
    assert (df["qrf"] > -np.inf).all()

    assert np.isnan(
        df["singular_value_drift"].iloc[0]
    ), (
        "singular_value_drift must be NaN at t=0 "
        "(no previous window)"
    )
    assert (
        df["singular_value_drift"].iloc[1:].isna().sum()
        == 0
    ), (
        "singular_value_drift must be finite "
        "for all windows after t=0"
    )

    assert np.isnan(
        df["energy_continuity"].iloc[0]
    ), "energy_continuity must be NaN at t=0"
    assert (
        df["energy_continuity"].iloc[1:].isna().sum() == 0
    ), (
        "energy_continuity must be finite "
        "for all windows after t=0"
    )

    finite_fmax = df["f_max_t0"].dropna()
    assert len(finite_fmax) > 0
    assert (finite_fmax > 1.0).all(), (
        "f_max_c0 must log raw frequency in Hz, "
        "not variance"
    )


def test_run_summary_contains_freq_drift() -> None:
    """run_summary.json must exist with freq_drift_c0."""
    summary_path = Path(f"{OUTPUT_DIR}/run_summary.json")
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert "freq_drift_t0" in summary
    val = summary["freq_drift_t0"]
    assert np.isfinite(val) or np.isnan(val)
