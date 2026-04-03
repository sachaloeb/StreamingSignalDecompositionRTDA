"""Unit tests for metric temporal-scope contracts.

Each metric must be computable only within its documented temporal
scope: intra-window, cross-window (pairwise), or global aggregate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics.stability import (
    energy_continuity,
    freq_drift_aggregate,
    qrf,
    singular_value_drift,
)


def test_qrf_intra_window_perfect_reconstruction() -> None:
    """Perfect reconstruction returns +inf without history."""
    signal = np.sin(
        2 * np.pi * 10 * np.linspace(0, 1, 500)
    )
    result = qrf(signal, signal.copy())
    assert np.isinf(result) and result > 0


def test_qrf_intra_window_known_value() -> None:
    """QRF matches the analytic formula for a noisy recon."""
    rng = np.random.default_rng(0)
    signal = rng.standard_normal(500)
    noise = rng.standard_normal(500) * 0.1
    recon = signal + noise
    result = qrf(signal, recon)
    expected = 20.0 * np.log10(
        np.linalg.norm(signal)
        / np.linalg.norm(signal - recon)
    )
    assert abs(result - expected) < 1e-9
    assert result > 0


def test_singular_value_drift_nan_at_t0() -> None:
    """No previous window -> NaN."""
    S = np.array([5.0, 3.0, 1.0])
    assert np.isnan(singular_value_drift(S, None))


def test_singular_value_drift_cross_window() -> None:
    """Positive drift between two known S vectors."""
    S_prev = np.array([5.0, 3.0, 1.0])
    S_curr = np.array([5.1, 2.9, 1.1])
    result = singular_value_drift(S_curr, S_prev)
    expected = float(np.linalg.norm(S_curr - S_prev))
    assert abs(result - expected) < 1e-9
    assert result > 0


def test_energy_continuity_nan_at_t0() -> None:
    """No previous components -> NaN."""
    c = [np.ones(100), np.ones(100) * 0.5]
    matching = {0: 0, 1: 1}
    assert np.isnan(energy_continuity(c, None, matching))


def test_energy_continuity_cross_window() -> None:
    """Squared energy differences match analytic expectation."""
    prev = [
        np.ones(100) * 2.0,
        np.ones(100) * 1.0,
    ]
    curr = [
        np.ones(100) * 2.1,
        np.ones(100) * 0.9,
    ]
    matching = {0: 0, 1: 1}
    result = energy_continuity(curr, prev, matching)
    E_prev_0 = float(np.dot(prev[0], prev[0]))
    E_curr_0 = float(np.dot(curr[0], curr[0]))
    E_prev_1 = float(np.dot(prev[1], prev[1]))
    E_curr_1 = float(np.dot(curr[1], curr[1]))
    expected = (
        (E_curr_0 - E_prev_0) ** 2
        + (E_curr_1 - E_prev_1) ** 2
    )
    assert abs(result - expected) < 1e-6
    assert result > 0


def test_freq_drift_aggregate_is_postrun_only() -> None:
    """Variance matches numpy and NaN entries are ignored."""
    rng = np.random.default_rng(42)
    freqs = 10.0 + rng.standard_normal(50) * 0.5
    result = freq_drift_aggregate(freqs)
    assert abs(result - float(np.var(freqs, ddof=0))) < 1e-9

    freqs_with_nan = np.concatenate(
        [freqs, [np.nan, np.nan]]
    )
    result_nan = freq_drift_aggregate(freqs_with_nan)
    assert abs(result_nan - result) < 1e-9


def test_freq_drift_aggregate_nan_on_insufficient_data(
) -> None:
    """Fewer than 2 finite values -> NaN."""
    assert np.isnan(
        freq_drift_aggregate([np.nan, np.nan])
    )
    assert np.isnan(freq_drift_aggregate([10.0]))
