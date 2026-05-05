"""Tests for the normalized energy continuity metric.

Feeds a sequence of energies [1.0, 1.0, 0.0, 1e6] and asserts the
normalized metric is [nan, 0, 1, ~1].

The matching dict maps curr_index 0 -> prev_index 0 for all windows.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics.stability import energy_continuity_norm


def _make_components(energy: float) -> list[np.ndarray]:
    """Return a single-component list whose L2 energy equals `energy`."""
    # dot(c, c) = energy  =>  c = [sqrt(energy)]  (1-D unit vector scaled)
    return [np.array([float(np.sqrt(max(energy, 0.0)))])]


# Energies: [1.0, 1.0, 0.0, 1e6]
# Transitions and expected norm values:
#   t=0: no previous                           -> nan
#   t=1: E_curr=1.0, E_prev=1.0  -> |0|/(2+eps)  -> 0.0
#   t=2: E_curr=0.0, E_prev=1.0  -> 1/(1+eps)    -> ~1.0
#   t=3: E_curr=1e6, E_prev=0.0  -> 1e6/(1e6+eps) -> ~1.0


def test_first_window_returns_nan():
    curr = _make_components(1.0)
    result = energy_continuity_norm(curr, None, {0: 0})
    assert np.isnan(result), f"Expected nan for first window, got {result}"


def test_equal_energies_returns_zero():
    curr = _make_components(1.0)
    prev = _make_components(1.0)
    result = energy_continuity_norm(curr, prev, {0: 0})
    assert result == pytest.approx(0.0, abs=1e-10), (
        f"Equal energies should give 0, got {result}"
    )


def test_zero_curr_energy_returns_one():
    curr = _make_components(0.0)
    prev = _make_components(1.0)
    result = energy_continuity_norm(curr, prev, {0: 0})
    # |0 - 1| / (0 + 1 + 1e-12) = 1/(1 + 1e-12) ≈ 1.0
    assert result == pytest.approx(1.0, rel=1e-6), (
        f"Zero current energy should give ~1, got {result}"
    )


def test_large_curr_energy_returns_near_one():
    curr = _make_components(1e6)
    prev = _make_components(0.0)
    result = energy_continuity_norm(curr, prev, {0: 0})
    # |1e6 - 0| / (1e6 + 0 + 1e-12) ≈ 1.0
    assert result == pytest.approx(1.0, rel=1e-5), (
        f"Large energy vs zero should give ~1, got {result}"
    )


def test_bounded_in_zero_one():
    """Metric must stay in [0, 1] for any finite energy values."""
    rng = np.random.default_rng(42)
    for _ in range(100):
        e1 = float(rng.uniform(0, 1e8))
        e2 = float(rng.uniform(0, 1e8))
        curr = _make_components(e1)
        prev = _make_components(e2)
        result = energy_continuity_norm(curr, prev, {0: 0})
        assert 0.0 <= result <= 1.0, (
            f"Out of [0,1]: got {result} for E_curr={e1}, E_prev={e2}"
        )


def test_empty_matching_returns_zero():
    curr = _make_components(1.0)
    prev = _make_components(2.0)
    result = energy_continuity_norm(curr, prev, {})
    assert result == 0.0, f"Empty matching should return 0.0, got {result}"


def test_unmatched_entries_skipped():
    curr = _make_components(1.0)
    prev = _make_components(2.0)
    # prev_j = None means unmatched
    result = energy_continuity_norm(curr, prev, {0: None})
    assert result == 0.0, (
        f"All-unmatched entries should return 0.0, got {result}"
    )


def test_full_sequence():
    """End-to-end sequence: energies [1.0, 1.0, 0.0, 1e6]."""
    energies = [1.0, 1.0, 0.0, 1e6]
    components_seq = [_make_components(e) for e in energies]
    matching = {0: 0}

    results = []
    for i, curr in enumerate(components_seq):
        prev = components_seq[i - 1] if i > 0 else None
        results.append(energy_continuity_norm(curr, prev, matching))

    # t=0: nan
    assert np.isnan(results[0])
    # t=1: 0
    assert results[1] == pytest.approx(0.0, abs=1e-10)
    # t=2: ~1
    assert results[2] == pytest.approx(1.0, rel=1e-6)
    # t=3: ~1
    assert results[3] == pytest.approx(1.0, rel=1e-5)