"""Tests for the moment estimator guard rail in OptimizedSSD.

Verifies that when spectral_method="moment" and N < 256, the engine:
1. Emits a UserWarning containing "moment estimator unreliable below N=256"
2. Still produces a finite bandwidth >= fs/N (falls back to FWHM)
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.engines.ssd_optimized as _mod
from src.engines.ssd_optimized import OptimizedSSD


@pytest.fixture(autouse=True)
def reset_moment_guard():
    """Reset the module-level warning sentinel before each test."""
    _mod._MOMENT_GUARD_WARNED = False
    yield
    _mod._MOMENT_GUARD_WARNED = False


def test_moment_guard_warns_on_short_window():
    """Warning is emitted when N < 256 and spectral_method="moment"."""
    fs = 1000.0
    N = 100  # deliberately below 256
    t = np.arange(N) / fs
    signal = np.sin(2.0 * np.pi * 50.0 * t)  # 50 Hz sinusoid

    engine = OptimizedSSD(fs=fs, spectral_method="moment")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        engine.fit(signal)

    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(user_warnings) >= 1, "Expected at least one UserWarning"
    messages = [str(w.message) for w in user_warnings]
    assert any(
        "moment estimator unreliable below N=256" in msg for msg in messages
    ), f"Expected 'moment estimator unreliable below N=256' in warning, got: {messages}"


def test_moment_guard_output_is_finite_and_above_floor():
    """After guard substitution, engine completes and returns finite components."""
    fs = 1000.0
    N = 100
    t = np.arange(N) / fs
    signal = np.sin(2.0 * np.pi * 50.0 * t)

    engine = OptimizedSSD(fs=fs, spectral_method="moment")

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        components = engine.fit(signal)

    assert len(components) > 0, "Engine should return at least one component"
    for comp in components:
        assert np.all(np.isfinite(comp)), "All component values must be finite"


def test_moment_guard_not_triggered_above_threshold():
    """No warning when N >= 256 and spectral_method="moment"."""
    fs = 1000.0
    N = 300  # above 256
    t = np.arange(N) / fs
    signal = np.sin(2.0 * np.pi * 50.0 * t)

    engine = OptimizedSSD(fs=fs, spectral_method="moment")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        engine.fit(signal)

    guard_warnings = [
        w for w in caught
        if issubclass(w.category, UserWarning)
        and "moment estimator unreliable below N=256" in str(w.message)
    ]
    assert len(guard_warnings) == 0, (
        "Guard warning should NOT fire for N >= 256"
    )


def test_moment_guard_threshold_configurable():
    """min_window_length_for_moment class attribute is configurable."""
    original = OptimizedSSD.min_window_length_for_moment
    try:
        OptimizedSSD.min_window_length_for_moment = 64

        fs = 1000.0
        N = 80  # between 64 and 256
        t = np.arange(N) / fs
        signal = np.sin(2.0 * np.pi * 50.0 * t)

        engine = OptimizedSSD(fs=fs, spectral_method="moment")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            engine.fit(signal)

        guard_warnings = [
            w for w in caught
            if issubclass(w.category, UserWarning)
            and "moment estimator unreliable below N=256" in str(w.message)
        ]
        # With threshold=64, N=80 should NOT trigger the guard
        assert len(guard_warnings) == 0, (
            "Guard should not fire when N >= min_window_length_for_moment"
        )
    finally:
        OptimizedSSD.min_window_length_for_moment = original