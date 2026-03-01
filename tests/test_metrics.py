"""Tests for src.metrics.similarity and src.metrics.stability."""

from __future__ import annotations

import numpy as np
import pytest

from src.metrics.similarity import d_corr, d_freq, subspace_angle, w_correlation
from src.metrics.stability import frequency_drift, qrf


class TestDCorr:
    """Tests for the normalised inner-product distance."""

    def test_d_corr_identical(self) -> None:
        x = np.sin(2.0 * np.pi * 5.0 * np.arange(200) / 200.0)
        assert d_corr(x, x) == pytest.approx(0.0, abs=1e-10)

    def test_d_corr_orthogonal(self) -> None:
        t = np.arange(1000) / 1000.0
        s = np.sin(2.0 * np.pi * t)
        c = np.cos(2.0 * np.pi * t)
        assert d_corr(s, c) == pytest.approx(1.0, abs=0.05)

    def test_d_corr_range(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(50):
            a = rng.standard_normal(100)
            b = rng.standard_normal(100)
            val = d_corr(a, b)
            assert 0.0 <= val <= 1.0 + 1e-12


class TestQrf:
    """Tests for the Quality of Reconstruction Factor."""

    def test_qrf_perfect(self) -> None:
        x = np.arange(10, dtype=np.float64)
        assert qrf(x, x) == np.inf


class TestSubspaceAngle:
    """Tests for the principal subspace angle."""

    def test_subspace_angle_identical(self) -> None:
        rng = np.random.default_rng(7)
        U = np.linalg.qr(rng.standard_normal((20, 5)))[0]
        assert subspace_angle(U, U) == pytest.approx(0.0, abs=1e-10)


class TestWCorrelation:
    """Tests for the weighted correlation."""

    def test_w_correlation_separable(self) -> None:
        N = 1000
        L = 100
        t = np.arange(N) / float(N)
        x = np.sin(2.0 * np.pi * 3.0 * t)
        y = np.sin(2.0 * np.pi * 50.0 * t)
        wc = w_correlation(x, y, L)
        assert wc < 0.1


class TestFrequencyDrift:
    """Tests for frequency_drift."""

    def test_frequency_drift_constant(self) -> None:
        assert frequency_drift([5.0, 5.0, 5.0, 5.0]) == pytest.approx(
            0.0
        )
