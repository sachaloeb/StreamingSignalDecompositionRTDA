"""Tests for src.engines.ssd (SSD) and src.engines.ssa (autoSSA)."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.synthetic.generators import two_sinusoids
from src.metrics.stability import nmse, qrf
from src.engines.ssd import SSD
from src.engines.ssa import auto_ssa, build_trajectory_matrix


class TestSSD:
    """Integration tests for the full SSD algorithm."""

    def test_ssd_two_sinusoids(self) -> None:
        fs = 200.0
        x = two_sinusoids(N=1000, f1=5.0, f2=25.0, fs=fs, seed=0)
        ssd = SSD(fs=fs, nmse_threshold=0.01, max_iter=20)
        components = ssd.fit(x)
        assert len(components) >= 2

        recon_sum = sum(components)
        overall_qrf = qrf(x, recon_sum)
        assert overall_qrf > 10.0

    def test_wrapped_trajectory_matrix(self) -> None:
        x = np.arange(10, dtype=np.float64)
        M = 4
        X = SSD._build_trajectory_matrix(x, M)
        assert X.shape == (M, len(x))
        np.testing.assert_array_equal(X[0, :], x)

    def test_nmse_decreases(self) -> None:
        fs = 200.0
        x = two_sinusoids(N=500, f1=5.0, f2=25.0, fs=fs, seed=1)
        ssd = SSD(fs=fs, nmse_threshold=1e-6, max_iter=10)
        components = ssd.fit(x)

        residual = x.copy()
        nmse_vals: list[float] = []
        for c in components[:-1]:
            residual = residual - c
            nmse_vals.append(nmse(residual, x))

        for i in range(1, len(nmse_vals)):
            assert nmse_vals[i] <= nmse_vals[i - 1] + 1e-6


class TestAutoSSA:
    """Tests for the auto_ssa grouping function."""

    def test_auto_ssa_grouping(self) -> None:
        fs = 200.0
        N = 1000
        x = two_sinusoids(N=N, f1=5.0, f2=25.0, fs=fs, seed=2)
        L = 100
        groups = auto_ssa(x, r=2, L=L)
        assert len(groups) == 2

        recon = groups[0] + groups[1]
        energy_diff = np.dot(x - recon, x - recon) / np.dot(x, x)
        assert energy_diff < 0.01


class TestTrajectoryMatrix:
    """Tests for the standard Hankel trajectory matrix."""

    def test_standard_shape(self) -> None:
        x = np.arange(20, dtype=np.float64)
        L = 7
        X = build_trajectory_matrix(x, L)
        K = len(x) - L + 1
        assert X.shape == (L, K)
