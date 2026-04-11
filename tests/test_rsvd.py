"""Tests for the randomised SVD implementation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

from src.engines.rsvd import rsvd
from src.engines.ssa import build_trajectory_matrix, svd_decompose


class TestRSVD:
    """Test the randomised SVD approximation quality."""

    def test_rsvd_vs_full_random_matrix(self) -> None:
        """rSVD on a random matrix should approximate full SVD well."""
        rng = np.random.default_rng(42)
        m, n = 50, 40
        X = rng.standard_normal((m, n))

        k = 10
        U_r, S_r, Vt_r = rsvd(X, k=k, n_oversamples=5, n_power_iter=2, seed=0)

        # Full SVD for reference
        U_f, S_f, Vt_f = np.linalg.svd(X, full_matrices=False)

        # The top-k singular values should be close
        np.testing.assert_allclose(S_r, S_f[:k], rtol=0.1)

    def test_rsvd_reconstruction_quality(self) -> None:
        """Low-rank reconstruction from rSVD should be close to best rank-k."""
        rng = np.random.default_rng(0)
        m, n = 80, 60
        # Create a matrix with fast-decaying spectrum
        U_true = np.linalg.qr(rng.standard_normal((m, 20)))[0]
        S_true = np.exp(-np.arange(20) * 0.5)
        Vt_true = np.linalg.qr(rng.standard_normal((n, 20)))[0].T
        X = U_true @ np.diag(S_true) @ Vt_true

        k = 10
        U_r, S_r, Vt_r = rsvd(X, k=k, seed=1)
        X_approx = U_r @ np.diag(S_r) @ Vt_r

        # Best rank-k approximation
        U_f, S_f, Vt_f = np.linalg.svd(X, full_matrices=False)
        X_best = U_f[:, :k] @ np.diag(S_f[:k]) @ Vt_f[:k, :]

        err_rsvd = np.linalg.norm(X - X_approx, 'fro')
        err_best = np.linalg.norm(X - X_best, 'fro')

        # rSVD error should be within 2x of the optimal
        assert err_rsvd < 2.0 * err_best + 1e-10

    def test_rsvd_on_hankel_matrix(self) -> None:
        """rSVD should work well on Hankel trajectory matrices."""
        t = np.arange(200) / 100.0
        x = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 12 * t)
        L = 50
        X = build_trajectory_matrix(x, L)

        k = 5
        U_r, S_r, Vt_r = rsvd(X, k=k, n_power_iter=2, seed=42)
        X_approx = U_r @ np.diag(S_r) @ Vt_r

        rel_err = np.linalg.norm(X - X_approx, 'fro') / np.linalg.norm(X, 'fro')
        assert rel_err < 0.05, f"rSVD relative error {rel_err:.4f} on Hankel matrix"

    def test_rsvd_graceful_degradation(self) -> None:
        """As k decreases, approximation quality should degrade gracefully."""
        rng = np.random.default_rng(10)
        m, n = 50, 40
        X = rng.standard_normal((m, n))

        errors = []
        for k in [20, 10, 5, 2]:
            U_r, S_r, Vt_r = rsvd(X, k=k, seed=0)
            X_approx = U_r @ np.diag(S_r) @ Vt_r
            err = np.linalg.norm(X - X_approx, 'fro')
            errors.append(err)

        # Errors should be monotonically non-decreasing as k decreases
        for i in range(len(errors) - 1):
            assert errors[i] <= errors[i + 1] + 1e-10, (
                f"Error did not increase with smaller k: {errors}"
            )

    def test_rsvd_shapes(self) -> None:
        """Output shapes should match (m, k), (k,), (k, n)."""
        rng = np.random.default_rng(0)
        m, n = 30, 25
        X = rng.standard_normal((m, n))
        k = 7
        U, S, Vt = rsvd(X, k=k, seed=0)

        assert U.shape == (m, k)
        assert S.shape == (k,)
        assert Vt.shape == (k, n)

    def test_svd_decompose_randomized_method(self) -> None:
        """svd_decompose with method='randomized' should work."""
        t = np.arange(100) / 50.0
        x = np.sin(2 * np.pi * 5 * t)
        L = 25
        X = build_trajectory_matrix(x, L)

        U, S, Vt = svd_decompose(X, rank=5, method="randomized")
        assert U.shape[1] == 5
        assert len(S) == 5
        assert Vt.shape[0] == 5

    def test_svd_decompose_full_unchanged(self) -> None:
        """svd_decompose with method='full' should behave identically to before."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 15))

        U1, S1, Vt1 = svd_decompose(X, rank=5, method="full")
        U2, S2, Vt2 = svd_decompose(X, rank=5)

        np.testing.assert_array_equal(S1, S2)