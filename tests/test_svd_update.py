"""Tests for the RankOneUpdater (Brand's rank-1 SVD update)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

from src.engines.svd_update import RankOneUpdater, _build_hankel


class TestRankOneUpdate:
    """Test the rank-1 SVD update preserves factorisation quality."""

    def _make_signal(self, n: int = 200, seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        t = np.arange(n) / 100.0
        return np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 12 * t) + 0.1 * rng.standard_normal(n)

    def test_single_update_accuracy(self) -> None:
        """After 1 rank-1 update, Frobenius error should be small."""
        x = self._make_signal(100)
        L = 20
        X = _build_hankel(x, L)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        r = 10
        updater = RankOneUpdater(U[:, :r], S[:r], Vt[:r, :])

        # Perform a rank-1 update: add a random outer product
        rng = np.random.default_rng(0)
        a = rng.standard_normal(L)
        b = rng.standard_normal(X.shape[1])

        updater.update(a, b)
        X_new = X + np.outer(a, b)

        # Reconstruct from updated factors
        X_approx = updater.U @ np.diag(updater.S) @ updater.Vt
        err = np.linalg.norm(X_new - X_approx, 'fro') / np.linalg.norm(X_new, 'fro')
        assert err < 0.1, f"Relative Frobenius error {err:.6f} too large after 1 update"

    def test_multiple_updates_accuracy(self) -> None:
        """After 10 updates, error should remain bounded."""
        rng = np.random.default_rng(1)
        x = self._make_signal(100)
        L = 20
        X = _build_hankel(x, L)
        K = X.shape[1]
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        r = 10
        updater = RankOneUpdater(U[:, :r], S[:r], Vt[:r, :])

        X_running = X.copy()
        for _ in range(10):
            a = rng.standard_normal(L) * 0.1
            b = rng.standard_normal(K) * 0.1
            updater.update(a, b)
            X_running += np.outer(a, b)

        X_approx = updater.U @ np.diag(updater.S) @ updater.Vt
        err = np.linalg.norm(X_running - X_approx, 'fro') / np.linalg.norm(X_running, 'fro')
        assert err < 0.2, f"Relative Frobenius error {err:.6f} too large after 10 updates"

    def test_refresh_resets_error(self) -> None:
        """After a full SVD reset, error should drop back to truncation level."""
        rng = np.random.default_rng(2)
        x = self._make_signal(100)
        L = 20
        X = _build_hankel(x, L)
        K = X.shape[1]
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        r = 10
        updater = RankOneUpdater(U[:, :r], S[:r], Vt[:r, :], refresh_every=5)

        X_running = X.copy()
        for _ in range(20):
            a = rng.standard_normal(L) * 0.1
            b = rng.standard_normal(K) * 0.1
            updater.update(a, b)
            X_running += np.outer(a, b)

        # Force a reset
        updater._full_svd_reset(X_running)
        X_approx = updater.U @ np.diag(updater.S) @ updater.Vt
        err = np.linalg.norm(X_running - X_approx, 'fro') / np.linalg.norm(X_running, 'fro')
        # After full reset with rank r, error should be small (just truncation)
        assert err < 0.1, f"Error after reset {err:.6f} should be near truncation level"

    def test_zero_input(self) -> None:
        """Update with zero vectors should not crash."""
        x = self._make_signal(50)
        L = 10
        X = _build_hankel(x, L)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        r = 5
        updater = RankOneUpdater(U[:, :r], S[:r], Vt[:r, :])

        a = np.zeros(L)
        b = np.zeros(X.shape[1])
        U_new, S_new, Vt_new = updater.update(a, b)

        assert not np.any(np.isnan(S_new)), "NaN in singular values after zero update"

    def test_constant_input(self) -> None:
        """Update with constant vectors should not produce NaN."""
        x = np.ones(50)
        L = 10
        X = _build_hankel(x, L)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        r = min(3, len(S))
        updater = RankOneUpdater(U[:, :r], S[:r], Vt[:r, :])

        a = np.ones(L)
        b = np.ones(X.shape[1])
        U_new, S_new, Vt_new = updater.update(a, b)
        assert not np.any(np.isnan(S_new))

    def test_rank_deficient_matrix(self) -> None:
        """Rank-1 update on a rank-1 matrix should not crash."""
        L, K = 10, 20
        u = np.random.default_rng(3).standard_normal(L)
        v = np.random.default_rng(4).standard_normal(K)
        X = np.outer(u, v)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        r = 1
        updater = RankOneUpdater(U[:, :r], S[:r], Vt[:r, :])

        a = np.random.default_rng(5).standard_normal(L)
        b = np.random.default_rng(6).standard_normal(K)
        U_new, S_new, Vt_new = updater.update(a, b)
        assert S_new.shape == (1,)
        assert not np.any(np.isnan(S_new))


class TestSlideWindow:
    """Test the slide_window method for Hankel matrix sliding."""

    def test_slide_window_runs(self) -> None:
        """slide_window should run without error on a simple signal."""
        rng = np.random.default_rng(10)
        N = 60
        L = 15
        x = np.sin(2 * np.pi * 3 * np.arange(N) / 100.0)

        X = _build_hankel(x[:50], L)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        r = 5
        updater = RankOneUpdater(U[:, :r], S[:r], Vt[:r, :])

        # Slide by one sample
        new_window = x[1:51]
        U_new, S_new, Vt_new = updater.slide_window(
            float(x[50]), new_window,
        )
        assert U_new.shape[0] == L
        assert len(S_new) == r
        assert not np.any(np.isnan(S_new))

    def test_slide_window_multiple_steps(self) -> None:
        """Multiple slide_window steps should not accumulate NaN."""
        N = 100
        L = 15
        x = np.sin(2 * np.pi * 5 * np.arange(N) / 100.0)
        window_size = 50

        X = _build_hankel(x[:window_size], L)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        r = 5
        updater = RankOneUpdater(U[:, :r], S[:r], Vt[:r, :])

        for i in range(1, N - window_size):
            new_window = x[i: i + window_size]
            U_new, S_new, Vt_new = updater.slide_window(
                float(x[i + window_size - 1]), new_window,
            )
            assert not np.any(np.isnan(S_new)), f"NaN at step {i}"


class TestBuildHankel:
    """Test the _build_hankel utility function."""

    def test_hankel_shape(self) -> None:
        x = np.arange(10, dtype=float)
        L = 4
        X = _build_hankel(x, L)
        assert X.shape == (4, 7)

    def test_hankel_values(self) -> None:
        x = np.arange(5, dtype=float)
        L = 3
        X = _build_hankel(x, L)
        expected = np.array([
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
        ], dtype=float)
        np.testing.assert_array_equal(X, expected)