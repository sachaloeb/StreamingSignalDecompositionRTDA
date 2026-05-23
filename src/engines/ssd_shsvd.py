"""SSD with Strobach (1997) Square Hankel SVD subspace tracking (SHSVD 1).

Implements the SHSVD 1 algorithm from Table 1 of:

    Strobach, P. (1997). Square Hankel SVD subspace tracking algorithms.
    Signal Processing, 57(1), 1–18.

Two key departures from the generic IncrementalSSD (Option B wrapper):

1. **Standard (non-wrapped) Hankel matrix** of size L × K where
   L = (N + 1) // 2, giving L ≈ K (approximately square).  This is
   what Strobach's paper actually uses.

2. **Single subspace tracked** (Q_r, L × r left singular vectors).
   Because the square Hankel is symmetric, U = V exactly.  Here L ≈ K
   so U ≈ V; we track Q_r as the left subspace and approximate the
   right subspace via one projection.  Vt is recovered from B = X^T Q_L.

3. **Standard diagonal averaging** for component reconstruction
   (not the wrapped variant used by the base SSD).

Algorithm (Strobach 1997, Eq. 4 / Table 1 SHSVD 1)
---------------------------------------------------
    State : Q_r(t−1)  (L × r)

    A(t)     = X_sq(t) Q_r(t−1)       X_sq = X[:, :L]  (L × L square)
    Q_r(t), R(t) = QR( A(t) )
    Ω(t)     = Q_r^T(t−1) Q_r(t)      [Eq. 13]
    f(t)     = R(t) Ω(t)               [Eq. 45, singular value estimate]
    S(t)     = |diag( f(t) )|

    Vt recovered via  B = X^T Q_r(t),  Vt = (B / ‖B‖_col)^T

Q_r is updated once per ``fit`` call from the full window trajectory
matrix; ``_decompose_trajectory`` reads Q_r but never writes it.
"""

from __future__ import annotations

import numpy as np

from src.engines.ssa import build_trajectory_matrix, diagonal_averaging, svd_decompose
from src.engines.ssd import SSD


class SHSVDIncrementalSSD(SSD):
    """SSD with Strobach (1997) SHSVD 1 square Hankel subspace tracking.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz.
    nmse_threshold : float, optional
        NMSE stopping criterion.  Default 0.01.
    max_iter : int, optional
        Maximum SSD extraction iterations.  Default 20.
    """

    def __init__(
        self,
        fs: float,
        **kwargs: object,
    ) -> None:
        super().__init__(fs=fs, **kwargs)

        # SHSVD 1 tracker state — left singular subspace (L × r).
        # L is the embedding dimension of the square Hankel (≈ N/2).
        # Updated once per fit call; never modified inside _decompose_trajectory.
        self._Q_r: np.ndarray | None = None  # (L, r)
        self._shsvd_L: int = 0
        self._shsvd_K: int = 0

    # ------------------------------------------------------------------
    # Overrides — standard Hankel + standard reconstruction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_trajectory_matrix(x: np.ndarray, M: int) -> np.ndarray:
        """Standard (non-wrapped) Hankel with L = (N+1)//2 (approximately square).

        The frequency-adaptive M parameter from SSD is ignored; the
        embedding dimension is fixed to L = (N+1)//2 so that L ≈ K,
        matching Strobach's square Hankel assumption.

        Parameters
        ----------
        x : np.ndarray
            Signal of length N.
        M : int
            Ignored; kept for interface compatibility.

        Returns
        -------
        np.ndarray
            Standard Hankel matrix of shape (L, K) where L = (N+1)//2
            and K = N - L + 1.
        """
        N = len(x)
        L = N // 2  # guarantees L ≤ K so X[:, :L] is a true L×L square sub-matrix
        return build_trajectory_matrix(x, L)  # (L, K)  standard Hankel

    @staticmethod
    def _reconstruct_component(X_sub: np.ndarray, N: int) -> np.ndarray:
        """Standard anti-diagonal averaging (not wrapped).

        Parameters
        ----------
        X_sub : np.ndarray
            Rank-reduced trajectory matrix (L, K).
        N : int
            Original signal length; used only as a sanity check — the
            result of diagonal_averaging already has length L+K-1 = N.

        Returns
        -------
        np.ndarray
            Reconstructed component of length N.
        """
        return diagonal_averaging(X_sub)

    # ------------------------------------------------------------------
    # SHSVD 1 decomposition hook
    # ------------------------------------------------------------------

    def _decompose_trajectory(
        self,
        X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One SHSVD 1 step on the approximately square Hankel.

        Uses the square sub-matrix X_sq = X[:, :L] (L × L) for the
        orthogonal iteration step, matching Strobach's square Hankel
        assumption exactly.  Vt is then recovered from the full X (L × K).

        Falls back to a full SVD cold start when Q_r is uninitialised or
        the matrix dimensions have changed.
        """
        L, K = X.shape

        if self._Q_r is None or self._shsvd_L != L or self._shsvd_K != K:
            return self._cold_start(X, L, K)

        Q_r_prev = self._Q_r          # (L, r)

        # --- Square sub-matrix X_sq = X[:, :L]  (exactly L × L) ---
        X_sq = X[:, :L]

        # --- SHSVD 1 Eq. (4): A(t) = X_sq(t) Q_r(t−1)  [L × r] ---
        A = X_sq @ Q_r_prev           # (L, r)

        # --- QR factorisation: A(t) = Q_r(t) R(t) ---
        Q_r_new, R = np.linalg.qr(A)  # (L, r), (r, r)

        # --- Cosines matrix Ω(t) = Q_r^T(t−1) Q_r(t)  [Strobach Eq. 13] ---
        Omega = Q_r_prev.T @ Q_r_new  # (r, r)

        # --- Singular value estimate f(t) = R(t) Ω(t)  [Strobach Eq. 45] ---
        F = R @ Omega                  # (r, r)
        S = np.abs(np.diag(F))        # (r,)

        # --- Recover Vt from full X (L × K) ---
        B = X.T @ Q_r_new             # (K, r)
        col_norms = np.maximum(np.linalg.norm(B, axis=0), 1e-14)
        Vt = (B / col_norms).T        # (r, K)

        # Sort by descending singular value.
        order = np.argsort(S)[::-1]
        return Q_r_new[:, order], S[order], Vt[order, :]

    # ------------------------------------------------------------------
    # Override fit — advance SHSVD state after each window
    # ------------------------------------------------------------------

    def fit(self, x: np.ndarray) -> list[np.ndarray]:
        """Decompose *x* with SHSVD 1 square Hankel subspace tracking.

        Runs the full SSD pipeline using the cached Q_r, then advances
        Q_r by one SHSVD 1 step on the full window's square Hankel.
        """
        result = super().fit(x)

        x_arr = np.asarray(x, dtype=np.float64)
        x_zm = x_arr - np.mean(x_arr)
        # M argument is ignored by _build_trajectory_matrix override.
        X_full = self._build_trajectory_matrix(x_zm, 0)
        self._advance_shsvd_state(X_full)

        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _cold_start(
        self,
        X: np.ndarray,
        L: int,
        K: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Full SVD cold start; does not modify tracker state."""
        return svd_decompose(X)

    def _advance_shsvd_state(self, X: np.ndarray) -> None:
        """Advance Q_r by one SHSVD 1 step on the full window matrix.

        Seeds Q_r from the left singular vectors of a full SVD when
        uninitialised or when dimensions change.

        Parameters
        ----------
        X : np.ndarray
            Standard Hankel trajectory matrix (L, K) for the current window.
        """
        L, K = X.shape

        if self._Q_r is None or self._shsvd_L != L or self._shsvd_K != K:
            U, _, _ = svd_decompose(X)
            self._Q_r = U.copy()      # (L, r)
            self._shsvd_L = L
            self._shsvd_K = K
            return

        # One SHSVD 1 step using the square sub-matrix.
        X_sq = X[:, :L]
        A = X_sq @ self._Q_r          # (L, r)
        Q_r_new, _ = np.linalg.qr(A)  # (L, r)
        self._Q_r = Q_r_new
        self._shsvd_L = L
        self._shsvd_K = K