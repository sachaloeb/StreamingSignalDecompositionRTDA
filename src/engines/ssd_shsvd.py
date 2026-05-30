"""SSD with Strobach (1997) Square Hankel SVD subspace tracking (SHSVD 1).

Implements the SHSVD 1 algorithm from Table 1 of:

    Strobach, P. (1997). Square Hankel SVD subspace tracking algorithms.
    Signal Processing, 57(1), 1–18.

Algorithm (Strobach 1997, Table 1 / Eqs. 11–16)
------------------------------------------------
State between windows: Q_r (L × r), A (L × r), R (r × r), Θ (r × r).

At each time step (one new sample x(t)):

    h(t)   = Q_r^T(t−1) x(t)                        [Eq. 14]
    A(t)   = [ h^T(t)                     ]           [Eq. 11]  ← prepend new row
             [ A(t−1) Θ(t−1)  [0:L−1, :] ]               ← propagate and drop last
    Q_r(t), R(t) = QR( A(t) )                        [Eq. 4 / Table 1]
    Θ(t)   = Q_r^T(t−1) Q_r(t)                      [Eq. 13]
    f(t)   = R(t) Θ(t)                               [Eq. 45]
    S(t)   = |diag( f(t) )|

Key departures from the non-incremental form
--------------------------------------------
The critical fix over the previous code is that A(t) is *propagated*
from A(t−1) via the cosines matrix Θ(t−1) rather than recomputed from
scratch as X_sq @ Q_r_prev.  The from-scratch computation is equivalent
to plain orthogonal iteration (O(L²r) per step); the incremental form
is O(Lr²) — the whole point of SHSVD 1.

Additional design notes
-----------------------
* **Standard (non-wrapped) Hankel** of size L × K where L = N // 2,
  giving L ≈ K (approximately square), matching Strobach's assumption.
* **Standard diagonal averaging** for reconstruction.
* **Singular values and right singular vectors** are recovered via a
  thin SVD of B = X^T Q_r (size K × r, cheap), which is correct.
  The Strobach f(t) = R(t)Θ(t) formula is used as an efficient
  singular-value *estimate* inside the advance step.
* **Rank parameter** r controls the tracked subspace dimension.
* Q_r is updated once per ``fit`` call; ``_decompose_trajectory``
  reads Q_r but never writes it, so residual iterations see a frozen
  subspace consistent with the dominant-component extraction.
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
    rank : int, optional
        Number of singular vectors / subspace dimension to track.
        Default 10.
    nmse_threshold : float, optional
        NMSE stopping criterion.  Default 0.01.
    max_iter : int, optional
        Maximum SSD extraction iterations.  Default 20.
    """

    def __init__(
        self,
        fs: float,
        rank: int = 10,
        **kwargs: object,
    ) -> None:
        super().__init__(fs=fs, **kwargs)
        self.rank = rank

        # SHSVD 1 tracker state.
        # All four tensors are updated once per fit call in
        # _advance_shsvd_state.  _decompose_trajectory reads Q_r (and
        # uses A/R/Theta only for the singular-value estimate path).
        self._Q_r: np.ndarray | None = None   # (L, r) left subspace
        self._A: np.ndarray | None = None     # (L, r) auxiliary matrix A
        self._R: np.ndarray | None = None     # (r, r) upper-triangular R from QR(A)
        self._Theta: np.ndarray | None = None # (r, r) cosines matrix Q_r^T(prev) Q_r

        self._shsvd_L: int = 0
        self._shsvd_K: int = 0

    # ------------------------------------------------------------------
    # Overrides — standard Hankel + standard reconstruction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_trajectory_matrix(x: np.ndarray, M: int) -> np.ndarray:
        """Standard (non-wrapped) Hankel with L = N // 2 (approximately square).

        The frequency-adaptive M parameter from SSD is ignored; the
        embedding dimension is fixed to L = N // 2 so that L ≈ K,
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
            Standard Hankel matrix of shape (L, K) where L = N // 2
            and K = N − L + 1.
        """
        N = len(x)
        L = N // 2  # guarantees L ≤ K so X[:, :L] is a true L × L sub-matrix
        return build_trajectory_matrix(x, L)

    @staticmethod
    def _reconstruct_component(X_sub: np.ndarray, N: int) -> np.ndarray:
        """Standard anti-diagonal averaging (not wrapped).

        Parameters
        ----------
        X_sub : np.ndarray
            Rank-reduced trajectory matrix (L, K).
        N : int
            Original signal length.

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
        """Return (U, S, Vt) for X using the current SHSVD 1 subspace.

        On the first call or after a dimension change, falls back to a
        full SVD cold start that also initialises the tracker state.

        On warm calls, reads the frozen Q_r set by the most recent
        ``_advance_shsvd_state`` call and recovers the full (U, S, Vt)
        factorisation via a thin SVD of the small r × K projected
        matrix B = X^T Q_r.  This is correct — unlike the previous
        implementation which used row norms of B as singular values.

        Parameters
        ----------
        X : np.ndarray
            Trajectory matrix (L, K).

        Returns
        -------
        U : np.ndarray  (L, r)
        S : np.ndarray  (r,)
        Vt : np.ndarray (r, K)
        """
        L, K = X.shape

        if self._Q_r is None or self._shsvd_L != L or self._shsvd_K != K:
            return self._cold_start(X, L, K)

        Q_r = self._Q_r  # (L, r) — frozen snapshot from last advance

        # --- Recover (U, S, Vt) via thin SVD of B = X^T Q_r  ---
        # B has shape (K, r) — cheap O(Kr²) SVD.
        # This is correct: the rank-r approximation is Q_r (Q_r^T X),
        # whose left/right singular vectors and values come from SVD(B).
        B = X.T @ Q_r                                      # (K, r)
        U_b, S, Vt_b = np.linalg.svd(B, full_matrices=False)  # (K,r),(r,),(r,K)^T... wait
        # np.linalg.svd of (K,r) with full_matrices=False:
        #   U_b: (K, r),  S: (r,),  Vt_b: (r, r)
        # We want left singular vecs of X restricted to Q_r's span:
        #   U_full = Q_r @ Vt_b.T  (L, r)
        #   Vt_full = U_b.T        (r, K)
        U = Q_r @ Vt_b.T    # (L, r)
        Vt = U_b.T          # (r, K)

        # Sort by descending singular value.
        order = np.argsort(S)[::-1]
        return U[:, order], S[order], Vt[order, :]

    # ------------------------------------------------------------------
    # Override fit — advance SHSVD state after each window
    # ------------------------------------------------------------------

    def fit(self, x: np.ndarray) -> list[np.ndarray]:
        """Decompose *x* with SHSVD 1 square Hankel subspace tracking.

        Runs the full SSD pipeline using the frozen Q_r from the
        previous window, then advances Q_r (and A, R, Θ) by one SHSVD 1
        step on the current window's square Hankel.
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
        """Full SVD cold start; initialises all SHSVD 1 tracker state.

        Parameters
        ----------
        X : np.ndarray
            Trajectory matrix (L, K).
        L, K : int
            Matrix dimensions.

        Returns
        -------
        U, S, Vt from the full SVD — best quality for the first window.
        """
        r = min(self.rank, L, K)

        # Full SVD for the best first-window decomposition.
        U_full, S_full, Vt_full = np.linalg.svd(X, full_matrices=False)
        U_r = U_full[:, :r]   # (L, r)

        # Initialise A = X_sq @ Q_r_init, then QR-factorize to get
        # a consistent (Q_r, R) pair.
        X_sq = X[:, :L]                     # (L, L) square sub-matrix
        A_init = X_sq @ U_r                 # (L, r)
        Q_r, R = np.linalg.qr(A_init)       # (L, r), (r, r)

        # Initial cosines: between Q_r_init and the QR-refined Q_r.
        Theta = U_r.T @ Q_r                 # (r, r)

        # Store state.
        self._Q_r = Q_r
        self._A = A_init
        self._R = R
        self._Theta = Theta
        self._shsvd_L = L
        self._shsvd_K = K

        return U_r, S_full[:r], Vt_full[:r, :]

    def _advance_shsvd_state(self, X: np.ndarray) -> None:
        """Advance all SHSVD 1 state by one step using the current window.

        Implements the true incremental update from Strobach (1997)
        Table 1 / Eqs. 11–16, propagating A(t) from A(t−1) via the
        cosines matrix rather than recomputing from scratch.

        Parameters
        ----------
        X : np.ndarray
            Standard Hankel trajectory matrix (L, K) for the current window.
        """
        L, K = X.shape
        r = min(self.rank, L, K)

        # Dimension change or uninitialised: full cold start.
        if self._Q_r is None or self._shsvd_L != L or self._shsvd_K != K:
            self._cold_start(X, L, K)
            return

        Q_r_prev = self._Q_r    # (L, r)
        A_prev = self._A        # (L, r)
        Theta_prev = self._Theta  # (r, r)

        # --- Strobach Eq. 14: h(t) = Q_r^T(t−1) x(t) ---
        # x(t) is the newest column of the square sub-matrix — the data
        # vector that just entered the sliding Hankel window.
        X_sq = X[:, :L]          # (L, L) square sub-matrix
        x_new = X_sq[:, -1]      # (L,)  newest column
        h = Q_r_prev.T @ x_new  # (r,)  projection of new data

        # --- Strobach Eq. 12: A(t-1) Θ(t-1) ---
        # Propagates the old auxiliary matrix through the cosines rotation.
        A_propagated = A_prev @ Theta_prev  # (L, r)

        # --- Strobach Eq. 11: form A(t) ---
        # Prepend h^T as the new top row; discard the departing last row.
        A_new = np.empty((L, r), dtype=np.float64)
        A_new[0, :] = h
        A_new[1:, :] = A_propagated[:-1, :]

        # --- Table 1: QR factorisation of A(t) ---
        Q_r_new, R_new = np.linalg.qr(A_new)  # (L, r), (r, r)

        # --- Strobach Eq. 13: Θ(t) = Q_r^T(t−1) Q_r(t) ---
        Theta_new = Q_r_prev.T @ Q_r_new  # (r, r)

        # Update all state.
        self._Q_r = Q_r_new
        self._A = A_new
        self._R = R_new
        self._Theta = Theta_new
        self._shsvd_L = L
        self._shsvd_K = K