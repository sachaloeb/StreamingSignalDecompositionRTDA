"""SSD with GROUSE subspace tracking.

Implements the Grassmannian Rank-One Update Subspace Estimation (GROUSE)
algorithm of Balzano, Nowak & Recht (2010), as analysed in relation to
incremental SVD by Balzano & Wright (2013).

References
----------
Balzano, L., Nowak, R., & Recht, B. (2010). Online identification and
    tracking of subspaces from highly incomplete information.
    In Proc. 48th Allerton Conference.

Balzano, L. & Wright, S. J. (2013). On GROUSE and incremental SVD.
    arXiv:1307.5494.

Algorithm
---------
For each column v = X[:, j] (complete observations, no missing data):

    p      = U^T v                         projection onto subspace
    r_vec  = v - U p                       residual orthogonal to U
    r̂     = r_vec / ‖r_vec‖
    p̂     = p / ‖p‖
    θ      = step_size                     fixed step on Grassmannian
    δ      = (cos θ − 1) U p̂ + sin θ r̂  rank-1 tangent step
    U_new  = U + δ p̂^T                    geodesic update; U remains orthonormal

After processing all N columns, singular values and right singular
vectors are recovered from U^T X.

The left subspace U (M × rank) is updated once per ``fit`` call from
all columns of the full window trajectory matrix.  ``_decompose_trajectory``
reads U but never writes it, preventing residual-signal updates from
corrupting the tracker.
"""

from __future__ import annotations

import numpy as np

from src.engines.ssa import svd_decompose
from src.engines.ssd import SSD


class GrouseIncrementalSSD(SSD):
    """SSD with GROUSE (Balzano et al. 2010 / Balzano & Wright 2013) subspace tracking.

    Maintains a rank-*rank* left singular subspace U (M × rank) and
    advances it with GROUSE geodesic steps as new trajectory matrices
    arrive each window.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz.
    rank : int, optional
        Subspace rank to track.  Should be ≥ the number of dominant
        spectral components expected in the signal.  Default 10.
    step_size : float, optional
        Fixed geodesic step size θ (radians) for each GROUSE update.
        Smaller values give smoother tracking; larger values adapt faster.
        Default 0.1.
    nmse_threshold : float, optional
        NMSE stopping criterion.  Default 0.01.
    max_iter : int, optional
        Maximum SSD extraction iterations.  Default 20.
    """

    def __init__(
        self,
        fs: float,
        rank: int = 10,
        step_size: float = 0.1,
        **kwargs: object,
    ) -> None:
        super().__init__(fs=fs, **kwargs)
        self.rank = rank
        self.step_size = step_size

        # GROUSE tracker state — left singular subspace, shape (M, rank).
        # Updated once per fit call from all columns of the full window.
        # _decompose_trajectory reads this but never writes it.
        self._U: np.ndarray | None = None  # (M, rank)
        self._grouse_M: int = 0

    # ------------------------------------------------------------------
    # GROUSE decomposition hook
    # ------------------------------------------------------------------

    def _decompose_trajectory(
        self,
        X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decompose X using the current GROUSE subspace estimate.

        If a cached U is available and the row dimension matches, the
        cached subspace is used to compute (U, S, Vt) directly from
        U^T X.  Otherwise falls back to a full SVD cold start.

        Note: U is NOT updated here.  Updates happen in ``fit`` via
        ``_advance_grouse_state`` to prevent residual trajectory matrices
        from corrupting the subspace estimate.
        """
        M, K = X.shape
        r = min(self.rank, M, K)

        if self._U is None or self._grouse_M != M:
            return self._cold_start(X, M, r)

        U = self._U  # (M, rank)

        # Recover S and Vt from U^T X.
        B = U.T @ X                                    # (rank, K)
        col_norms_B = np.linalg.norm(B, axis=1)        # (rank,) — S estimates
        col_norms_B = np.maximum(col_norms_B, 1e-14)
        Vt = B / col_norms_B[:, None]                  # (rank, K)  normalised rows

        # Sort by descending singular value.
        order = np.argsort(col_norms_B)[::-1]
        return U[:, order], col_norms_B[order], Vt[order, :]

    # ------------------------------------------------------------------
    # Override fit — advance GROUSE state after each window
    # ------------------------------------------------------------------

    def fit(self, x: np.ndarray) -> list[np.ndarray]:
        """Decompose *x* with GROUSE subspace tracking.

        Runs the full SSD pipeline using the read-only cached U, then
        advances U by one full pass of GROUSE over all columns of the
        current window's trajectory matrix.
        """
        result = super().fit(x)

        x_arr = np.asarray(x, dtype=np.float64)
        x_zm = x_arr - np.mean(x_arr)
        M = self._choose_window_length(x_zm)
        X_full = self._build_trajectory_matrix(x_zm, M)
        self._advance_grouse_state(X_full)

        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _cold_start(
        self,
        X: np.ndarray,
        M: int,
        r: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Full SVD cold start; does not modify tracker state."""
        return svd_decompose(X, rank=r)

    def _advance_grouse_state(self, X: np.ndarray) -> None:
        """Advance U by one full GROUSE pass over all columns of X.

        Seeds U from the top-*rank* left singular vectors of a full SVD
        when uninitialised or when the trajectory matrix row dimension M
        has changed.

        Parameters
        ----------
        X : np.ndarray
            Trajectory matrix (M, K) for the current window.
        """
        M, K = X.shape
        r = min(self.rank, M, K)

        if self._U is None or self._grouse_M != M:
            U, _, _ = svd_decompose(X, rank=r)
            self._U = U.copy()         # (M, r)
            self._grouse_M = M
            return

        U = self._U.copy()             # (M, rank) — work on a copy

        # One GROUSE pass: update U using each column of X in sequence.
        theta = self.step_size
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        for j in range(K):
            v = X[:, j]                # (M,)

            p = U.T @ v                # (r,) projection
            r_vec = v - U @ p          # (M,) residual

            rho_r = float(np.linalg.norm(r_vec))
            rho_p = float(np.linalg.norm(p))

            if rho_r < 1e-14 or rho_p < 1e-14:
                continue               # v already in subspace or trivial

            r_hat = r_vec / rho_r      # (M,) unit residual
            p_hat = p / rho_p          # (r,) unit projection

            # Geodesic rank-1 update on the Grassmannian:
            #   U_new = U + δ p̂^T
            #   δ = (cos θ − 1) U p̂ + sin θ r̂
            delta = (cos_t - 1.0) * (U @ p_hat) + sin_t * r_hat  # (M,)
            U = U + np.outer(delta, p_hat)                         # (M, r)

        self._U = U
        self._grouse_M = M