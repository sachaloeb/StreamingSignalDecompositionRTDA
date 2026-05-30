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

Algorithm (fully observed case, Ω = all indices)
-------------------------------------------------
For each column v = X[:, j] of the trajectory matrix:

    w      = U^T v                         least-squares weight (= U^T v for full data)
    p      = U w                           projection onto subspace
    r_vec  = v − p                         residual orthogonal to U
    r̂     = r_vec / ‖r_vec‖               unit residual
    p̂     = w / ‖w‖                       unit weight
    θ      = step_size                     fixed geodesic step (radians)
    δ      = (cos θ − 1) U p̂ + sin θ r̂  rank-1 tangent direction
    U_new  = U + δ p̂^T                    Grassmannian geodesic step

Recovering (U, S, Vt) from the tracked subspace
------------------------------------------------
Given the tracked orthonormal U (M × r), the rank-r approximation of X
is Û = U (U^T X).  Let B = U^T X (shape r × K, cheap to compute).  The
thin SVD of B gives B = U_b Σ Vt_b, so:

    X ≈ U U_b Σ Vt_b  →  U_out = U @ U_b,  S = diag(Σ),  Vt = Vt_b

The previous implementation used *row norms* of B as singular-value
estimates, which is only exact when the rows of B are orthogonal (i.e.
at convergence).  The thin SVD of B is always correct and is cheap
because B is only r × K.

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
        """Return (U, S, Vt) for X using the current GROUSE subspace.

        If a cached U is available and the row dimension matches, the
        tracked subspace is used to compute the full (U, S, Vt) via a
        thin SVD of B = U^T X (shape r × K, cheap).  This correctly
        recovers singular values and right singular vectors — the
        previous row-norm approach was only exact at convergence.

        Falls back to a full SVD cold start when U is uninitialised or
        the trajectory matrix row dimension M has changed.

        Note: U is NOT updated here.  Updates happen in ``fit`` via
        ``_advance_grouse_state`` to prevent residual trajectory matrices
        from corrupting the subspace estimate.

        Parameters
        ----------
        X : np.ndarray
            Trajectory matrix (M, K).

        Returns
        -------
        U_out : np.ndarray  (M, r)
        S     : np.ndarray  (r,)
        Vt    : np.ndarray  (r, K)
        """
        M, K = X.shape
        r = min(self.rank, M, K)

        if self._U is None or self._grouse_M != M:
            return self._cold_start(X, M, r)

        U = self._U  # (M, rank)  — frozen, orthonormal

        # --- Project X onto the tracked subspace ---
        # B = U^T X has shape (rank, K).  The rank-r approximation of X
        # is U B, and its singular triplets come from the thin SVD of B.
        B = U.T @ X                                            # (rank, K)

        # --- Thin SVD of B (cheap: rank × K, rank << min(M, K)) ---
        # B = U_b Σ Vt_b  →  X ≈ (U U_b) Σ Vt_b
        # U_b: (rank, rank),  S: (rank,),  Vt_b: (rank, K)
        U_b, S, Vt_b = np.linalg.svd(B, full_matrices=False)

        # Rotate the tracked U to align with the true singular directions.
        U_out = U @ U_b   # (M, rank)

        # Sort by descending singular value (svd already returns desc, but
        # defensive sort in case of ties or numerical noise).
        order = np.argsort(S)[::-1]
        return U_out[:, order], S[order], Vt_b[order, :]

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
        """Full SVD cold start; does not modify tracker state.

        The tracker state is initialised lazily in
        ``_advance_grouse_state``, not here, to maintain the invariant
        that ``_decompose_trajectory`` never mutates state.
        """
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
        # This is Algorithm 4 of Balzano et al. (2013 review) with
        # fully observed data (Ω = all indices).
        theta = self.step_size
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        for j in range(K):
            v = X[:, j]                # (M,)  — current column

            # w = U^T v  (least-squares weight for fully observed case)
            w = U.T @ v                # (r,)
            p = U @ w                  # (M,)  projection onto subspace
            r_vec = v - p              # (M,)  residual orthogonal to U

            rho_r = float(np.linalg.norm(r_vec))
            rho_w = float(np.linalg.norm(w))

            if rho_r < 1e-14 or rho_w < 1e-14:
                continue               # v already in subspace or trivial

            r_hat = r_vec / rho_r      # (M,) unit residual direction
            w_hat = w / rho_w          # (r,) unit weight / projection

            # Grassmannian geodesic rank-1 update (Balzano et al. Eq. 23):
            #   U_new = U + δ w_hat^T
            #   δ = (cos θ − 1) U w_hat + sin θ r_hat
            delta = (cos_t - 1.0) * (U @ w_hat) + sin_t * r_hat  # (M,)
            U = U + np.outer(delta, w_hat)                         # (M, r)

        self._U = U
        self._grouse_M = M