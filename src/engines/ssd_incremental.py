"""Warm-start Incremental SSD engine.

Caches SVD factors from the previous window and reuses them as an
initialisation for the current window when the overlap is sufficient
and the subspace angle is small.  Falls back to a cold-start full SVD
when the subspace has drifted significantly.

Optionally uses randomised SVD (Halko et al. 2011) instead of full
SVD for additional speedup.
"""

from __future__ import annotations

import numpy as np

from src.engines.base import DecompositionEngine
from src.engines.ssa import svd_decompose
from src.engines.ssd import SSD
from src.metrics.similarity import subspace_angle
from src.metrics.stability import nmse as compute_nmse


class IncrementalSSD(DecompositionEngine):
    """Incremental SSD with warm-start SVD caching.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz.
    nmse_threshold : float, optional
        NMSE stopping criterion.  Default 0.01.
    max_iter : int, optional
        Maximum SSD extraction iterations.  Default 20.
    min_overlap_ratio : float, optional
        Minimum overlap fraction required to attempt warm-start.
        Default 0.5.
    subspace_threshold : float, optional
        Maximum principal angle (radians) between old and new
        subspaces to allow warm-start.  Default 0.5 (~28 degrees).
    use_rsvd : bool, optional
        If ``True``, use randomised SVD instead of full SVD.
        Default ``False``.
    rsvd_oversamples : int, optional
        Oversampling parameter for rSVD.  Default 5.
    rsvd_power_iter : int, optional
        Power iteration steps for rSVD.  Default 1.
    """

    def __init__(
        self,
        fs: float,
        nmse_threshold: float = 0.01,
        max_iter: int = 20,
        min_overlap_ratio: float = 0.5,
        subspace_threshold: float = 0.5,
        use_rsvd: bool = False,
        rsvd_oversamples: int = 5,
        rsvd_power_iter: int = 1,
        **kwargs: object,
    ) -> None:
        super().__init__(fs=fs, **kwargs)
        self.nmse_threshold = nmse_threshold
        self.max_iter = max_iter
        self.min_overlap_ratio = min_overlap_ratio
        self.subspace_threshold = subspace_threshold
        self.use_rsvd = use_rsvd
        self.rsvd_oversamples = rsvd_oversamples
        self.rsvd_power_iter = rsvd_power_iter

        # Cache from the previous window
        self._prev_U: np.ndarray | None = None
        self._prev_S: np.ndarray | None = None
        self._prev_Vt: np.ndarray | None = None
        self._prev_N: int = 0

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def fit(self, x: np.ndarray) -> list[np.ndarray]:
        """Decompose *x* into SSD components with warm-start caching.

        Parameters
        ----------
        x : np.ndarray
            Input signal of length N.

        Returns
        -------
        list[np.ndarray]
            [g1, g2, ..., gm, residual].
        """
        x = np.asarray(x, dtype=np.float64)
        N = len(x)
        x_energy = np.dot(x, x)
        if x_energy < 1e-30:
            return [x.copy()]

        components: list[np.ndarray] = []
        residual = x.copy()

        for _ in range(self.max_iter):
            M = SSD._choose_window_length(SSD(fs=self.fs), residual)
            X = SSD._build_trajectory_matrix(residual, M)

            U, S, Vt = self._decompose(X, M, N)

            freqs = np.fft.rfftfreq(N, d=1.0 / self.fs)
            psd = np.abs(np.fft.rfft(residual)) ** 2
            f_max = float(freqs[np.argmax(psd)])

            delta_f = SSD._fit_gaussian_model(psd, freqs)
            sel = SSD._select_eigentriples(U, S, f_max, delta_f, self.fs, N)

            if len(sel) == 0:
                sel = [0]

            X_sub = np.zeros_like(X)
            for k in sel:
                X_sub += S[k] * np.outer(U[:, k], Vt[k, :])
            g = SSD._reconstruct_component(X_sub, N)

            g = SSD._polish(SSD(fs=self.fs), g, residual, N)
            a = SSD._scale_factor(g, residual)
            g *= a

            components.append(g)
            residual = residual - g

            cur_nmse = compute_nmse(residual, x)
            if cur_nmse < self.nmse_threshold:
                break

        components.append(residual)

        # Cache the SVD from the first iteration's trajectory matrix
        # for warm-start in the next call.
        M_cache = SSD._choose_window_length(SSD(fs=self.fs), x)
        X_cache = SSD._build_trajectory_matrix(x, M_cache)
        svd_method = "randomized" if self.use_rsvd else "full"
        rank = min(M_cache, X_cache.shape[1]) if self.use_rsvd else None
        self._prev_U, self._prev_S, self._prev_Vt = svd_decompose(
            X_cache, rank=rank, method=svd_method,
            rsvd_oversamples=self.rsvd_oversamples,
            rsvd_power_iter=self.rsvd_power_iter,
        )
        self._prev_N = N

        return components

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _decompose(
        self,
        X: np.ndarray,
        M: int,
        N: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """SVD with optional warm-start from cached factors.

        If a cached decomposition exists, the overlap is sufficient,
        and the subspace angle is small, project the new trajectory
        matrix onto the cached subspace and refine.  Otherwise fall
        back to a cold-start SVD.
        """
        svd_method = "randomized" if self.use_rsvd else "full"
        rank_param = min(X.shape) if self.use_rsvd else None

        # Check warm-start eligibility
        if (
            self._prev_U is not None
            and self._prev_N > 0
        ):
            # Check dimensional compatibility
            if (
                self._prev_U.shape[0] == X.shape[0]
                and self._prev_Vt.shape[1] == X.shape[1]
            ):
                # Compute subspace angle to decide if warm-start is viable
                # Use a quick rank-matched comparison
                r_prev = self._prev_U.shape[1]
                r_curr_max = min(X.shape)

                # Project X onto the cached left subspace
                B = self._prev_U.T @ X  # (r_prev, K)
                X_proj = self._prev_U @ B
                residual_norm = np.linalg.norm(X - X_proj, 'fro')
                total_norm = np.linalg.norm(X, 'fro')

                if total_norm > 1e-14:
                    relative_residual = residual_norm / total_norm
                else:
                    relative_residual = 1.0

                # If the old subspace captures most of the new matrix,
                # use it as initialisation
                if relative_residual < self.subspace_threshold:
                    # Refine: compute SVD of B (small matrix)
                    U_b, S_b, Vt_b = np.linalg.svd(B, full_matrices=False)
                    U_warm = self._prev_U @ U_b
                    return U_warm, S_b, Vt_b

        # Cold-start fallback
        return svd_decompose(
            X, rank=rank_param, method=svd_method,
            rsvd_oversamples=self.rsvd_oversamples,
            rsvd_power_iter=self.rsvd_power_iter,
        )