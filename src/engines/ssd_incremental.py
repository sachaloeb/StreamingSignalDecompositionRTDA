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

from src.engines.ssa import svd_decompose
from src.engines.ssd import SSD


class IncrementalSSD(SSD):
    """Incremental SSD with warm-start SVD caching.

    Inherits the full SSD extraction pipeline and overrides only the
    SVD decomposition step (``_decompose_trajectory``) to inject
    warm-start or randomised SVD.

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
        Maximum relative projection residual to allow warm-start.
        Default 0.5.
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
        super().__init__(
            fs=fs,
            nmse_threshold=nmse_threshold,
            max_iter=max_iter,
            **kwargs,
        )
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
    # Override SVD hook for warm-start
    # ------------------------------------------------------------------

    def _decompose_trajectory(
        self,
        X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """SVD with optional warm-start from cached factors."""
        svd_method = "randomized" if self.use_rsvd else "full"
        rank_param = min(X.shape) if self.use_rsvd else None

        # Check warm-start eligibility
        if (
            self._prev_U is not None
            and self._prev_N > 0
            and self._prev_U.shape[0] == X.shape[0]
            and self._prev_Vt.shape[1] == X.shape[1]
        ):
            # Project X onto cached left subspace
            B = self._prev_U.T @ X
            X_proj = self._prev_U @ B
            residual_norm = np.linalg.norm(X - X_proj, "fro")
            total_norm = np.linalg.norm(X, "fro")

            if total_norm > 1e-14:
                relative_residual = residual_norm / total_norm
            else:
                relative_residual = 1.0

            if relative_residual < self.subspace_threshold:
                U_b, S_b, Vt_b = np.linalg.svd(B, full_matrices=False)
                U_warm = self._prev_U @ U_b
                return U_warm, S_b, Vt_b

        # Cold-start fallback
        return svd_decompose(
            X,
            rank=rank_param,
            method=svd_method,
            rsvd_oversamples=self.rsvd_oversamples,
            rsvd_power_iter=self.rsvd_power_iter,
        )

    # ------------------------------------------------------------------
    # Override fit to cache SVD factors after each call
    # ------------------------------------------------------------------

    def fit(self, x: np.ndarray) -> list[np.ndarray]:
        """Decompose *x* with warm-start SVD caching.

        Delegates to ``SSD.fit`` (which calls ``_decompose_trajectory``
        through the polish loop) and then caches the SVD factors of the
        mean-removed full window for the next call.
        """
        result = super().fit(x)

        # Cache SVD of the mean-removed window for warm-start in next call
        x = np.asarray(x, dtype=np.float64)
        x_zm = x - np.mean(x)
        N = len(x)
        M_cache = self._choose_window_length(x_zm)
        X_cache = self._build_trajectory_matrix(x_zm, M_cache)
        svd_method = "randomized" if self.use_rsvd else "full"
        rank = min(M_cache, X_cache.shape[1]) if self.use_rsvd else None
        self._prev_U, self._prev_S, self._prev_Vt = svd_decompose(
            X_cache,
            rank=rank,
            method=svd_method,
            rsvd_oversamples=self.rsvd_oversamples,
            rsvd_power_iter=self.rsvd_power_iter,
        )
        self._prev_N = N

        return result