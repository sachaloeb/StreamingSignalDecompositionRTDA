"""SSD with Randomised SVD decomposition.

Replaces the full SVD in every ``_decompose_trajectory`` call with the
randomised SVD algorithm of Halko, Martinsson & Tropp (2011).  No state
is maintained between windows; each call is independent.

Reference
---------
Halko, N., Martinsson, P. G., & Tropp, J. A. (2011). Finding structure
    with randomness: Probabilistic algorithms for constructing approximate
    matrix decompositions.  SIAM Review, 53(2), 217–288.
"""

from __future__ import annotations

import numpy as np

from src.engines.rsvd import rsvd
from src.engines.ssd import SSD


class RsvdSSD(SSD):
    """SSD with randomised SVD (Halko et al. 2011) at every trajectory decomposition.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz.
    n_oversamples : int, optional
        Oversampling columns added to the random sketch.  Default 10.
    n_power_iter : int, optional
        Power-iteration refinement steps.  Default 2.
    nmse_threshold : float, optional
        NMSE stopping criterion.  Default 0.01.
    max_iter : int, optional
        Maximum SSD extraction iterations.  Default 20.
    """

    def __init__(
        self,
        fs: float,
        n_oversamples: int = 10,
        n_power_iter: int = 2,
        **kwargs: object,
    ) -> None:
        super().__init__(fs=fs, **kwargs)
        self.n_oversamples = n_oversamples
        self.n_power_iter = n_power_iter

    def _decompose_trajectory(
        self,
        X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Randomised SVD of the trajectory matrix (Halko et al. 2011)."""
        k = min(X.shape)
        return rsvd(
            X,
            k=k,
            n_oversamples=self.n_oversamples,
            n_power_iter=self.n_power_iter,
        )