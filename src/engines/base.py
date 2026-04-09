"""Abstract base class for decomposition engines (Strategy pattern)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class DecompositionEngine(ABC):
    """Strategy interface for signal decomposition algorithms.

    Concrete engines (SSD, SSA, ...) implement :meth:`fit` which
    decomposes a 1-D signal into a list of component arrays.
    """

    def __init__(self, fs: float, **kwargs: object) -> None:
        self.fs = fs

    @abstractmethod
    def fit(self, x: np.ndarray) -> list[np.ndarray]:
        """Decompose *x* into constituent components.

        Parameters
        ----------
        x : np.ndarray
            Input signal of length N.

        Returns
        -------
        list[np.ndarray]
            Component arrays. For iterative engines the final element
            is typically the residual.
        """