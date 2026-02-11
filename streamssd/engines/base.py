"""Base engine interface for signal decomposition."""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class DecompositionResult:
    """Result of signal decomposition.
    
    Attributes:
        components: List of reconstructed time-domain components, each of length W.
        residual: Residual signal after extracting components, length W.
        meta: Dictionary with additional metadata (singular values, ranks, frequencies, etc.).
    """
    components: list[np.ndarray]
    residual: np.ndarray
    meta: dict[str, Any]


class BaseEngine:
    """Base class for signal decomposition engines.
    
    All engines must implement fit_window() to decompose a single window.
    """
    
    def fit_window(self, x_window: np.ndarray, fs: float, **kwargs) -> DecompositionResult:
        """Decompose a single window of signal.
        
        Args:
            x_window: 1D array of length W (window length).
            fs: Sampling frequency in Hz.
            **kwargs: Additional engine-specific parameters.
            
        Returns:
            DecompositionResult with components, residual, and metadata.
        """
        raise NotImplementedError("Subclasses must implement fit_window()")
