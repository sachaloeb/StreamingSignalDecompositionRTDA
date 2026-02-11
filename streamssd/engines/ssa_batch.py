"""Batch SSA engine: classical SSA with full SVD per window."""

from typing import Optional

import numpy as np
from scipy.linalg import svd

from streamssd.embed import hankel_embed
from streamssd.engines.base import BaseEngine, DecompositionResult
from streamssd.reconstruct import reconstruct_component


class SSABatchEngine(BaseEngine):
    """Classical SSA engine using batch SVD per window.
    
    For each window:
    1. Build Hankel matrix X(L x K) from window
    2. Compute SVD: X = U S V^T
    3. For i=1..r, reconstruct elementary matrix Xi = s[i] * U[:,i] * V[:,i]^T
    4. Apply diagonal averaging to get component ci of length W
    5. Residual = x_window - sum(ci)
    """
    
    def __init__(self, L: int, r: int):
        """Initialize SSA engine.
        
        Args:
            L: Embedding dimension (Hankel matrix rows).
            r: Number of components to extract.
        """
        if L <= 0:
            raise ValueError(f"L must be > 0, got {L}")
        if r <= 0:
            raise ValueError(f"r must be > 0, got {r}")
        
        self.L = L
        self.r = r
    
    def fit_window(self, x_window: np.ndarray, fs: float, **kwargs) -> DecompositionResult:
        """Decompose window using batch SSA.
        
        Args:
            x_window: 1D array of length W.
            fs: Sampling frequency (not used in SSA, but kept for interface consistency).
            **kwargs: Ignored.
            
        Returns:
            DecompositionResult with r components.
        """
        W = len(x_window)
        
        # Build Hankel matrix
        X = hankel_embed(x_window, self.L)
        L, K = X.shape
        
        # SVD
        U, s, Vt = svd(X, full_matrices=False)
        V = Vt.T  # Convert to column vectors
        
        # Reconstruct r components
        components = []
        for i in range(min(self.r, len(s))):
            comp = reconstruct_component(U[:, i], s[i], V[:, i], W)
            components.append(comp)
        
        # Residual
        reconstructed = sum(components)
        residual = x_window - reconstructed
        
        # Metadata
        meta = {
            "singular_values": s[:self.r].copy(),
            "rank": min(self.r, len(s)),
            "L": self.L,
            "K": K,
        }
        
        return DecompositionResult(
            components=components,
            residual=residual,
            meta=meta,
        )
