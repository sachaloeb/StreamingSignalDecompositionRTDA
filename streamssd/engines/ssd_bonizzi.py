"""SSD (Singular Spectrum Decomposition) engine based on Bonizzi 2014.

SSD is an iterative algorithm that extracts narrowband oscillatory components
one at a time using SSA-like embedding and a selection rule for dominant modes.
"""

from typing import Optional

import numpy as np
from scipy.linalg import svd
from scipy.fft import fft, fftfreq

from streamssd.embed import hankel_embed
from streamssd.engines.base import BaseEngine, DecompositionResult
from streamssd.reconstruct import reconstruct_component


class SSDBonizziEngine(BaseEngine):
    """SSD engine implementing iterative component extraction.
    
    Algorithm:
    1. Start with residual = x_window
    2. For m in 1..M (M components):
       a) Embed residual into Hankel X
       b) SVD of X
       c) Identify candidate components and select the one with highest
          narrowbandedness (peak_power / total_power)
       d) Reconstruct selected component
       e) Subtract from residual (deflation)
       f) Store component
    3. Final residual is the remainder
    
    This is a practical operationalization of Bonizzi 2014 SSD. The selection
    criterion uses narrowbandedness as a proxy for oscillatory mode quality.
    """
    
    def __init__(
        self,
        L: int,
        M: int,
        fmin: Optional[float] = None,
        fmax: Optional[float] = None,
    ):
        """Initialize SSD engine.
        
        Args:
            L: Embedding dimension (Hankel matrix rows).
            M: Number of components to extract.
            fmin: Minimum frequency for component selection (Hz, optional).
            fmax: Maximum frequency for component selection (Hz, optional).
        """
        if L <= 0:
            raise ValueError(f"L must be > 0, got {L}")
        if M <= 0:
            raise ValueError(f"M must be > 0, got {M}")
        
        self.L = L
        self.M = M
        self.fmin = fmin
        self.fmax = fmax
    
    def _compute_narrowbandedness(
        self,
        comp: np.ndarray,
        fs: float,
    ) -> tuple[float, float]:
        """Compute narrowbandedness score for a component.
        
        Narrowbandedness = peak_power / total_power (higher = more narrowband).
        Also returns the dominant frequency.
        
        Args:
            comp: 1D component array.
            fs: Sampling frequency.
            
        Returns:
            Tuple of (score, dominant_freq).
        """
        # FFT
        fft_vals = fft(comp)
        freqs = fftfreq(len(comp), 1 / fs)
        
        # Power spectrum
        power = np.abs(fft_vals) ** 2
        
        # Find peak (exclude DC and negative frequencies)
        positive_mask = freqs > 0
        if not np.any(positive_mask):
            return 0.0, 0.0
        
        positive_freqs = freqs[positive_mask]
        positive_power = power[positive_mask]
        
        # Apply frequency filter if specified
        if self.fmin is not None:
            positive_power[positive_freqs < self.fmin] = 0
        if self.fmax is not None:
            positive_power[positive_freqs > self.fmax] = 0
        
        if np.sum(positive_power) == 0:
            return 0.0, 0.0
        
        peak_idx = np.argmax(positive_power)
        peak_power = positive_power[peak_idx]
        dominant_freq = positive_freqs[peak_idx]
        
        total_power = np.sum(positive_power)
        score = peak_power / total_power if total_power > 0 else 0.0
        
        return score, dominant_freq
    
    def _select_best_component(
        self,
        U: np.ndarray,
        s: np.ndarray,
        V: np.ndarray,
        W: int,
        fs: float,
    ) -> tuple[int, np.ndarray, float]:
        """Select the best component from SVD factors.
        
        Evaluates first p singular vectors (p = min(10, rank)) and selects
        the one with highest narrowbandedness score.
        
        Args:
            U: Left singular vectors (L x rank).
            s: Singular values (rank,).
            V: Right singular vectors (K x rank).
            W: Window length.
            fs: Sampling frequency.
            
        Returns:
            Tuple of (selected_idx, component, dominant_freq).
        """
        rank = len(s)
        p = min(10, rank)
        
        best_score = -1.0
        best_idx = 0
        best_comp = None
        best_freq = 0.0
        
        for i in range(p):
            # Reconstruct component
            comp = reconstruct_component(U[:, i], s[i], V[:, i], W)
            
            # Compute narrowbandedness
            score, freq = self._compute_narrowbandedness(comp, fs)
            
            if score > best_score:
                best_score = score
                best_idx = i
                best_comp = comp
                best_freq = freq
        
        return best_idx, best_comp, best_freq
    
    def fit_window(self, x_window: np.ndarray, fs: float, **kwargs) -> DecompositionResult:
        """Decompose window using iterative SSD.
        
        Args:
            x_window: 1D array of length W.
            fs: Sampling frequency in Hz.
            **kwargs: Ignored.
            
        Returns:
            DecompositionResult with M components.
        """
        W = len(x_window)
        residual = x_window.copy()
        components = []
        meta_components = []
        
        # Iteratively extract M components
        for m in range(self.M):
            # Embed residual
            X = hankel_embed(residual, self.L)
            L, K = X.shape
            
            # SVD
            U, s, Vt = svd(X, full_matrices=False)
            V = Vt.T
            
            if len(s) == 0:
                # No more signal to extract
                break
            
            # Select best component
            selected_idx, comp, dominant_freq = self._select_best_component(
                U, s, V, W, fs
            )
            
            if comp is None:
                break
            
            # Store component
            components.append(comp)
            meta_components.append({
                "singular_value": s[selected_idx],
                "dominant_freq": dominant_freq,
                "iteration": m,
            })
            
            # Deflate: subtract from residual
            residual = residual - comp
        
        # Final residual
        reconstructed = sum(components) if components else np.zeros(W)
        final_residual = x_window - reconstructed
        
        # Metadata
        meta = {
            "num_components": len(components),
            "components": meta_components,
            "L": self.L,
            "K": K if self.M > 0 else 0,
        }
        
        return DecompositionResult(
            components=components,
            residual=final_residual,
            meta=meta,
        )
