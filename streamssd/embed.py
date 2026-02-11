"""Hankel embedding utilities for time series."""

from typing import Optional, Tuple

import numpy as np


def hankel_embed(x: np.ndarray, L: int) -> np.ndarray:
    """Embed a 1D time series into a Hankel matrix.
    
    Args:
        x: 1D array of length W.
        L: Embedding dimension (window length).
        
    Returns:
        Hankel matrix of shape (L, K) where K = W - L + 1.
        The matrix has the form:
        [x[0]   x[1]   ... x[K-1]  ]
        [x[1]   x[2]   ... x[K]    ]
        [ ...   ...   ... ...      ]
        [x[L-1] x[L]   ... x[W-1]  ]
    """
    W = len(x)
    K = W - L + 1
    if K <= 0:
        raise ValueError(f"L={L} must be <= W={W}")
    
    H = np.zeros((L, K), dtype=x.dtype)
    for i in range(L):
        for j in range(K):
            H[i, j] = x[i + j]
    return H


def hankel_embed_fast(x: np.ndarray, L: int) -> np.ndarray:
    """Fast Hankel embedding using stride tricks (alternative implementation).
    
    Args:
        x: 1D array of length W.
        L: Embedding dimension.
        
    Returns:
        Hankel matrix of shape (L, K).
    """
    W = len(x)
    K = W - L + 1
    if K <= 0:
        raise ValueError(f"L={L} must be <= W={W}")
    
    # Use stride tricks for efficiency
    strides = (x.strides[0], x.strides[0])
    shape = (L, K)
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


def get_embedding_params(W: int, L: Optional[int] = None) -> Tuple[int, int]:
    """Get embedding dimension L and K from window length W.
    
    Common choice: L = W // 2 (or W // 3 for longer windows).
    
    Args:
        W: Window length.
        L: Embedding dimension. If None, uses W // 2.
        
    Returns:
        Tuple of (L, K) where K = W - L + 1.
    """
    if L is None:
        L = W // 2
    if L < 1 or L >= W:
        raise ValueError(f"L={L} must be in [1, W-1] for W={W}")
    K = W - L + 1
    return L, K
