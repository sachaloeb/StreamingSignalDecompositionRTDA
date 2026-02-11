"""Diagonal averaging (hankelization) for reconstructing time series from matrices."""

import numpy as np


def diagonal_average(X: np.ndarray, W: int) -> np.ndarray:
    """Reconstruct time series from matrix using diagonal averaging.
    
    This is the inverse operation of Hankel embedding. Given a matrix X
    of shape (L, K), reconstructs a 1D array of length W = L + K - 1
    by averaging along anti-diagonals.
    
    Args:
        X: Matrix of shape (L, K) to reconstruct from.
        W: Desired output length. Must satisfy W = L + K - 1.
        
    Returns:
        1D array of length W.
    """
    L, K = X.shape
    if W != L + K - 1:
        raise ValueError(f"W={W} must equal L+K-1={L+K-1}")
    
    result = np.zeros(W)
    counts = np.zeros(W)
    
    # Average along anti-diagonals
    for i in range(L):
        for j in range(K):
            idx = i + j  # Index in output array
            result[idx] += X[i, j]
            counts[idx] += 1
    
    # Normalize by counts
    result /= counts
    
    return result


def reconstruct_component(U: np.ndarray, s: float, V: np.ndarray, W: int) -> np.ndarray:
    """Reconstruct a single component from SVD factors.
    
    Given U[:, i], s[i], V[:, i] from SVD, reconstructs the time-domain
    component by forming the elementary matrix and applying diagonal averaging.
    
    Args:
        U: Left singular vector (1D array of length L).
        s: Singular value (scalar).
        V: Right singular vector (1D array of length K).
        W: Desired output length.
        
    Returns:
        Reconstructed component of length W.
    """
    # Form elementary matrix: s * U * V^T
    L = len(U)
    K = len(V)
    X = s * np.outer(U, V)
    
    # Diagonal average to get time series
    return diagonal_average(X, W)
