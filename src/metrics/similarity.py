"""Similarity metrics for comparing signal components and subspaces.

Provides distance and correlation measures used by the component matcher
and for evaluation of decomposition quality.
"""

from __future__ import annotations

import numpy as np


def d_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Normalised inner-product distance.

    Parameters
    ----------
    x : np.ndarray
        First signal vector.
    y : np.ndarray
        Second signal vector.

    Returns
    -------
    float
        Distance in [0, 1].  0 = perfectly correlated (or
        anti-correlated), 1 = orthogonal or degenerate input.

    Notes
    -----
    d(x, y) = 1 - |<x, y>| / (||x|| * ||y||)
    as defined in Harmouche et al. (2017, IEEE TSP).
    """
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    if norm_x < 1e-12 or norm_y < 1e-12:
        return 1.0
    return 1.0 - float(np.abs(np.dot(x, y)) / (norm_x * norm_y))


def d_freq(
    g1: np.ndarray,
    g2: np.ndarray,
    fs: float = 1.0,
) -> float:
    """Absolute difference of dominant frequencies.

    Parameters
    ----------
    g1 : np.ndarray
        First component signal.
    g2 : np.ndarray
        Second component signal.
    fs : float, optional
        Sampling frequency in Hz.  Default 1.0.

    Returns
    -------
    float
        |f_max(g1) - f_max(g2)| in Hz.
    """
    freqs1 = np.fft.rfftfreq(len(g1), d=1.0 / fs)
    freqs2 = np.fft.rfftfreq(len(g2), d=1.0 / fs)
    mag1 = np.abs(np.fft.rfft(g1))
    mag2 = np.abs(np.fft.rfft(g2))
    f1 = float(freqs1[np.argmax(mag1)])
    f2 = float(freqs2[np.argmax(mag2)])
    return abs(f1 - f2)


def subspace_angle(
    U_prev: np.ndarray,
    U_curr: np.ndarray,
) -> float:
    """Principal angle between two subspaces.

    Parameters
    ----------
    U_prev : np.ndarray
        Column-orthonormal basis of the previous subspace (n x k1).
    U_curr : np.ndarray
        Column-orthonormal basis of the current subspace (n x k2).

    Returns
    -------
    float
        Largest principal angle in radians.

    Notes
    -----
    Computed as arccos(sigma_max(U_prev^T @ U_curr)), where sigma_max
    is the largest singular value of the cross-product matrix.
    """
    M = U_prev.T @ U_curr
    sigma = np.linalg.svd(M, compute_uv=False)
    cos_theta = np.clip(sigma[0], -1.0, 1.0)
    return float(np.arccos(cos_theta))


def w_correlation(
    x: np.ndarray,
    y: np.ndarray,
    L: int,
) -> float:
    """Weighted correlation (w-correlation) per Golyandina.

    Parameters
    ----------
    x : np.ndarray
        First reconstructed component of length N.
    y : np.ndarray
        Second reconstructed component of length N.
    L : int
        Window (embedding) length used for the trajectory matrix.

    Returns
    -------
    float
        Weighted correlation in [0, 1].

    Notes
    -----
    Weights: w_i = min(i+1, L, N-L+1, N-i) for i = 0 .. N-1.
    The weights are normalised to sum to 1 before computing the
    inner product.
    """
    N = len(x)
    K = N - L + 1
    idx = np.arange(N)
    w = np.minimum.reduce([idx + 1, np.full(N, L),
                           np.full(N, K), N - idx])
    w = w.astype(np.float64)
    w_sum = w.sum()
    if w_sum < 1e-12:
        return 0.0
    w /= w_sum

    wx = np.sqrt(np.dot(w, x * x))
    wy = np.sqrt(np.dot(w, y * y))
    if wx < 1e-12 or wy < 1e-12:
        return 0.0
    return float(np.abs(np.dot(w, x * y)) / (wx * wy))
