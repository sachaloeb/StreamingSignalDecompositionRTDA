"""Randomized SVD via Halko, Martinsson & Tropp (2011).

Implements Algorithm 4.3 (randomised range finder + deterministic SVD
of the projected matrix) with optional power iteration for improved
accuracy on matrices with slowly decaying singular values.

Reference
---------
Halko, N., Martinsson, P. G., & Tropp, J. A. (2011). Finding structure
with randomness: Probabilistic algorithms for constructing approximate
matrix decompositions.  *SIAM Review*, 53(2), 217–288.
"""

from __future__ import annotations

import numpy as np


def rsvd(
    X: np.ndarray,
    k: int = 10,
    n_oversamples: int = 5,
    n_power_iter: int = 1,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Randomised truncated SVD.

    Parameters
    ----------
    X : np.ndarray
        Input matrix of shape (m, n).
    k : int, optional
        Target rank.  Default 10.
    n_oversamples : int, optional
        Number of oversampling columns for the random projection.
        The sketch uses ``k + n_oversamples`` random vectors.
        Default 5.
    n_power_iter : int, optional
        Number of power-iteration steps to sharpen the spectrum.
        Default 1.
    seed : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    U : np.ndarray
        Left singular vectors, shape (m, k).
    S : np.ndarray
        Singular values, shape (k,).
    Vt : np.ndarray
        Right singular vectors, shape (k, n).
    """
    rng = np.random.default_rng(seed)
    m, n = X.shape
    p = k + n_oversamples
    p = min(p, min(m, n))  # cannot exceed matrix dimensions
    k = min(k, p)

    # Stage A: form an approximate orthonormal basis for the range of X
    Omega = rng.standard_normal((n, p))          # (n, p)
    Y = X @ Omega                                 # (m, p)

    # Power iteration for spectral sharpening
    for _ in range(n_power_iter):
        Y = X @ (X.T @ Y)

    Q, _ = np.linalg.qr(Y)                       # (m, p)

    # Stage B: form the small matrix B and compute its SVD
    B = Q.T @ X                                    # (p, n)
    U_hat, S, Vt = np.linalg.svd(B, full_matrices=False)

    # Recover the approximate left singular vectors of X
    U = Q @ U_hat                                  # (m, p)

    # Truncate to rank k
    return U[:, :k], S[:k], Vt[:k, :]