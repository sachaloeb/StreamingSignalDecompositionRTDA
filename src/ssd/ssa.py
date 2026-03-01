"""Base Singular Spectrum Analysis (SSA) and autoSSA with hierarchical
grouping.

Provides the fundamental building blocks — embedding, SVD decomposition,
diagonal averaging, and an automated grouping step based on
agglomerative clustering with the d_corr distance.
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

from src.metrics.similarity import d_corr


def build_trajectory_matrix(
    x: np.ndarray,
    L: int,
) -> np.ndarray:
    """Build the standard (non-wrapped) Hankel trajectory matrix.

    Parameters
    ----------
    x : np.ndarray
        Input signal of length N.
    L : int
        Window (embedding) length.  Must satisfy 2 <= L <= N.

    Returns
    -------
    np.ndarray
        Trajectory matrix of shape (L, K) where K = N - L + 1.
    """
    N = len(x)
    K = N - L + 1
    X = np.empty((L, K), dtype=np.float64)
    for i in range(L):
        X[i, :] = x[i: i + K]
    return X


def diagonal_averaging(X: np.ndarray) -> np.ndarray:
    """Reconstruct a 1-D signal via anti-diagonal averaging.

    Parameters
    ----------
    X : np.ndarray
        Matrix of shape (L, K).

    Returns
    -------
    np.ndarray
        Reconstructed signal of length L + K - 1.
    """
    L, K = X.shape
    N = L + K - 1
    y = np.zeros(N, dtype=np.float64)
    counts = np.zeros(N, dtype=np.float64)
    for i in range(L):
        for j in range(K):
            y[i + j] += X[i, j]
            counts[i + j] += 1.0
    counts = np.maximum(counts, 1e-12)
    return y / counts


def svd_decompose(
    X: np.ndarray,
    rank: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Singular value decomposition with optional rank truncation.

    Parameters
    ----------
    X : np.ndarray
        Matrix to decompose (L x K).
    rank : int or None, optional
        If given, retain only the top-*rank* singular triplets.

    Returns
    -------
    U : np.ndarray
        Left singular vectors (L x r).
    S : np.ndarray
        Singular values (r,).
    Vt : np.ndarray
        Right singular vectors (r x K).

    Notes
    -----
    Uses ``np.linalg.svd`` with ``full_matrices=False``.
    """
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    if rank is not None:
        rank = min(rank, len(S))
        U = U[:, :rank]
        S = S[:rank]
        Vt = Vt[:rank, :]
    return U, S, Vt


def auto_ssa(
    x: np.ndarray,
    r: int,
    L: int,
) -> list[np.ndarray]:
    """Automated SSA with hierarchical grouping into *r* components.

    Parameters
    ----------
    x : np.ndarray
        Input signal of length N.
    r : int
        Desired number of grouped components.
    L : int
        Window (embedding) length.

    Returns
    -------
    list[np.ndarray]
        List of *r* reconstructed component arrays, each of length N.

    Notes
    -----
    1. Embed *x* into a trajectory matrix.
    2. Compute full SVD → elementary reconstructed components.
    3. Compute pairwise d_corr distance matrix.
    4. Apply agglomerative (complete linkage) clustering to merge
       elementary components into *r* groups.
    5. Diagonal-average each grouped matrix to obtain final components.
    """
    N = len(x)
    X = build_trajectory_matrix(x, L)
    U, S, Vt = svd_decompose(X)

    n_et = len(S)
    if n_et <= r:
        elementary = []
        for k in range(n_et):
            Xk = S[k] * np.outer(U[:, k], Vt[k, :])
            elementary.append(diagonal_averaging(Xk))
        return elementary

    elementary = []
    for k in range(n_et):
        Xk = S[k] * np.outer(U[:, k], Vt[k, :])
        elementary.append(diagonal_averaging(Xk))

    dist_vec = np.zeros(n_et * (n_et - 1) // 2, dtype=np.float64)
    idx = 0
    for i in range(n_et):
        for j in range(i + 1, n_et):
            dist_vec[idx] = d_corr(elementary[i], elementary[j])
            idx += 1

    dist_vec = np.clip(dist_vec, 0.0, 1.0)
    Z = linkage(dist_vec, method="complete")
    labels = fcluster(Z, t=r, criterion="maxclust")

    groups: list[np.ndarray] = []
    for g in range(1, r + 1):
        members = [k for k in range(n_et) if labels[k] == g]
        if len(members) == 0:
            groups.append(np.zeros(N, dtype=np.float64))
            continue
        X_group = np.zeros_like(X)
        for k in members:
            X_group += S[k] * np.outer(U[:, k], Vt[k, :])
        groups.append(diagonal_averaging(X_group))

    return groups
