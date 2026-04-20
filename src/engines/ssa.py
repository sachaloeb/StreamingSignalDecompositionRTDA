"""Base Singular Spectrum Analysis (SSA) and autoSSA with hierarchical
grouping.

Provides the fundamental building blocks — embedding, SVD decomposition,
diagonal averaging, and an automated grouping step based on
agglomerative clustering with the d_corr distance.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy.cluster.hierarchy import fcluster, linkage

from src.engines.base import DecompositionEngine
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
    x = np.asarray(x, dtype=np.float64)
    itemsize = x.strides[0]
    K = len(x) - L + 1
    X = as_strided(x, shape=(L, K), strides=(itemsize, itemsize))
    return np.ascontiguousarray(X, dtype=np.float64)


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

    # Vectorized scatter-add via pre-computed anti-diagonal indices.
    i_idx = np.arange(L, dtype=np.intp)[:, None]   # (L, 1)
    j_idx = np.arange(K, dtype=np.intp)[None, :]   # (1, K)
    diag_idx = (i_idx + j_idx).ravel()              # (L*K,)
    y = np.zeros(N, dtype=np.float64)
    np.add.at(y, diag_idx, X.ravel())

    # Analytic counts — no loop, no second np.add.at.
    # For a standard (L, K) Hankel matrix with N = L + K - 1:
    #   n in [0,           min(L,K)-1]:   counts[n] = n + 1
    #   n in [min(L,K),    max(L,K)-1]:   counts[n] = min(L, K)
    #   n in [max(L,K),    N-1]:          counts[n] = N - n
    counts = np.empty(N, dtype=np.float64)
    lo, hi = min(L, K), max(L, K)
    n = np.arange(N, dtype=np.float64)
    counts = np.where(n < lo, n + 1.0,
                      np.where(n < hi, float(lo), N - n))

    return y / counts


def svd_decompose(
    X: np.ndarray,
    rank: int | None = None,
    method: str = "full",
    rsvd_oversamples: int = 5,
    rsvd_power_iter: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Singular value decomposition with optional rank truncation.

    Parameters
    ----------
    X : np.ndarray
        Matrix to decompose (L x K).
    rank : int or None, optional
        If given, retain only the top-*rank* singular triplets.
    method : str, optional
        ``"full"`` (default) uses ``np.linalg.svd``.
        ``"randomized"`` uses the randomised SVD from
        :func:`src.engines.rsvd.rsvd`.  Requires *rank* to be set.
    rsvd_oversamples : int, optional
        Oversampling parameter for randomised SVD.  Default 5.
    rsvd_power_iter : int, optional
        Power iteration steps for randomised SVD.  Default 1.

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
    When ``method="full"``, uses ``np.linalg.svd`` with
    ``full_matrices=False``.
    """
    if method == "randomized":
        from src.engines.rsvd import rsvd
        k = rank if rank is not None else min(X.shape)
        return rsvd(
            X, k=k,
            n_oversamples=rsvd_oversamples,
            n_power_iter=rsvd_power_iter,
        )

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


class SSA(DecompositionEngine):
    """Automated SSA decomposition engine.

    Thin wrapper around :func:`auto_ssa` exposing the
    :class:`DecompositionEngine` interface.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz (kept for interface uniformity; not
        used by plain SSA).
    n_components : int, optional
        Number of grouped components to return. Default 2.
    window_length : int or None, optional
        Embedding window length L. If ``None`` (default), uses N // 3.
    """

    def __init__(
        self,
        fs: float,
        n_components: int = 2,
        window_length: int | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(fs=fs, **kwargs)
        self.n_components = n_components
        self.window_length = window_length

    def fit(self, x: np.ndarray) -> list[np.ndarray]:
        x = np.asarray(x, dtype=np.float64)
        L = self.window_length if self.window_length is not None else max(2, len(x) // 3)
        L = int(min(L, len(x) - 1))
        return auto_ssa(x, r=self.n_components, L=L)
