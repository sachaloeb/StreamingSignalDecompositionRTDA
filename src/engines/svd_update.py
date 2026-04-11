"""Rank-1 SVD update for USSA (Unified Singular Spectrum Analysis).

Implements the O(k^2) Brand (2003) rank-1 update applied to the Hankel
trajectory matrix, as described in:

    Brand, M. (2003). Fast online SVD revisions for lightweight recommender
    systems. *Proc. SIAM International Conference on Data Mining* (pp. 37–46).

    Saeed, M., & Alty, S. R. (2020). USSA: A unified singular spectrum
    analysis framework with application to real-time data. In *Proc. IEEE
    ICASSP 2020* (pp. 4837–4841).

The sliding Hankel matrix update proceeds as two rank-1 modifications:
1. **Downdate**: remove the contribution of the departing (top) row.
2. **Update**: add the contribution of the arriving (bottom) row.

A periodic full SVD reset bounds the accumulation error that arises
from repeated rank-1 modifications.
"""

from __future__ import annotations

import numpy as np


class RankOneUpdater:
    """Incremental rank-1 SVD updater for streaming SSA.

    Parameters
    ----------
    U : np.ndarray
        Left singular vectors (L x r).
    S : np.ndarray
        Singular values (r,).
    Vt : np.ndarray
        Right singular vectors (r x K).
    refresh_every : int, optional
        Perform a full SVD reset every *refresh_every* updates to
        bound accumulation error.  Default 2000.
    """

    def __init__(
        self,
        U: np.ndarray,
        S: np.ndarray,
        Vt: np.ndarray,
        refresh_every: int = 2000,
    ) -> None:
        self.U = U.copy().astype(np.float64)
        self.S = S.copy().astype(np.float64)
        self.Vt = Vt.copy().astype(np.float64)
        self.refresh_every = refresh_every
        self._step = 0

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def update(
        self,
        a: np.ndarray,
        b: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform a single rank-1 update: X_new = X_old + a @ b.T.

        Applies Brand's (2003) rank-1 SVD update to incorporate the
        outer product a @ b.T into the current factorisation.

        Parameters
        ----------
        a : np.ndarray
            Left vector of the rank-1 term, shape (L,).
        b : np.ndarray
            Right vector of the rank-1 term, shape (K,).

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Updated (U, S, Vt).
        """
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)

        U, S, Vt = self.U, self.S, self.Vt
        r = len(S)

        # Project a onto current left subspace
        m = U.T @ a                      # (r,)
        p = a - U @ m                    # (L,)  orthogonal residual
        p_norm = float(np.linalg.norm(p))
        if p_norm > 1e-14:
            P = p / p_norm               # (L,)
        else:
            P = np.zeros_like(p)
            p_norm = 0.0

        # Project b onto current right subspace
        n = Vt @ b                       # (r,)
        q = b - Vt.T @ n                 # (K,)  orthogonal residual
        q_norm = float(np.linalg.norm(q))
        if q_norm > 1e-14:
            Q = q / q_norm               # (K,)
        else:
            Q = np.zeros_like(q)
            q_norm = 0.0

        # Build the (r+1) x (r+1) center matrix
        # K_center = [[diag(S) + m n^T,  ||q|| m],
        #             [||p|| n^T,         ||p|| ||q||]]
        K_center = np.zeros((r + 1, r + 1), dtype=np.float64)
        K_center[:r, :r] = np.diag(S) + np.outer(m, n)
        K_center[:r, r] = q_norm * m
        K_center[r, :r] = p_norm * n
        K_center[r, r] = p_norm * q_norm

        # Diagonalise K_center
        U_k, S_new, Vt_k = np.linalg.svd(K_center, full_matrices=False)

        # Reconstruct full U and Vt
        # U_new = [U | P] @ U_k,  Vt_new = Vt_k @ [Vt; Q^T]
        U_ext = np.column_stack([U, P.reshape(-1, 1)])  # (L, r+1)
        U_full = U_ext @ U_k                            # (L, r+1)

        Vt_ext = np.vstack([Vt, Q.reshape(1, -1)])      # (r+1, K)
        Vt_full = Vt_k @ Vt_ext                         # (r+1, K)

        # Truncate back to rank r
        self.U = U_full[:, :r]
        self.S = S_new[:r]
        self.Vt = Vt_full[:r, :]

        self._step += 1

        return self.U, self.S, self.Vt

    def slide_window(
        self,
        new_sample: float,
        window_data: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Slide the Hankel matrix by one sample.

        Performs a downdate (remove contribution of departing top row)
        followed by an update (add contribution of arriving bottom row),
        then optionally triggers a full SVD reset.

        Parameters
        ----------
        new_sample : float
            The incoming scalar sample appended to the window.
        window_data : np.ndarray
            The *new* window's raw samples (after the slide), used for
            full SVD reset when ``refresh_every`` steps are reached.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Updated (U, S, Vt).
        """
        L = self.U.shape[0]   # embedding dimension (rows)
        K = self.Vt.shape[1]  # number of columns

        # --- Downdate: remove the departing top row ---
        # The top row of the old Hankel matrix is the oldest L-length
        # lagged vector.  We subtract it as a rank-1 outer product.
        # In a standard Hankel matrix, shifting up means removing
        # row 0 and shifting all rows up.  The departing row was
        # e_0 (first standard basis vector in R^L) times the old
        # top row (which is encoded in U @ diag(S) @ Vt row 0).
        # Instead of tracking the old top row explicitly, we note
        # that window_data already reflects the *new* window state.
        # The old top row of the Hankel was:
        #   old_window[0 : K] (the first K samples of the old window)
        # After the slide, old_window[i] = window_data[i-1] for i>=1,
        # and the sample that left is window_data[-1] shifted out...
        #
        # For a clean implementation: we reconstruct the departing row
        # from the current SVD factors (row 0 of X = U[0,:] * S * Vt).

        departing_row = self.U[0, :] * self.S @ self.Vt  # (K,)
        a_down = np.zeros(L, dtype=np.float64)
        a_down[0] = 1.0
        # Downdate: X -= a_down @ departing_row.T
        self.update(-a_down, departing_row)

        # --- Shift the interpretation of rows (implicit reindexing) ---
        # After removing the top row, conceptually rows 1..L-1 become
        # rows 0..L-2, and a new row L-1 is added.  We achieve this by
        # permuting U's rows: shift up by 1.
        self.U = np.roll(self.U, -1, axis=0)

        # --- Update: add the arriving bottom row ---
        # The new bottom row of the Hankel matrix contains the newest
        # lagged samples.  For a Hankel matrix built from window_data
        # with embedding dimension L:
        #   new_row[j] = window_data[L-1 + j]  for j = 0..K-1
        N = len(window_data)
        K_new = N - L + 1
        if K_new != K:
            # Dimensions must match; if not, fall back to full reset.
            self._full_svd_reset(
                _build_hankel(window_data, L),
            )
            return self.U, self.S, self.Vt

        arriving_row = window_data[L - 1: L - 1 + K].copy()

        # The current row L-1 in the shifted U should be zero (it was
        # rolled from the old row 0 which we downdated).  We subtract
        # whatever is there and add the correct row.
        current_bottom = self.U[-1, :] * self.S @ self.Vt  # (K,)
        a_up = np.zeros(L, dtype=np.float64)
        a_up[L - 1] = 1.0

        # Remove stale content at the bottom row
        self.update(-a_up, current_bottom)
        # Add the correct new bottom row
        self.update(a_up, arriving_row)

        # --- Periodic full SVD reset ---
        if self._step >= self.refresh_every:
            self._full_svd_reset(
                _build_hankel(window_data, L),
            )

        return self.U, self.S, self.Vt

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _full_svd_reset(self, X: np.ndarray) -> None:
        """Reset internal SVD factors from a full decomposition.

        Parameters
        ----------
        X : np.ndarray
            Full trajectory matrix to re-decompose.
        """
        r = len(self.S)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        rank = min(r, len(S))
        self.U = U[:, :rank]
        self.S = S[:rank]
        self.Vt = Vt[:rank, :]
        self._step = 0


def _build_hankel(x: np.ndarray, L: int) -> np.ndarray:
    """Build a standard Hankel trajectory matrix.

    Parameters
    ----------
    x : np.ndarray
        Signal of length N.
    L : int
        Embedding dimension (number of rows).

    Returns
    -------
    np.ndarray
        Trajectory matrix of shape (L, N - L + 1).
    """
    N = len(x)
    K = N - L + 1
    X = np.empty((L, K), dtype=np.float64)
    for i in range(L):
        X[i, :] = x[i: i + K]
    return X