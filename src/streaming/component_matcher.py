"""Hungarian-algorithm component matcher for streaming decomposition.

Matches extracted components across successive sliding windows using
the distance function d(x,y) = 1 - |<x,y>| / (||x||·||y||) from
Harmouche et al. (2017, IEEE TSP), solved optimally via the Hungarian
(Kuhn–Munkres) algorithm.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.metrics.similarity import d_corr, d_freq


class ComponentMatcher:
    """Match components from the current window to previous trajectories.

    Parameters
    ----------
    distance : str, optional
        Distance metric: ``"d_corr"``, ``"d_freq"``, or ``"hybrid"``.
        Default ``"d_corr"``.
    freq_weight : float, optional
        Weight of frequency distance in hybrid mode.  The hybrid cost
        is ``(1 - freq_weight) * d_corr + freq_weight * d_freq_norm``.
        Default 0.0.
    fs : float, optional
        Sampling frequency in Hz.  Default 1.0.
    """

    def __init__(
        self,
        distance: str = "d_corr",
        freq_weight: float = 0.0,
        fs: float = 1.0,
    ) -> None:
        if distance not in {"d_corr", "d_freq", "hybrid"}:
            raise ValueError(
                f"Unknown distance '{distance}'. "
                "Choose from 'd_corr', 'd_freq', 'hybrid'."
            )
        self.distance = distance
        self.freq_weight = freq_weight
        self.fs = fs

    def match(
        self,
        prev: list[np.ndarray],
        curr: list[np.ndarray],
        overlap: int,
    ) -> dict[int, int | None]:
        """Match current components to previous components.

        Parameters
        ----------
        prev : list[np.ndarray]
            Previous window's components.
        curr : list[np.ndarray]
            Current window's components.
        overlap : int
            Number of overlapping samples between windows.

        Returns
        -------
        dict[int, int | None]
            Mapping ``{curr_idx: prev_idx}``.  When
            ``len(curr) > len(prev)`` the extra current components
            map to ``None``.
        """
        if len(prev) == 0:
            return {i: None for i in range(len(curr))}

        cost = self.build_cost_matrix(prev, curr, overlap)
        n_curr, n_prev = cost.shape

        row_ind, col_ind = linear_sum_assignment(cost)

        mapping: dict[int, int | None] = {}
        matched_curr = set()
        for r, c in zip(row_ind, col_ind):
            mapping[int(r)] = int(c)
            matched_curr.add(int(r))

        for i in range(n_curr):
            if i not in matched_curr:
                mapping[i] = None

        return mapping

    def build_cost_matrix(
        self,
        prev: list[np.ndarray],
        curr: list[np.ndarray],
        overlap: int,
    ) -> np.ndarray:
        """Build the assignment cost matrix.

        Parameters
        ----------
        prev : list[np.ndarray]
            Previous window's components.
        curr : list[np.ndarray]
            Current window's components.
        overlap : int
            Overlap length.

        Returns
        -------
        np.ndarray
            Cost matrix of shape ``(len(curr), len(prev))``.
        """
        n_curr = len(curr)
        n_prev = len(prev)
        C = np.zeros((n_curr, n_prev), dtype=np.float64)

        for i in range(n_curr):
            for j in range(n_prev):
                c_seg = curr[i][:overlap]
                p_seg = prev[j][-overlap:]

                if self.distance == "d_corr":
                    C[i, j] = d_corr(p_seg, c_seg)
                elif self.distance == "d_freq":
                    C[i, j] = d_freq(p_seg, c_seg, fs=self.fs)
                else:
                    dc = d_corr(p_seg, c_seg)
                    df = d_freq(p_seg, c_seg, fs=self.fs)
                    nyquist = self.fs / 2.0
                    df_norm = df / nyquist if nyquist > 0 else 0.0
                    C[i, j] = (
                        (1.0 - self.freq_weight) * dc
                        + self.freq_weight * df_norm
                    )
        return C
