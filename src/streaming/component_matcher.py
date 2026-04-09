"""Hungarian-algorithm component matcher for streaming decomposition.

Implements a stateful, multi-window-lookback matcher inspired by
Harmouche et al. (2017, IEEE TSP) Algorithm 2.  The matcher maintains
the last *K* windows of accepted (trajectory_id, component) pairs and
matches every new window's components against the most recent
representative of *every* trajectory that is still active inside the
lookback horizon, not only the immediately previous window.  An
optional cost threshold prevents forced bad matches.

A backward-compatible stateless ``match()`` /
``build_cost_matrix()`` API is preserved so that older callers and
visualisation helpers (e.g. ``plot_matching_graph``) continue to work
unchanged.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.metrics.similarity import d_corr, d_freq


class ComponentMatcher:
    """Match components from the current window to existing trajectories.

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
    lookback : int, optional
        Number of previous windows kept in the lookback buffer.  A
        trajectory remains "active" — and therefore matchable — as
        long as it produced a component within the last *lookback*
        windows.  Default 3.
    max_cost : float, optional
        Maximum acceptable assignment cost.  Hungarian assignments
        whose cost exceeds this threshold are rejected and the
        corresponding current component is treated as a brand-new
        trajectory.  Default 0.5.
    max_trajectories : int or None, optional
        Hard cap on the total number of distinct trajectory ids the
        matcher will ever allocate.  Once the cap is hit, surplus
        current components that cannot be matched to an existing
        active trajectory are flagged with id ``-1`` (which
        :class:`TrajectoryStore` skips).  ``None`` means unbounded.
        Default ``None``.
    """

    DROP_ID: int = -1

    def __init__(
        self,
        distance: str = "d_corr",
        freq_weight: float = 0.0,
        fs: float = 1.0,
        lookback: int = 3,
        max_cost: float = 0.5,
        max_trajectories: int | None = None,
    ) -> None:
        if distance not in {"d_corr", "d_freq", "hybrid"}:
            raise ValueError(
                f"Unknown distance '{distance}'. "
                "Choose from 'd_corr', 'd_freq', 'hybrid'."
            )
        self.distance = distance
        self.freq_weight = freq_weight
        self.fs = fs
        self.lookback = max(1, int(lookback))
        self.max_cost = float(max_cost)
        self.max_trajectories = (
            int(max_trajectories)
            if max_trajectories is not None
            else None
        )

        # Stateful history: list (oldest -> newest) of windows.
        # Each window is a list of (traj_id, component_array).
        self._history: list[list[tuple[int, np.ndarray]]] = []
        self._next_id: int = 0

    # ------------------------------------------------------------------
    # state management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all stateful history (lookback buffer and id counter)."""
        self._history = []
        self._next_id = 0

    def _allocate_id(self) -> int:
        if (
            self.max_trajectories is not None
            and self._next_id >= self.max_trajectories
        ):
            return self.DROP_ID
        tid = self._next_id
        self._next_id += 1
        return tid

    def _active_trajectories(
        self,
    ) -> tuple[list[int], list[np.ndarray]]:
        """Return (traj_ids, components) for all active trajectories.

        For each trajectory id present in the lookback buffer, the
        most recent component representative is returned.
        """
        seen: dict[int, np.ndarray] = {}
        for window in reversed(self._history):
            for tid, comp in window:
                if tid not in seen:
                    seen[tid] = comp
        return list(seen.keys()), list(seen.values())

    # ------------------------------------------------------------------
    # stateful API (preferred)
    # ------------------------------------------------------------------

    def match_stateful(
        self,
        curr: list[np.ndarray],
        overlap: int,
    ) -> dict[int, int]:
        """Match current components against active trajectories.

        Parameters
        ----------
        curr : list[np.ndarray]
            Current window's extracted components.
        overlap : int
            Number of overlapping samples between windows.

        Returns
        -------
        dict[int, int]
            Mapping ``{curr_idx: traj_id}``.  Trajectory ids are
            persistent across windows.  Components that cannot be
            linked to an existing trajectory are assigned a fresh
            id (the matcher itself manages the id allocator, so the
            caller never has to deal with ``None``).
        """
        active_ids, active_comps = self._active_trajectories()
        n_curr = len(curr)
        n_act = len(active_ids)

        mapping: dict[int, int] = {}

        if n_act == 0 or n_curr == 0:
            for i in range(n_curr):
                mapping[i] = self._allocate_id()
        else:
            cost = self._build_cost_matrix(active_comps, curr, overlap)
            big = 1e6
            cost_solve = np.where(np.isfinite(cost), cost, big)
            row_ind, col_ind = linear_sum_assignment(cost_solve)

            assigned: set[int] = set()
            for r, c in zip(row_ind, col_ind):
                r_i = int(r)
                c_i = int(c)
                if (
                    np.isfinite(cost[r_i, c_i])
                    and cost[r_i, c_i] <= self.max_cost
                ):
                    mapping[r_i] = active_ids[c_i]
                    assigned.add(r_i)
            for i in range(n_curr):
                if i not in mapping:
                    mapping[i] = self._allocate_id()

        # commit to history (skip dropped components)
        self._history.append([
            (mapping[i], curr[i])
            for i in range(n_curr)
            if mapping[i] != self.DROP_ID
        ])
        if len(self._history) > self.lookback:
            self._history.pop(0)

        return mapping

    def previous_window_mapping(
        self,
    ) -> dict[int, int | None]:
        """Mapping of the most recently matched window vs. its predecessor.

        Returns ``{curr_idx: prev_window_idx}`` where ``prev_window_idx``
        is the index *within the immediately previous window's
        component list* of the matched component, or ``None`` if the
        current component did not appear in the previous window.

        Useful for legacy per-window metrics (energy_continuity,
        matching_confidence) that index directly into the previous
        window's components.
        """
        if len(self._history) < 2:
            return {}
        prev_ids = [tid for tid, _ in self._history[-2]]
        curr_ids = [tid for tid, _ in self._history[-1]]
        out: dict[int, int | None] = {}
        for ci, tid in enumerate(curr_ids):
            if tid in prev_ids:
                out[ci] = prev_ids.index(tid)
            else:
                out[ci] = None
        return out

    # ------------------------------------------------------------------
    # backward-compatible stateless API
    # ------------------------------------------------------------------

    def match(
        self,
        prev: list[np.ndarray],
        curr: list[np.ndarray],
        overlap: int,
    ) -> dict[int, int | None]:
        """Stateless single-window match (backward-compatible API).

        Compares ``curr`` only against the explicitly supplied
        ``prev`` list and returns ``{curr_idx: prev_idx | None}``.
        Used by visualisation helpers and existing unit tests.  New
        code should prefer :meth:`match_stateful`.
        """
        if len(prev) == 0:
            return {i: None for i in range(len(curr))}

        cost = self.build_cost_matrix(prev, curr, overlap)
        n_curr = cost.shape[0]
        cost_solve = np.where(np.isfinite(cost), cost, 1e6)
        row_ind, col_ind = linear_sum_assignment(cost_solve)

        mapping: dict[int, int | None] = {}
        matched_curr: set[int] = set()
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
        """Cost matrix of shape ``(len(curr), len(prev))``.

        Backward-compatible helper preserved for ``plot_matching_graph``
        and the existing matching_confidence metric.
        """
        return self._build_cost_matrix(prev, curr, overlap)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _pair_cost(
        self,
        p_seg: np.ndarray,
        c_seg: np.ndarray,
    ) -> float:
        if self.distance == "d_corr":
            return float(d_corr(p_seg, c_seg))
        if self.distance == "d_freq":
            return float(d_freq(p_seg, c_seg, fs=self.fs))
        dc = float(d_corr(p_seg, c_seg))
        df = float(d_freq(p_seg, c_seg, fs=self.fs))
        nyquist = self.fs / 2.0
        df_norm = df / nyquist if nyquist > 0 else 0.0
        return (1.0 - self.freq_weight) * dc + self.freq_weight * df_norm

    def _build_cost_matrix(
        self,
        prev: list[np.ndarray],
        curr: list[np.ndarray],
        overlap: int,
    ) -> np.ndarray:
        n_curr = len(curr)
        n_prev = len(prev)
        C = np.full((n_curr, n_prev), np.inf, dtype=np.float64)
        for i in range(n_curr):
            for j in range(n_prev):
                eff = min(overlap, len(curr[i]), len(prev[j]))
                if eff <= 0:
                    continue
                c_seg = curr[i][:eff]
                p_seg = prev[j][-eff:]
                C[i, j] = self._pair_cost(p_seg, c_seg)
        return C