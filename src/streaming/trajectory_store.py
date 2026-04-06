"""Rolling component trajectory management for streaming decomposition.

Stores and extends per-component trajectories across successive
analysis windows, averaging overlapping regions with prior estimates.
"""

from __future__ import annotations

import numpy as np


class TrajectoryStore:
    """Accumulates component trajectories across sliding windows.

    Parameters
    ----------
    max_components : int
        Maximum number of component trajectories to maintain.
    max_len : int or None, optional
        Maximum trajectory length.  ``None`` means unlimited.
    """

    def __init__(
        self,
        max_components: int,
        max_len: int | None = None,
    ) -> None:
        self.max_components = max_components
        self.max_len = max_len
        self._trajectories: dict[int, np.ndarray] = {}
        self._counts: dict[int, np.ndarray] = {}

    def update(
        self,
        window_start: int,
        components: list[np.ndarray],
        matching: dict[int, int | None],
        overlap: int,
    ) -> None:
        """Merge current-window components into stored trajectories.

        Parameters
        ----------
        window_start : int
            Global sample index of the window's first sample.
        components : list[np.ndarray]
            Extracted components for the current window.
        matching : dict[int, int | None]
            Mapping ``{curr_idx: prev_idx}``.  ``None`` values denote
            newly appeared components.
        overlap : int
            Number of overlapping samples with the previous window.
        """
        for curr_idx, prev_idx in matching.items():
            comp = components[curr_idx]
            comp_len = len(comp)
            end = window_start + comp_len

            if prev_idx is None:
                traj_id = self._next_free_id()
                if traj_id is None:
                    # Store is full: honour the max_components cap
                    # instead of unbounded id allocation.
                    continue
            else:
                traj_id = prev_idx

            if traj_id not in self._trajectories:
                required = end
                if self.max_len is not None:
                    required = min(required, self.max_len)
                self._trajectories[traj_id] = np.full(
                    required, np.nan, dtype=np.float64,
                )
                self._counts[traj_id] = np.zeros(
                    required, dtype=np.float64,
                )

            self._ensure_length(traj_id, end)

            traj = self._trajectories[traj_id]
            cnt = self._counts[traj_id]
            for k in range(comp_len):
                pos = window_start + k
                if pos >= len(traj):
                    break
                if np.isnan(traj[pos]):
                    traj[pos] = comp[k]
                    cnt[pos] = 1.0
                else:
                    cnt[pos] += 1.0
                    traj[pos] += (comp[k] - traj[pos]) / cnt[pos]

    def get(self, component_idx: int) -> np.ndarray:
        """Return the full trajectory for a component.

        Parameters
        ----------
        component_idx : int
            Component identifier (trajectory key).

        Returns
        -------
        np.ndarray
            Trajectory array.  Positions never written are ``np.nan``.

        Raises
        ------
        KeyError
            If *component_idx* is not tracked.
        """
        return self._trajectories[component_idx].copy()

    def get_all(self) -> dict[int, np.ndarray]:
        """Return all stored trajectories.

        Returns
        -------
        dict[int, np.ndarray]
            Mapping of component id to trajectory array.
        """
        return {k: v.copy() for k, v in self._trajectories.items()}

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _ensure_length(self, traj_id: int, required: int) -> None:
        """Extend trajectory and count arrays to at least *required*."""
        if self.max_len is not None:
            required = min(required, self.max_len)
        current = len(self._trajectories[traj_id])
        if required > current:
            extra = required - current
            self._trajectories[traj_id] = np.concatenate([
                self._trajectories[traj_id],
                np.full(extra, np.nan, dtype=np.float64),
            ])
            self._counts[traj_id] = np.concatenate([
                self._counts[traj_id],
                np.zeros(extra, dtype=np.float64),
            ])

    def _next_free_id(self) -> int | None:
        """Return the smallest free id in ``[0, max_components)``.

        Returns ``None`` when every slot is occupied, so callers can
        honour the ``max_components`` cap instead of allocating
        unbounded trajectory ids.
        """
        used = set(self._trajectories.keys())
        for i in range(self.max_components):
            if i not in used:
                return i
        return None
