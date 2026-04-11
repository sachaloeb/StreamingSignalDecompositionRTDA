"""Decomposition engines (Strategy pattern).

Each engine implements :class:`DecompositionEngine` and is dispatched
by name via :func:`get_engine`. Adding a new engine requires only a
new module here and a single line in the ``_REGISTRY`` below.
"""

from __future__ import annotations

from src.engines.base import DecompositionEngine
from src.engines.ssa import (
    SSA,
    auto_ssa,
    build_trajectory_matrix,
    diagonal_averaging,
    svd_decompose,
)
from src.engines.rsvd import rsvd
from src.engines.ssd import SSD
from src.engines.ssd_incremental import IncrementalSSD
from src.engines.svd_update import RankOneUpdater

_REGISTRY: dict[str, type[DecompositionEngine]] = {
    "ssd": SSD,
    "ssa": SSA,
    "ssd_incremental": IncrementalSSD,
}


def get_engine(name: str, fs: float, **kwargs: object) -> DecompositionEngine:
    """Instantiate a decomposition engine by name.

    Parameters
    ----------
    name : str
        Engine identifier (e.g. ``"ssd"``, ``"ssa"``).
    fs : float
        Sampling frequency in Hz.
    **kwargs
        Engine-specific keyword arguments forwarded to the constructor.

    Returns
    -------
    DecompositionEngine
    """
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown engine '{name}'. Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[key](fs=fs, **kwargs)


__all__ = [
    "DecompositionEngine",
    "IncrementalSSD",
    "SSA",
    "SSD",
    "RankOneUpdater",
    "auto_ssa",
    "build_trajectory_matrix",
    "diagonal_averaging",
    "get_engine",
    "rsvd",
    "svd_decompose",
]
