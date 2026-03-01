"""SSD sub-package: Singular Spectrum Decomposition and related tools."""

from src.ssd.core import SSD
from src.ssd.ssa import auto_ssa, build_trajectory_matrix, diagonal_averaging, svd_decompose
from src.ssd.svd_update import RankOneUpdater

__all__ = [
    "SSD",
    "auto_ssa",
    "build_trajectory_matrix",
    "diagonal_averaging",
    "svd_decompose",
    "RankOneUpdater",
]
