"""Visualization utilities for the streaming SSD pipeline.

Provides publication-quality matplotlib figures for decomposition
results, component trajectories, spectral analysis, streaming
metrics, and a unified pipeline dashboard.
"""

from __future__ import annotations

from src.visualization.component_plots import (
    plot_component_spectra,
    plot_decomposition,
    plot_matching_graph,
    plot_trajectory_overlay,
)
from src.visualization.metrics_plots import plot_metrics_over_windows
from src.visualization.pipeline_dashboard import plot_pipeline_dashboard
from src.visualization.window_inspector import (
    plot_nmse_over_time,
    plot_window_grid,
    plot_window_reconstruction,
)

__all__ = [
    "plot_decomposition",
    "plot_trajectory_overlay",
    "plot_component_spectra",
    "plot_matching_graph",
    "plot_metrics_over_windows",
    "plot_pipeline_dashboard",
    "plot_window_reconstruction",
    "plot_window_grid",
    "plot_nmse_over_time",
]
