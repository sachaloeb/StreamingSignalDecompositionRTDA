"""Metrics sub-package: similarity and stability measures."""

from src.metrics.similarity import d_corr, d_freq, subspace_angle, w_correlation
from src.metrics.stability import (
    dominant_frequency,  # WEEK6-METRICS-FIX: new export
    energy_continuity,
    freq_drift_aggregate,  # WEEK6-METRICS-FIX: new export
    frequency_drift,
    matching_confidence,
    nmse,
    qrf,
    singular_value_drift,
)

__all__ = [
    "d_corr",
    "d_freq",
    "subspace_angle",
    "w_correlation",
    "dominant_frequency",  # WEEK6-METRICS-FIX: new export
    "energy_continuity",
    "freq_drift_aggregate",  # WEEK6-METRICS-FIX: new export
    "frequency_drift",
    "matching_confidence",
    "nmse",
    "qrf",
    "singular_value_drift",
]
