"""Tests for the window-inspection visualisation module."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.visualization.window_inspector import (
    plot_nmse_over_time,
    plot_window_grid,
    plot_window_reconstruction,
)


def test_plot_window_reconstruction_saves_file(
    tmp_path: Path,
) -> None:
    """Saved PNG is non-trivial (> 1 kB)."""
    t = np.arange(300) / 1000.0
    signal = np.sin(2 * np.pi * 10 * t)
    comp1 = 0.8 * np.sin(2 * np.pi * 10 * t)
    out = str(tmp_path / "w.png")
    plot_window_reconstruction(
        signal, [comp1], 0, 0, fs=1000.0, save_path=out,
    )
    assert os.path.exists(out)
    assert os.path.getsize(out) > 1000


def test_plot_window_grid_saves_file(
    tmp_path: Path,
) -> None:
    """Grid PNG is created for 12 records, 9 shown."""
    records: list[dict] = []
    fs = 1000.0
    for i in range(12):
        sig = np.sin(
            2 * np.pi * 10 * np.arange(300) / fs,
        )
        c1 = 0.7 * sig
        records.append({
            "window_idx": i,
            "sample_start": i * 30,
            "window_signal": sig,
            "components": [c1],
        })
    out = str(tmp_path / "grid.png")
    plot_window_grid(
        records, n_windows=9, fs=fs, save_path=out,
    )
    assert os.path.exists(out)


def test_nmse_over_time_perfect_reconstruction() -> None:
    """Perfect reconstruction yields NMSE ~ 0 for every second."""
    fs = 100.0
    N = 1000
    signal = np.sin(
        2 * np.pi * 5 * np.arange(N) / fs,
    )
    t_axis, nmse_vals = plot_nmse_over_time(
        signal, signal.copy(), fs=fs, save_path=None,
    )
    valid = nmse_vals[np.isfinite(nmse_vals)]
    assert len(valid) == 10
    assert np.all(valid < 1e-10)


def test_nmse_over_time_nan_reconstruction() -> None:
    """All-NaN reconstruction → NMSE = 1.0 per second."""
    fs = 100.0
    N = 500
    signal = np.sin(
        2 * np.pi * 5 * np.arange(N) / fs,
    )
    recon = np.full(N, np.nan)
    t_axis, nmse_vals = plot_nmse_over_time(
        signal, recon, fs=fs, save_path=None,
    )
    valid = nmse_vals[np.isfinite(nmse_vals)]
    assert np.allclose(valid, 1.0, atol=1e-6)
