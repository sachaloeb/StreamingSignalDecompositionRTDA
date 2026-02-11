"""Tests for consistency metrics."""

import numpy as np
import pytest

from streamssd.metrics import compute_pair_metrics


def test_compute_pair_metrics_identical():
    """Test metrics for identical components."""
    comp_k = np.sin(np.linspace(0, 4 * np.pi, 100))
    comp_k1 = np.sin(np.linspace(0, 4 * np.pi, 100))
    
    metrics = compute_pair_metrics(comp_k, comp_k1, overlap_len=100, fs=1.0)
    
    assert metrics["corr"] > 0.99
    assert metrics["overlap_l2"] < 0.1
    assert metrics["energy_delta"] < 0.1
    assert metrics["freq_delta"] is not None


def test_compute_pair_metrics_different():
    """Test metrics for different components."""
    comp_k = np.sin(np.linspace(0, 4 * np.pi, 100))
    comp_k1 = np.random.randn(100)
    
    metrics = compute_pair_metrics(comp_k, comp_k1, overlap_len=100, fs=1.0)
    
    assert metrics["corr"] < 0.5
    assert metrics["overlap_l2"] > 0.5
    assert metrics["energy_delta"] > 0.1


def test_compute_pair_metrics_sign_invariant():
    """Test that metrics are sign-invariant."""
    comp_k = np.sin(np.linspace(0, 4 * np.pi, 100))
    comp_k1 = -np.sin(np.linspace(0, 4 * np.pi, 100))
    
    metrics = compute_pair_metrics(comp_k, comp_k1, overlap_len=100, fs=1.0)
    
    # Correlation should be high (sign-invariant)
    assert metrics["corr"] > 0.99


def test_compute_pair_metrics_frequency():
    """Test frequency delta computation."""
    fs = 100.0
    t = np.linspace(0, 1, int(fs))
    comp_k = np.sin(2 * np.pi * 5 * t)  # 5 Hz
    comp_k1 = np.sin(2 * np.pi * 7 * t)  # 7 Hz
    
    metrics = compute_pair_metrics(comp_k, comp_k1, overlap_len=len(comp_k), fs=fs)
    
    assert metrics["freq_delta"] is not None
    assert abs(metrics["freq_delta"] - 2.0) < 1.0  # Should be around 2 Hz difference
