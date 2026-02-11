"""Cross-window consistency metrics for component tracking."""

from typing import Optional

import numpy as np
from scipy.fft import fft, fftfreq


def compute_pair_metrics(
    comp_k: np.ndarray,
    comp_k1: np.ndarray,
    overlap_len: int,
    fs: float = 1.0,
) -> dict[str, float]:
    """Compute pairwise metrics on overlap region between two components.
    
    Metrics:
    - corr: Sign-invariant normalized correlation
    - overlap_l2: Normalized L2 difference
    - energy_delta: Relative energy change
    - freq_delta: Difference in FFT peak frequency
    
    Args:
        comp_k: Component from window k.
        comp_k1: Component from window k+1 (should be sign-corrected).
        overlap_len: Length of overlap region.
        fs: Sampling frequency.
        
    Returns:
        Dictionary with metric values.
    """
    # Extract overlap regions
    ov_k = comp_k[-overlap_len:]
    ov_k1 = comp_k1[:overlap_len]
    
    # Correlation (sign-invariant)
    corr_pos = np.corrcoef(ov_k, ov_k1)[0, 1]
    corr_neg = np.corrcoef(ov_k, -ov_k1)[0, 1]
    corr = max(abs(corr_pos), abs(corr_neg))
    
    # Normalized L2 difference
    norm_k = np.linalg.norm(ov_k)
    norm_k1 = np.linalg.norm(ov_k1)
    if norm_k > 0 and norm_k1 > 0:
        # Normalize both
        ov_k_norm = ov_k / norm_k
        ov_k1_norm = ov_k1 / norm_k1
        overlap_l2 = np.linalg.norm(ov_k_norm - ov_k1_norm)
    else:
        overlap_l2 = 1.0 if norm_k != norm_k1 else 0.0
    
    # Energy delta (relative change)
    energy_k = np.sum(ov_k ** 2)
    energy_k1 = np.sum(ov_k1 ** 2)
    if energy_k > 0:
        energy_delta = abs(energy_k1 - energy_k) / energy_k
    else:
        energy_delta = 1.0 if energy_k1 > 0 else 0.0
    
    # Frequency delta
    freq_k = _estimate_dominant_freq(ov_k, fs)
    freq_k1 = _estimate_dominant_freq(ov_k1, fs)
    
    if freq_k > 0 and freq_k1 > 0:
        freq_delta = abs(freq_k - freq_k1)
    else:
        freq_delta = np.nan
    
    return {
        "corr": float(corr),
        "overlap_l2": float(overlap_l2),
        "energy_delta": float(energy_delta),
        "freq_delta": float(freq_delta) if not np.isnan(freq_delta) else None,
    }


def _estimate_dominant_freq(signal: np.ndarray, fs: float) -> float:
    """Estimate dominant frequency via FFT peak.
    
    Args:
        signal: 1D signal array.
        fs: Sampling frequency.
        
    Returns:
        Dominant frequency in Hz, or 0.0 if not found.
    """
    fft_vals = fft(signal)
    freqs = fftfreq(len(signal), 1 / fs)
    
    # Find peak in positive frequencies
    positive_mask = freqs > 0
    if not np.any(positive_mask):
        return 0.0
    
    positive_freqs = freqs[positive_mask]
    positive_power = np.abs(fft_vals[positive_mask]) ** 2
    
    if np.sum(positive_power) == 0:
        return 0.0
    
    peak_idx = np.argmax(positive_power)
    return positive_freqs[peak_idx]
