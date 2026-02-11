"""Component tracking and alignment across consecutive windows."""

from typing import Optional

import numpy as np
from scipy.fft import fft, fftfreq
from scipy.optimize import linear_sum_assignment


def compute_similarity_matrix(
    components_k: list[np.ndarray],
    components_k1: list[np.ndarray],
    overlap_len: int,
    fs: float,
    freq_penalty_weight: float = 0.0,
) -> np.ndarray:
    """Compute similarity matrix between components in two consecutive windows.
    
    Similarity is based on normalized correlation on the overlap region.
    Optionally includes frequency penalty.
    
    Args:
        components_k: List of components from window k.
        components_k1: List of components from window k+1.
        overlap_len: Length of overlap region (typically W - stride).
        fs: Sampling frequency.
        freq_penalty_weight: Weight for frequency penalty (0 = disabled).
        
    Returns:
        Similarity matrix of shape (len(components_k), len(components_k1)).
        Higher values indicate better matches.
    """
    n_k = len(components_k)
    n_k1 = len(components_k1)
    
    if n_k == 0 or n_k1 == 0:
        return np.zeros((n_k, n_k1))
    
    # Extract overlap regions (last overlap_len samples from k, first from k1)
    overlaps_k = [comp[-overlap_len:] for comp in components_k]
    overlaps_k1 = [comp[:overlap_len] for comp in components_k1]
    
    similarity = np.zeros((n_k, n_k1))
    
    for i, ov_k in enumerate(overlaps_k):
        for j, ov_k1 in enumerate(overlaps_k1):
            # Normalized correlation (sign-invariant)
            corr_pos = np.corrcoef(ov_k, ov_k1)[0, 1]
            corr_neg = np.corrcoef(ov_k, -ov_k1)[0, 1]
            corr = max(abs(corr_pos), abs(corr_neg))
            
            # Frequency penalty (if enabled)
            freq_penalty = 0.0
            if freq_penalty_weight > 0:
                # Estimate frequencies
                freq_k = _estimate_dominant_freq(ov_k, fs)
                freq_k1 = _estimate_dominant_freq(ov_k1, fs)
                
                if freq_k > 0 and freq_k1 > 0:
                    freq_diff = abs(freq_k - freq_k1)
                    # Normalize penalty (assume max diff ~ fs/2)
                    freq_penalty = freq_penalty_weight * (freq_diff / (fs / 2))
            
            similarity[i, j] = corr - freq_penalty
    
    return similarity


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


def align_components(
    components_k: list[np.ndarray],
    components_k1: list[np.ndarray],
    overlap_len: int,
    similarity_threshold: float = 0.3,
    fs: float = 1.0,
    freq_penalty_weight: float = 0.0,
) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    """Align components between two consecutive windows using Hungarian assignment.
    
    Args:
        components_k: Components from window k.
        components_k1: Components from window k+1.
        overlap_len: Length of overlap region.
        similarity_threshold: Minimum similarity for valid match.
        fs: Sampling frequency.
        freq_penalty_weight: Weight for frequency penalty.
        
    Returns:
        Tuple of:
        - matches: List of (i, j, similarity) tuples for matched pairs.
        - unmatched_k: Set of unmatched indices from window k.
        - unmatched_k1: Set of unmatched indices from window k+1.
    """
    if len(components_k) == 0 or len(components_k1) == 0:
        unmatched_k = set(range(len(components_k)))
        unmatched_k1 = set(range(len(components_k1)))
        return [], unmatched_k, unmatched_k1
    
    # Compute similarity matrix
    similarity = compute_similarity_matrix(
        components_k,
        components_k1,
        overlap_len,
        fs,
        freq_penalty_weight=freq_penalty_weight,
    )
    
    # Hungarian assignment (maximize similarity)
    # Convert to cost matrix (negate for minimization)
    cost = -similarity
    row_indices, col_indices = linear_sum_assignment(cost)
    
    # Filter by threshold and collect matches
    matches = []
    matched_k = set()
    matched_k1 = set()
    
    for i, j in zip(row_indices, col_indices):
        sim = similarity[i, j]
        if sim >= similarity_threshold:
            matches.append((i, j, sim))
            matched_k.add(i)
            matched_k1.add(j)
    
    # Unmatched components
    unmatched_k = set(range(len(components_k))) - matched_k
    unmatched_k1 = set(range(len(components_k1))) - matched_k1
    
    return matches, unmatched_k, unmatched_k1


def fix_component_sign(
        comp_k: np.ndarray,
        comp_k1: np.ndarray,
        overlap_len: int,
) -> np.ndarray:
    """Fix sign of comp_k1 to match comp_k on overlap region.

    Args:
        comp_k: Component from window k.
        comp_k1: Component from window k+1 (may have wrong sign).
        overlap_len: Length of overlap region.

    Returns:
        comp_k1 with sign corrected if needed.
    """
    ov_k = comp_k[-overlap_len:]
    ov_k1 = comp_k1[:overlap_len]

    corr_pos = np.corrcoef(ov_k, ov_k1)[0, 1]
    corr_neg = np.corrcoef(ov_k, -ov_k1)[0, 1]

    # Flip if negative correlation is better, or if they're equal, prefer positive
    if abs(corr_neg) > abs(corr_pos) or (abs(corr_neg) == abs(corr_pos) and corr_pos < 0):
        return -comp_k1
    return comp_k1
