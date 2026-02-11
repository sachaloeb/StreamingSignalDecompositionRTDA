"""Tests for component tracking and alignment."""

import numpy as np
import pytest

from streamssd.tracking import align_components, compute_similarity_matrix, fix_component_sign


def test_compute_similarity_matrix_same():
    """Test similarity matrix for identical components."""
    comp1 = [np.sin(np.linspace(0, 4 * np.pi, 100))]
    comp2 = [np.sin(np.linspace(0, 4 * np.pi, 100))]
    
    sim = compute_similarity_matrix(comp1, comp2, overlap_len=100, fs=1.0)
    
    assert sim.shape == (1, 1)
    assert sim[0, 0] > 0.9  # Should be highly correlated


def test_compute_similarity_matrix_different():
    """Test similarity matrix for different components."""
    comp1 = [np.sin(np.linspace(0, 4 * np.pi, 100))]
    comp2 = [np.random.randn(100)]  # Random noise
    
    sim = compute_similarity_matrix(comp1, comp2, overlap_len=100, fs=1.0)
    
    assert sim.shape == (1, 1)
    assert sim[0, 0] < 0.5  # Should be low correlation


def test_align_components_perfect_match():
    """Test alignment with perfect matches."""
    # Create two windows with matching components
    comp_k = [
        np.sin(np.linspace(0, 4 * np.pi, 100)),
        np.cos(np.linspace(0, 4 * np.pi, 100)),
    ]
    comp_k1 = [
        np.sin(np.linspace(0, 4 * np.pi, 100)),
        np.cos(np.linspace(0, 4 * np.pi, 100)),
    ]
    
    matches, unmatched_k, unmatched_k1 = align_components(
        comp_k, comp_k1, overlap_len=100, similarity_threshold=0.3, fs=1.0
    )
    
    assert len(matches) == 2
    assert len(unmatched_k) == 0
    assert len(unmatched_k1) == 0


def test_align_components_no_match():
    """Test alignment with no matches."""
    comp_k = [np.sin(np.linspace(0, 4 * np.pi, 100))]
    comp_k1 = [np.random.randn(100)]
    
    matches, unmatched_k, unmatched_k1 = align_components(
        comp_k, comp_k1, overlap_len=100, similarity_threshold=0.9, fs=1.0
    )
    
    assert len(matches) == 0
    assert len(unmatched_k) == 1
    assert len(unmatched_k1) == 1


def test_fix_component_sign():
    """Test sign correction."""
    comp_k = np.sin(np.linspace(0, 4 * np.pi, 100))
    comp_k1 = -np.sin(np.linspace(0, 4 * np.pi, 100))  # Flipped sign
    
    comp_k1_fixed = fix_component_sign(comp_k, comp_k1, overlap_len=100)
    
    # Should be highly correlated after fixing
    corr = np.corrcoef(comp_k[-100:], comp_k1_fixed[:100])[0, 1]
    assert corr > 0.9


def test_align_components_empty():
    """Test alignment with empty component lists."""
    matches, unmatched_k, unmatched_k1 = align_components(
        [], [], overlap_len=100, fs=1.0
    )
    
    assert len(matches) == 0
    assert len(unmatched_k) == 0
    assert len(unmatched_k1) == 0
