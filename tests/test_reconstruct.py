"""Tests for diagonal averaging and reconstruction."""

import numpy as np
import pytest

from streamssd.embed import hankel_embed
from streamssd.reconstruct import diagonal_average, reconstruct_component


def test_diagonal_average_roundtrip():
    """Test that diagonal_average is inverse of hankel_embed."""
    x = np.random.randn(20)
    L = 10
    H = hankel_embed(x, L)
    x_recon = diagonal_average(H, len(x))
    
    np.testing.assert_allclose(x, x_recon, rtol=1e-10)


def test_diagonal_average_dimensions():
    """Test diagonal averaging dimensions."""
    L, K = 5, 7
    W = L + K - 1
    X = np.random.randn(L, K)
    result = diagonal_average(X, W)
    
    assert len(result) == W
    assert len(result) == 11  # 5 + 7 - 1


def test_diagonal_average_invalid():
    """Test diagonal averaging with invalid W."""
    X = np.random.randn(5, 7)
    
    with pytest.raises(ValueError):
        diagonal_average(X, W=10)  # W != L + K - 1


def test_reconstruct_component():
    """Test component reconstruction from SVD factors."""
    W = 20
    L = 10
    K = 11
    
    # Create test SVD factors
    U = np.random.randn(L)
    s = 2.5
    V = np.random.randn(K)
    
    comp = reconstruct_component(U, s, V, W)
    
    assert len(comp) == W
    # Check that it's the diagonal average of s * U * V^T
    X = s * np.outer(U, V)
    comp_expected = diagonal_average(X, W)
    np.testing.assert_allclose(comp, comp_expected, rtol=1e-10)
