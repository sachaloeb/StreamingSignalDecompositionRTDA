"""Tests for Hankel embedding utilities."""

import numpy as np
import pytest

from streamssd.embed import get_embedding_params, hankel_embed


def test_hankel_embed_basic():
    """Test basic Hankel embedding."""
    x = np.array([1, 2, 3, 4, 5])
    L = 3
    H = hankel_embed(x, L)
    
    expected = np.array([
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
    ])
    np.testing.assert_array_equal(H, expected)


def test_hankel_embed_dimensions():
    """Test Hankel embedding dimensions."""
    W = 10
    L = 4
    x = np.random.randn(W)
    H = hankel_embed(x, L)
    
    assert H.shape == (L, W - L + 1)
    assert H.shape[1] == 7  # K = 10 - 4 + 1


def test_hankel_embed_invalid():
    """Test Hankel embedding with invalid parameters."""
    x = np.array([1, 2, 3])
    
    with pytest.raises(ValueError):
        hankel_embed(x, L=5)  # L > W


def test_get_embedding_params():
    """Test embedding parameter computation."""
    W = 100
    L, K = get_embedding_params(W)
    
    assert L == 50  # Default: W // 2
    assert K == 51  # W - L + 1
    
    L_custom, K_custom = get_embedding_params(W, L=30)
    assert L_custom == 30
    assert K_custom == 71


def test_get_embedding_params_invalid():
    """Test embedding params with invalid L."""
    with pytest.raises(ValueError):
        get_embedding_params(10, L=10)  # L >= W
    
    with pytest.raises(ValueError):
        get_embedding_params(10, L=0)  # L < 1
