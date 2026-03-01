"""Tests for src.streaming.component_matcher."""

from __future__ import annotations

import numpy as np
import pytest

from src.streaming.component_matcher import ComponentMatcher


@pytest.fixture()
def matcher() -> ComponentMatcher:
    return ComponentMatcher(distance="d_corr", fs=100.0)


class TestComponentMatcher:
    """Tests for the Hungarian component matcher."""

    def test_perfect_match(self, matcher: ComponentMatcher) -> None:
        t = np.arange(200) / 100.0
        comps = [
            np.sin(2.0 * np.pi * 5.0 * t),
            np.sin(2.0 * np.pi * 20.0 * t),
        ]
        mapping = matcher.match(comps, comps, overlap=200)
        assert mapping[0] == 0
        assert mapping[1] == 1

    def test_permuted_match(self, matcher: ComponentMatcher) -> None:
        t = np.arange(200) / 100.0
        c0 = np.sin(2.0 * np.pi * 5.0 * t)
        c1 = np.sin(2.0 * np.pi * 20.0 * t)
        prev = [c0, c1]
        curr = [c1, c0]
        mapping = matcher.match(prev, curr, overlap=200)
        assert mapping[0] == 1
        assert mapping[1] == 0

    def test_new_component(self, matcher: ComponentMatcher) -> None:
        t = np.arange(200) / 100.0
        prev = [np.sin(2.0 * np.pi * 5.0 * t)]
        curr = [
            np.sin(2.0 * np.pi * 5.0 * t),
            np.sin(2.0 * np.pi * 20.0 * t),
        ]
        mapping = matcher.match(prev, curr, overlap=200)
        matched_prevs = [v for v in mapping.values() if v is not None]
        assert 0 in matched_prevs
        none_count = sum(1 for v in mapping.values() if v is None)
        assert none_count == 1

    def test_cost_matrix_shape(self, matcher: ComponentMatcher) -> None:
        t = np.arange(100) / 100.0
        prev = [np.sin(2.0 * np.pi * f * t) for f in [3.0, 7.0]]
        curr = [np.sin(2.0 * np.pi * f * t) for f in [3.0, 7.0, 15.0]]
        C = matcher.build_cost_matrix(prev, curr, overlap=100)
        assert C.shape == (3, 2)
