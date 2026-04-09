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


class TestStatefulMatcher:
    """Tests for the stateful, multi-window-lookback matcher."""

    def _signals(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        t = np.arange(200) / 100.0
        a = np.sin(2.0 * np.pi * 5.0 * t)
        b = np.sin(2.0 * np.pi * 20.0 * t)
        c = np.sin(2.0 * np.pi * 40.0 * t)
        return a, b, c

    def test_lookback_reconnects_across_gap(self) -> None:
        """A component absent for one window must rejoin its trajectory."""
        m = ComponentMatcher(
            distance="d_corr", fs=100.0, lookback=3, max_cost=0.5,
        )
        a, b, _ = self._signals()
        # window 0: [a, b]
        m1 = m.match_stateful([a, b], overlap=200)
        # window 1: [a]   (b drops out)
        m2 = m.match_stateful([a], overlap=200)
        # window 2: [a, b]  (b reappears)
        m3 = m.match_stateful([a, b], overlap=200)

        # b's id in window 2 must equal b's id in window 0
        assert m3[1] == m1[1]
        # exactly two trajectories overall
        all_ids = set(m1.values()) | set(m2.values()) | set(m3.values())
        assert len(all_ids) == 2

    def test_three_two_three_yields_three_trajectories(self) -> None:
        """[3,2,3] component-count scenario must give exactly 3 ids."""
        m = ComponentMatcher(
            distance="d_corr", fs=100.0, lookback=3, max_cost=0.5,
        )
        a, b, c = self._signals()
        m1 = m.match_stateful([a, b, c], overlap=200)
        m2 = m.match_stateful([a, c], overlap=200)
        m3 = m.match_stateful([a, b, c], overlap=200)

        all_ids = set(m1.values()) | set(m2.values()) | set(m3.values())
        assert len(all_ids) == 3
        # b reconnects across the gap
        assert m3[1] == m1[1]

    def test_threshold_rejects_bad_match(self) -> None:
        """A component too dissimilar must spawn a new id."""
        m = ComponentMatcher(
            distance="d_corr", fs=100.0, lookback=3, max_cost=0.1,
        )
        a, _, c = self._signals()
        m1 = m.match_stateful([a], overlap=200)
        # Present a wholly different signal — should not be forced
        # onto trajectory 0.
        m2 = m.match_stateful([c], overlap=200)
        assert m2[0] != m1[0]

    def test_previous_window_mapping(self) -> None:
        m = ComponentMatcher(
            distance="d_corr", fs=100.0, lookback=3, max_cost=0.5,
        )
        a, b, _ = self._signals()
        m.match_stateful([a, b], overlap=200)
        m.match_stateful([b, a], overlap=200)
        pwm = m.previous_window_mapping()
        # current idx 0 was b -> previous idx 1; current idx 1 was a -> 0
        assert pwm[0] == 1
        assert pwm[1] == 0

    def test_reset_clears_history(self) -> None:
        m = ComponentMatcher(
            distance="d_corr", fs=100.0, lookback=3, max_cost=0.5,
        )
        a, b, _ = self._signals()
        m.match_stateful([a, b], overlap=200)
        m.reset()
        m2 = m.match_stateful([a, b], overlap=200)
        # After reset, ids restart from 0
        assert set(m2.values()) == {0, 1}
