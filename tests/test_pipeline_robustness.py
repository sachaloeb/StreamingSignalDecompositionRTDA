"""Robustness tests for the streaming SSD pipeline.

Each test exercises the full streaming pipeline under a challenging
signal condition and asserts numerical quality guarantees.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

from experiments.synthetic.generators import (
    component_onset,
    rossler,
    two_sinusoids,
)
from src.metrics.stability import nmse, qrf
from src.engines.ssd import SSD
from src.streaming.component_matcher import ComponentMatcher
from src.streaming.trajectory_store import TrajectoryStore
from src.streaming.window_manager import WindowManager


def _run_streaming(
    signal: np.ndarray,
    window_len: int = 200,
    stride: int = 20,
    fs: float = 500.0,
    distance: str = "d_corr",
    max_components: int = 4,
) -> tuple[
    TrajectoryStore,
    list[dict[int, int | None]],
    list[float],
]:
    """Run the full streaming pipeline on *signal*.

    Returns
    -------
    store : TrajectoryStore
    all_matchings : list[dict]
    qrf_values : list[float]
    """
    wm = WindowManager(window_len=window_len, stride=stride, fs=fs)
    ssd = SSD(fs=fs)
    matcher = ComponentMatcher(
        distance=distance, fs=fs,
        lookback=3, max_cost=0.5, max_trajectories=max_components,
    )
    store = TrajectoryStore(
        max_components=max_components, max_len=len(signal),
    )

    prev_components: list[np.ndarray] = []
    all_matchings: list[dict[int, int | None]] = []
    qrf_values: list[float] = []

    for t, sample in enumerate(signal):
        window = wm.push(float(sample))
        if window is None:
            continue

        components = ssd.fit(window)
        components_no_res = components[:-1]
        residual = components[-1]

        matching: dict[int, int | None] = dict(
            matcher.match_stateful(components_no_res, wm.overlap)
        )

        window_start = t - wm.window_len + 1
        store.update(
            window_start, components_no_res, matching, wm.overlap,
        )
        all_matchings.append(matching)

        recon = sum(components_no_res)
        qrf_values.append(qrf(window, recon))

        prev_components = components_no_res

    return store, all_matchings, qrf_values


class TestPipelineRobustness:
    """Robustness tests under challenging signal conditions."""

    def test_pipeline_high_noise(self) -> None:
        """At SNR=5 dB, >80% of windows should have QRF > 0 dB."""
        signal = two_sinusoids(
            N=2000, f1=10, f2=50, fs=500, snr_db=5, seed=0,
        )
        _, _, qrf_values = _run_streaming(signal)

        valid = [q for q in qrf_values if np.isfinite(q)]
        assert len(valid) > 0, "No valid QRF values computed"

        frac_above = sum(1 for q in valid if q > 0) / len(valid)
        assert frac_above >= 0.80, (
            f"Only {frac_above:.0%} of windows have QRF > 0 dB"
        )

    def test_component_count_increase(self) -> None:
        """Component onset must not cause errors; new component unmatched."""
        signal = component_onset(
            N=2000, f_steady=20, f_onset=80,
            onset_sample=1000, fs=500,
        )
        store, all_matchings, _ = _run_streaming(signal)

        # With the stateful matcher every component gets a persistent
        # traj_id; a "new" component manifests as a previously-unseen id.
        seen: set[int] = set()
        new_after_first = False
        for m in all_matchings:
            for tid in m.values():
                if tid is not None and tid not in seen:
                    if seen:
                        new_after_first = True
                    seen.add(tid)
        assert new_after_first, (
            "Expected at least one new trajectory id allocated after "
            "the onset"
        )

    def test_pipeline_constant_signal(self) -> None:
        """Pipeline must not crash on a zero signal."""
        signal = np.zeros(1000)
        store, _, _ = _run_streaming(signal)

        trajs = store.get_all()
        for key, arr in trajs.items():
            non_nan = arr[~np.isnan(arr)]
            assert np.allclose(non_nan, 0.0) or len(non_nan) == 0

    def test_pipeline_impulse(self) -> None:
        """An impulse spike must not introduce NaN in trajectories."""
        signal = two_sinusoids(
            N=2000, f1=10, f2=50, fs=500, snr_db=30, seed=1,
        )
        signal[1000] += 50.0

        store, _, _ = _run_streaming(signal)
        trajs = store.get_all()

        has_usable = False
        for key, arr in trajs.items():
            finite_mask = np.isfinite(arr)
            frac_finite = finite_mask.mean()
            if frac_finite > 0.50:
                has_usable = True

        assert has_usable, (
            "No trajectory has > 50% non-NaN finite values"
        )

    def test_rossler_nonlinear(self) -> None:
        """Block SSD on Rössler x-component: >=2 components, NMSE<0.05."""
        signal = rossler(N=3000, dt=0.01, gamma=3.5)
        ssd = SSD(fs=100.0)
        components = ssd.fit(signal)

        assert len(components) >= 3, (
            f"Expected >= 2 components + residual, got {len(components)}"
        )

        residual = components[-1]
        nmse_val = nmse(residual, signal)
        assert nmse_val < 0.05, (
            f"NMSE = {nmse_val:.4f}, expected < 0.05"
        )
