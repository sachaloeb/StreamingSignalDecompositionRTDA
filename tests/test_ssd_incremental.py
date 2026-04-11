"""Tests for the IncrementalSSD engine."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

from src.engines import get_engine
from src.engines.ssd import SSD
from src.engines.ssd_incremental import IncrementalSSD
from src.metrics.stability import nmse
from src.streaming.component_matcher import ComponentMatcher
from src.streaming.trajectory_store import TrajectoryStore
from src.streaming.window_manager import WindowManager


class TestIncrementalSSD:
    """Test the IncrementalSSD engine."""

    def _make_signal(self, n: int = 500, fs: float = 500.0) -> np.ndarray:
        t = np.arange(n) / fs
        return np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 50 * t)

    def test_incremental_produces_components(self) -> None:
        """IncrementalSSD should return at least 2 components + residual."""
        x = self._make_signal()
        engine = IncrementalSSD(fs=500.0)
        components = engine.fit(x)
        assert len(components) >= 3, f"Expected >= 3 (2 + residual), got {len(components)}"

    def test_incremental_vs_standard_quality(self) -> None:
        """IncrementalSSD components should be comparable to standard SSD."""
        x = self._make_signal()
        fs = 500.0

        ssd = SSD(fs=fs)
        inc = IncrementalSSD(fs=fs)

        comps_ssd = ssd.fit(x)
        comps_inc = inc.fit(x)

        # Reconstruction NMSE for both should be low
        recon_ssd = sum(comps_ssd[:-1])
        recon_inc = sum(comps_inc[:-1])

        nmse_ssd = nmse(x - recon_ssd, x)
        nmse_inc = nmse(x - recon_inc, x)

        assert nmse_inc < 0.05, f"IncrementalSSD NMSE {nmse_inc:.4f} too high"
        # IncrementalSSD should be in the same ballpark as SSD
        assert nmse_inc < nmse_ssd + 0.05, (
            f"IncrementalSSD NMSE {nmse_inc:.4f} much worse than SSD {nmse_ssd:.4f}"
        )

    def test_warm_start_consistency(self) -> None:
        """Calling fit twice (warm start) should still produce good results."""
        fs = 500.0
        x1 = self._make_signal(500, fs)
        x2 = self._make_signal(500, fs)

        engine = IncrementalSSD(fs=fs)

        # First call (cold start)
        comps1 = engine.fit(x1)
        recon1 = sum(comps1[:-1])
        nmse1 = nmse(x1 - recon1, x1)

        # Second call (warm start possible)
        comps2 = engine.fit(x2)
        recon2 = sum(comps2[:-1])
        nmse2 = nmse(x2 - recon2, x2)

        assert nmse1 < 0.05, f"Cold start NMSE {nmse1:.4f}"
        assert nmse2 < 0.05, f"Warm start NMSE {nmse2:.4f}"

    def test_with_rsvd(self) -> None:
        """IncrementalSSD with use_rsvd=True should still produce components."""
        x = self._make_signal()
        engine = IncrementalSSD(fs=500.0, use_rsvd=True)
        components = engine.fit(x)
        assert len(components) >= 2

        recon = sum(components[:-1])
        n = nmse(x - recon, x)
        assert n < 0.1, f"rSVD IncrementalSSD NMSE {n:.4f}"

    def test_get_engine_factory(self) -> None:
        """get_engine('ssd_incremental') should return an IncrementalSSD."""
        engine = get_engine("ssd_incremental", fs=500.0)
        assert isinstance(engine, IncrementalSSD)

    def test_streaming_pipeline_integration(self) -> None:
        """IncrementalSSD should work correctly in the full streaming pipeline."""
        fs = 500.0
        N = 2000
        t = np.arange(N) / fs
        signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 50 * t)

        window_len = 200
        stride = 100

        wm = WindowManager(window_len=window_len, stride=stride, fs=fs)
        engine = IncrementalSSD(fs=fs)
        matcher = ComponentMatcher(
            distance="d_corr", fs=fs,
            lookback=3, max_cost=0.5, max_trajectories=4,
        )
        store = TrajectoryStore(max_components=4, max_len=N)

        n_windows = 0
        for sample_idx in range(N):
            window = wm.push(float(signal[sample_idx]))
            if window is None:
                continue

            components = engine.fit(window)
            components_no_res = components[:-1]

            matching = dict(
                matcher.match_stateful(components_no_res, wm.overlap)
            )
            window_start = sample_idx - wm.window_len + 1
            store.update(
                window_start, components_no_res, matching, wm.overlap,
            )
            n_windows += 1

        assert n_windows > 0, "No windows processed"
        trajs = store.get_all()
        assert len(trajs) > 0, "No trajectories stored"

    def test_cold_start_fallback(self) -> None:
        """When signal changes drastically, should fall back to cold start."""
        fs = 500.0
        engine = IncrementalSSD(fs=fs, subspace_threshold=0.01)

        # First signal: low frequency
        t1 = np.arange(500) / fs
        x1 = np.sin(2 * np.pi * 5 * t1)
        comps1 = engine.fit(x1)

        # Second signal: completely different (high frequency)
        x2 = np.sin(2 * np.pi * 200 * t1)
        comps2 = engine.fit(x2)

        # Should still produce valid output
        assert len(comps2) >= 2
        recon2 = sum(comps2[:-1])
        n = nmse(x2 - recon2, x2)
        assert n < 0.1