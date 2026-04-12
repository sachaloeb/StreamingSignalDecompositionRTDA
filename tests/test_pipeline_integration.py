"""Integration tests for the end-to-end streaming SSD pipeline.

Verifies full streaming loop convergence, trajectory quality,
reconstruction accuracy, and CSV output correctness.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest

from experiments.synthetic.generators import chirp_plus_sinusoid
from experiments.run_experiment import run
from src.engines.ssd import SSD
from src.streaming.component_matcher import ComponentMatcher
from src.streaming.trajectory_store import TrajectoryStore
from src.streaming.window_manager import WindowManager


class TestFullStreamingRun:
    """End-to-end sample-by-sample streaming test."""

    def test_full_streaming_run(self) -> None:
        """Stream chirp+sinusoid sample-by-sample; check quality."""
        signal = chirp_plus_sinusoid(
            N=5000, f_sin=50, f_start=10, f_end=150,
            fs=1000, snr_db=20, seed=42,
        )

        wm = WindowManager(window_len=300, stride=30, fs=1000.0)
        ssd = SSD(fs=1000.0)
        matcher = ComponentMatcher(
            distance="d_corr", fs=1000.0,
            lookback=3, max_cost=0.5, max_trajectories=4,
        )
        store = TrajectoryStore(
            max_components=4, max_len=len(signal),
        )

        prev_components: list[np.ndarray] = []
        for t, sample in enumerate(signal):
            window = wm.push(float(sample))
            if window is None:
                continue

            components = ssd.fit(window)
            components_no_res = components[:-1]

            matching = matcher.match_stateful(
                components_no_res, wm.overlap,
            )

            window_start = t - wm.window_len + 1
            store.update(
                window_start, components_no_res,
                matching, wm.overlap,
            )
            prev_components = components_no_res

        trajs = store.get_all()
        assert len(trajs) >= 1, "No trajectories stored"
        assert len(trajs) <= 4, (
            f"Trajectory count {len(trajs)} exceeds max_components=4"
        )

        has_usable = False
        for arr in trajs.values():
            frac_non_nan = np.sum(~np.isnan(arr)) / len(arr)
            if frac_non_nan > 0.50:
                has_usable = True
                break
        assert has_usable, (
            "No trajectory has > 50% non-NaN values"
        )

        reconstruction = np.zeros(len(signal))
        counts = np.zeros(len(signal))
        for arr in trajs.values():
            valid = ~np.isnan(arr)
            n = min(len(arr), len(signal))
            valid_n = valid[:n]
            reconstruction[:n][valid_n] += arr[:n][valid_n]
            counts[:n][valid_n] += 1.0

        covered = counts > 0
        assert covered.any(), "No samples covered by trajectories"

        sig_covered = signal[covered]
        rec_covered = reconstruction[covered]
        rel_err = (
            np.linalg.norm(sig_covered - rec_covered)
            / (np.linalg.norm(sig_covered) + 1e-15)
        )
        assert rel_err < 0.6, (
            f"Relative L2 reconstruction error = {rel_err:.3f}, "
            f"expected < 0.6"
        )


class TestMetricsCsvOutput:
    """Verify CSV output from the experiment runner."""

    def test_metrics_csv_output(self, tmp_path: Path) -> None:
        """Run baseline config and validate metrics CSV structure."""
        out_dir = str(tmp_path / "integration_test")
        run(
            config_path="experiments/configs/baseline.yaml",
            output_dir=out_dir,
        )

        csv_path = Path(out_dir) / "metrics.csv"
        assert csv_path.exists(), "metrics.csv not created"

        df = pd.read_csv(csv_path)
        assert len(df) > 0, "metrics.csv is empty"

        required = [
            "qrf", "energy_continuity",
            "singular_value_drift", "matching_confidence",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

        qrf_nan_frac = df["qrf"].isna().mean()
        assert qrf_nan_frac < 0.1, (
            f"QRF NaN fraction = {qrf_nan_frac:.2f}, expected < 0.1"
        )

        mc_nan_frac = df["matching_confidence"].isna().mean()
        assert mc_nan_frac < 0.1, (
            f"matching_confidence NaN fraction = "
            f"{mc_nan_frac:.2f}, expected < 0.1"
        )
