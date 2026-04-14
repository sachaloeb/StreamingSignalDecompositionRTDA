"""Tests for the OptimizedSSD engine.

Covers decomposition quality across all three spectral methods,
bandwidth consistency, edge cases, streaming integration, and speed.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest
from scipy.signal import welch

from experiments.synthetic.generators import chirp_plus_sinusoid
from src.engines.ssd import SSD
from src.engines.ssd_optimized import OptimizedSSD
from src.metrics.stability import nmse, qrf
from src.streaming.component_matcher import ComponentMatcher
from src.streaming.trajectory_store import TrajectoryStore
from src.streaming.window_manager import WindowManager


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------

def _make_signal(N: int = 3000, fs: float = 1000.0) -> np.ndarray:
    return chirp_plus_sinusoid(
        N=N, f_sin=50.0, f_start=10.0, f_end=150.0,
        fs=fs, snr_db=20.0,
    )


# ------------------------------------------------------------------
# Test 1: FWHM decomposition quality
# ------------------------------------------------------------------


class TestFWHMQuality:
    """FWHM method produces a valid decomposition."""

    def test_residual_nmse(self) -> None:
        signal = _make_signal()
        opt = OptimizedSSD(fs=1000.0, spectral_method="fwhm")
        components = opt.fit(signal)
        residual = components[-1]
        val = nmse(residual, signal)
        assert val < 0.05, f"NMSE = {val:.4f}, expected < 0.05"

    def test_component_count_vs_baseline(self) -> None:
        signal = _make_signal()
        baseline = SSD(fs=1000.0)
        opt = OptimizedSSD(fs=1000.0, spectral_method="fwhm")
        n_base = len(baseline.fit(signal))
        n_opt = len(opt.fit(signal))
        # FWHM produces wider bandwidth estimates, so component counts
        # may differ more than ±2 from the baseline.  A tolerance of 4
        # ensures the algorithm is still converging reasonably.
        assert abs(n_opt - n_base) <= 4, (
            f"Component count mismatch: baseline={n_base}, optimized={n_opt}"
        )


# ------------------------------------------------------------------
# Test 2: Moment decomposition quality
# ------------------------------------------------------------------


class TestMomentQuality:
    """Moment method produces a valid decomposition."""

    def test_residual_nmse(self) -> None:
        signal = _make_signal()
        opt = OptimizedSSD(fs=1000.0, spectral_method="moment")
        components = opt.fit(signal)
        residual = components[-1]
        val = nmse(residual, signal)
        assert val < 0.05, f"NMSE = {val:.4f}, expected < 0.05"

    def test_component_count_vs_baseline(self) -> None:
        signal = _make_signal()
        baseline = SSD(fs=1000.0)
        opt = OptimizedSSD(fs=1000.0, spectral_method="moment")
        n_base = len(baseline.fit(signal))
        n_opt = len(opt.fit(signal))
        assert abs(n_opt - n_base) <= 4, (
            f"Component count mismatch: baseline={n_base}, optimized={n_opt}"
        )


# ------------------------------------------------------------------
# Test 3: Gaussian+Jacobian decomposition quality
# ------------------------------------------------------------------


class TestGaussianJacobianQuality:
    """Gaussian+Jacobian should be nearly identical to standard SSD."""

    def test_residual_nmse(self) -> None:
        signal = _make_signal()
        opt = OptimizedSSD(fs=1000.0, spectral_method="gaussian")
        components = opt.fit(signal)
        residual = components[-1]
        val = nmse(residual, signal)
        assert val < 0.02, f"NMSE = {val:.4f}, expected < 0.02"

    def test_component_count_vs_baseline(self) -> None:
        signal = _make_signal()
        baseline = SSD(fs=1000.0)
        opt = OptimizedSSD(fs=1000.0, spectral_method="gaussian")
        n_base = len(baseline.fit(signal))
        n_opt = len(opt.fit(signal))
        assert abs(n_opt - n_base) <= 2, (
            f"Component count mismatch: baseline={n_base}, optimized={n_opt}"
        )


# ------------------------------------------------------------------
# Test 4: δf consistency across methods
# ------------------------------------------------------------------


class TestDeltaFConsistency:
    """All three methods should agree within 50% on a clean sinusoid."""

    def test_deltaf_agreement(self) -> None:
        fs = 1000.0
        N = 1000
        t = np.arange(N) / fs
        x = np.sin(2.0 * np.pi * 50.0 * t)

        freqs, psd = welch(x, fs=fs, nperseg=min(N, 256), nfft=4096)

        df_fwhm = OptimizedSSD._estimate_bandwidth_fwhm(psd, freqs)
        df_moment = OptimizedSSD._estimate_bandwidth_moment(psd, freqs)
        df_gauss = OptimizedSSD._fit_gaussian_with_jacobian(psd, freqs)

        values = [df_fwhm, df_moment, df_gauss]
        mean_val = np.mean(values)

        for name, v in zip(["fwhm", "moment", "gaussian"], values):
            ratio = abs(v - mean_val) / mean_val if mean_val > 0 else 0
            assert ratio < 0.50, (
                f"{name} δf={v:.4f} deviates >{50}% from mean={mean_val:.4f}"
            )


# ------------------------------------------------------------------
# Test 5: FWHM edge cases
# ------------------------------------------------------------------


class TestFWHMEdgeCases:
    """FWHM estimation handles degenerate inputs gracefully."""

    def test_flat_psd(self) -> None:
        freqs = np.linspace(0, 500, 2049)
        psd = np.ones_like(freqs) * 1e-10
        result = OptimizedSSD._estimate_bandwidth_fwhm(psd, freqs)
        assert np.isfinite(result) and result > 0

    def test_single_sample_peak(self) -> None:
        freqs = np.linspace(0, 500, 2049)
        psd = np.zeros_like(freqs)
        psd[100] = 1.0
        result = OptimizedSSD._estimate_bandwidth_fwhm(psd, freqs)
        assert np.isfinite(result) and result > 0

    def test_peak_at_boundary_left(self) -> None:
        freqs = np.linspace(0, 500, 2049)
        psd = np.exp(-freqs / 10.0)  # peak at index 0
        result = OptimizedSSD._estimate_bandwidth_fwhm(psd, freqs)
        assert np.isfinite(result) and result > 0

    def test_peak_at_boundary_right(self) -> None:
        freqs = np.linspace(0, 500, 2049)
        psd = np.exp((freqs - 500) / 10.0)  # peak at last index
        result = OptimizedSSD._estimate_bandwidth_fwhm(psd, freqs)
        assert np.isfinite(result) and result > 0

    def test_zero_psd(self) -> None:
        freqs = np.linspace(0, 500, 2049)
        psd = np.zeros_like(freqs)
        result = OptimizedSSD._estimate_bandwidth_fwhm(psd, freqs)
        assert np.isfinite(result) and result > 0


# ------------------------------------------------------------------
# Test 6: Streaming pipeline integration
# ------------------------------------------------------------------


class TestStreamingIntegration:
    """OptimizedSSD works as a drop-in in the full streaming pipeline."""

    def test_pipeline_runs(self) -> None:
        N = 5000
        fs = 1000.0
        signal = chirp_plus_sinusoid(
            N=N, f_sin=50.0, f_start=10.0, f_end=150.0,
            fs=fs, snr_db=20.0,
        )
        window_len = 300
        stride = 150

        wm = WindowManager(window_len=window_len, stride=stride, fs=fs)
        engine = OptimizedSSD(fs=fs, spectral_method="fwhm")
        matcher = ComponentMatcher(
            distance="d_corr", fs=fs, lookback=3,
            max_cost=0.5, max_trajectories=6,
        )
        store = TrajectoryStore(max_components=6, max_len=N)

        n_windows = 0
        qrf_values: list[float] = []

        for t in range(N):
            window = wm.push(float(signal[t]))
            if window is None:
                continue

            components = engine.fit(window)
            components_no_res = components[:-1]

            matching = dict(
                matcher.match_stateful(components_no_res, wm.overlap)
            )
            window_start = t - wm.window_len + 1
            store.update(
                window_start, components_no_res, matching, wm.overlap,
            )

            recon = (
                np.sum(components_no_res, axis=0)
                if components_no_res
                else np.zeros_like(window)
            )
            qrf_values.append(qrf(window, recon))
            n_windows += 1

        assert n_windows > 0, "No windows were processed"

        valid_qrf = [q for q in qrf_values if np.isfinite(q)]
        assert len(valid_qrf) > 0, "No valid QRF values"

        frac_above = sum(1 for q in valid_qrf if q > 0) / len(valid_qrf)
        assert frac_above >= 0.70, (
            f"Only {frac_above:.0%} of windows have QRF > 0 dB"
        )


# ------------------------------------------------------------------
# Test 7: Speed improvement
# ------------------------------------------------------------------


class TestSpeedImprovement:
    """OptimizedSSD (fwhm) should be faster than baseline SSD."""

    def test_fwhm_faster(self) -> None:
        signal = _make_signal(N=3000, fs=1000.0)

        baseline = SSD(fs=1000.0)
        optimized = OptimizedSSD(fs=1000.0, spectral_method="fwhm")

        # Warm up
        baseline.fit(signal[:500])
        optimized.fit(signal[:500])

        n_trials = 3
        base_times: list[float] = []
        opt_times: list[float] = []

        for _ in range(n_trials):
            t0 = time.perf_counter()
            baseline.fit(signal)
            base_times.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            optimized.fit(signal)
            opt_times.append(time.perf_counter() - t0)

        mean_base = np.mean(base_times)
        mean_opt = np.mean(opt_times)
        speedup = mean_base / mean_opt if mean_opt > 0 else 1.0

        assert speedup >= 1.5, (
            f"Speedup = {speedup:.2f}x, expected >= 1.5x "
            f"(baseline={mean_base:.3f}s, optimized={mean_opt:.3f}s)"
        )