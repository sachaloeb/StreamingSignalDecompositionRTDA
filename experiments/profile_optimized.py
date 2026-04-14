"""Profile the OptimizedSSD variants against baseline SSD.

Runs the same profiling methodology as ``profile_pipeline.py`` but
compares four engine configurations: baseline SSD, OptimizedSSD with
FWHM, moment, and Gaussian+Jacobian bandwidth estimation.

Usage
-----
    python experiments/profile_optimized.py
"""

from __future__ import annotations

import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.synthetic.generators import chirp_plus_sinusoid
from src.engines import get_engine
from src.engines.ssd import SSD
from src.engines.ssd_optimized import OptimizedSSD
from src.streaming.component_matcher import ComponentMatcher
from src.streaming.trajectory_store import TrajectoryStore
from src.streaming.window_manager import WindowManager


def _run_pipeline_instrumented(
    signal: np.ndarray,
    engine: SSD,
    fs: float = 1000.0,
    window_len: int = 300,
    stride: int = 150,
    max_components: int = 10,
) -> dict[str, object]:
    """Run pipeline with fine-grained decomposition breakdown."""
    N = len(signal)
    wm = WindowManager(window_len=window_len, stride=stride, fs=fs)
    matcher = ComponentMatcher(
        distance="hybrid", freq_weight=0.5, fs=fs, lookback=10,
        max_cost=0.6, max_trajectories=max_components,
    )
    store = TrajectoryStore(max_components=max_components, max_len=N)

    # Instrument bandwidth estimation
    bw_times: list[float] = []
    svd_times: list[float] = []
    select_times: list[float] = []
    recon_times: list[float] = []

    # Wrap _fit_gaussian_model (or optimized variant)
    if isinstance(engine, OptimizedSSD):
        if engine.spectral_method == "fwhm":
            orig_bw = OptimizedSSD._estimate_bandwidth_fwhm  # type: ignore[attr-defined]
            def timed_bw(*a, **kw):  # noqa: ANN002
                t0 = time.perf_counter()
                r = orig_bw(*a, **kw)
                bw_times.append(time.perf_counter() - t0)
                return r
            engine._estimate_bandwidth_fwhm = timed_bw  # type: ignore[assignment]
        elif engine.spectral_method == "moment":
            orig_bw = OptimizedSSD._estimate_bandwidth_moment  # type: ignore[attr-defined]
            def timed_bw(*a, **kw):  # noqa: ANN002
                t0 = time.perf_counter()
                r = orig_bw(*a, **kw)
                bw_times.append(time.perf_counter() - t0)
                return r
            engine._estimate_bandwidth_moment = timed_bw  # type: ignore[assignment]
        elif engine.spectral_method == "gaussian":
            orig_bw = OptimizedSSD._fit_gaussian_with_jacobian  # type: ignore[attr-defined]
            def timed_bw(*a, **kw):  # noqa: ANN002
                t0 = time.perf_counter()
                r = orig_bw(*a, **kw)
                bw_times.append(time.perf_counter() - t0)
                return r
            engine._fit_gaussian_with_jacobian = timed_bw  # type: ignore[assignment]
    else:
        orig_bw = SSD._fit_gaussian_model  # type: ignore[attr-defined]
        def timed_bw(*a, **kw):  # noqa: ANN002
            t0 = time.perf_counter()
            r = orig_bw(*a, **kw)
            bw_times.append(time.perf_counter() - t0)
            return r
        engine._fit_gaussian_model = timed_bw  # type: ignore[assignment]

    # Wrap SVD
    orig_svd = engine._decompose_trajectory
    def timed_svd(*a, **kw):  # noqa: ANN002
        t0 = time.perf_counter()
        r = orig_svd(*a, **kw)
        svd_times.append(time.perf_counter() - t0)
        return r
    engine._decompose_trajectory = timed_svd  # type: ignore[assignment]

    # Wrap eigentriple selection
    orig_sel = SSD._select_eigentriples  # type: ignore[attr-defined]
    def timed_sel(*a, **kw):  # noqa: ANN002
        t0 = time.perf_counter()
        r = orig_sel(*a, **kw)
        select_times.append(time.perf_counter() - t0)
        return r
    engine._select_eigentriples = timed_sel  # type: ignore[assignment]

    # Wrap reconstruction
    orig_recon = SSD._reconstruct_component  # type: ignore[attr-defined]
    def timed_recon(*a, **kw):  # noqa: ANN002
        t0 = time.perf_counter()
        r = orig_recon(*a, **kw)
        recon_times.append(time.perf_counter() - t0)
        return r
    engine._reconstruct_component = timed_recon  # type: ignore[assignment]

    t_decompose = 0.0
    t_total_start = time.perf_counter()
    n_windows = 0

    for sample_idx in range(N):
        window = wm.push(float(signal[sample_idx]))
        if window is None:
            continue

        t0 = time.perf_counter()
        components = engine.fit(window)
        t1 = time.perf_counter()
        t_decompose += t1 - t0

        components_no_res = components[:-1]
        matching = dict(
            matcher.match_stateful(components_no_res, wm.overlap)
        )
        window_start = sample_idx - wm.window_len + 1
        store.update(window_start, components_no_res, matching, wm.overlap)
        n_windows += 1

    t_total = time.perf_counter() - t_total_start

    return {
        "n_windows": n_windows,
        "decomposition_s": t_decompose,
        "total_s": t_total,
        "bandwidth_s": sum(bw_times),
        "svd_s": sum(svd_times),
        "eigentriple_sel_s": sum(select_times),
        "reconstruction_s": sum(recon_times),
        "other_decomp_s": max(
            0.0,
            t_decompose
            - sum(bw_times) - sum(svd_times)
            - sum(select_times) - sum(recon_times),
        ),
    }


def _measure_peak_memory(
    signal: np.ndarray,
    engine: SSD,
    **kwargs: object,
) -> float:
    """Measure peak memory in MiB."""
    N = len(signal)
    wm = WindowManager(window_len=300, stride=150, fs=1000.0)
    matcher = ComponentMatcher(
        distance="d_corr", fs=1000.0, lookback=10,
        max_cost=0.6, max_trajectories=10,
    )
    store = TrajectoryStore(max_components=10, max_len=N)

    tracemalloc.start()
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
        store.update(window_start, components_no_res, matching, wm.overlap)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024 * 1024)


def main() -> None:
    """Run profiling comparison of four engine variants."""
    out_dir = ROOT / "results" / "profiling"
    out_dir.mkdir(parents=True, exist_ok=True)

    N = 10000
    fs = 1000.0
    signal = chirp_plus_sinusoid(
        N=N, f_sin=50.0, f_start=10.0, f_end=150.0, fs=fs,
    )

    configs: list[tuple[str, SSD]] = [
        ("Baseline SSD", SSD(fs=fs)),
        ("OptimizedSSD (fwhm)", OptimizedSSD(fs=fs, spectral_method="fwhm")),
        ("OptimizedSSD (moment)", OptimizedSSD(fs=fs, spectral_method="moment")),
        ("OptimizedSSD (gaussian+jac)", OptimizedSSD(fs=fs, spectral_method="gaussian")),
    ]

    lines: list[str] = []
    lines.append("=" * 75)
    lines.append("OPTIMIZED SSD — PROFILING COMPARISON")
    lines.append(f"Signal: chirp_plus_sinusoid, N={N}, fs={fs}")
    lines.append(f"Pipeline: window_len=300, stride=150, max_components=10")
    lines.append("=" * 75)

    for label, engine in configs:
        lines.append("")
        lines.append("-" * 75)
        lines.append(f"Engine: {label}")
        lines.append("-" * 75)

        result = _run_pipeline_instrumented(signal, engine)
        n_win = result["n_windows"]
        decomp = result["decomposition_s"]
        total = result["total_s"]

        lines.append(f"  Windows processed      : {n_win}")
        lines.append(f"  Decomposition time     : {decomp:.4f} s")
        lines.append(f"  Total pipeline time    : {total:.4f} s")

        if decomp > 0:
            bw_pct = 100 * result["bandwidth_s"] / decomp
            svd_pct = 100 * result["svd_s"] / decomp
            sel_pct = 100 * result["eigentriple_sel_s"] / decomp
            rec_pct = 100 * result["reconstruction_s"] / decomp
            oth_pct = 100 * result["other_decomp_s"] / decomp
        else:
            bw_pct = svd_pct = sel_pct = rec_pct = oth_pct = 0.0

        lines.append(f"  Decomposition breakdown:")
        lines.append(f"    Bandwidth estimation : {result['bandwidth_s']:.4f} s  ({bw_pct:.1f}%)")
        lines.append(f"    SVD                  : {result['svd_s']:.4f} s  ({svd_pct:.1f}%)")
        lines.append(f"    Eigentriple selection : {result['eigentriple_sel_s']:.4f} s  ({sel_pct:.1f}%)")
        lines.append(f"    Reconstruction       : {result['reconstruction_s']:.4f} s  ({rec_pct:.1f}%)")
        lines.append(f"    Other                : {result['other_decomp_s']:.4f} s  ({oth_pct:.1f}%)")

        # Re-create engine for memory measurement (since instrumentation
        # modifies the instance).
        if "fwhm" in label:
            mem_engine = OptimizedSSD(fs=fs, spectral_method="fwhm")
        elif "moment" in label:
            mem_engine = OptimizedSSD(fs=fs, spectral_method="moment")
        elif "gaussian" in label:
            mem_engine = OptimizedSSD(fs=fs, spectral_method="gaussian")
        else:
            mem_engine = SSD(fs=fs)

        peak_mb = _measure_peak_memory(signal, mem_engine)
        lines.append(f"  Peak memory            : {peak_mb:.2f} MiB")

    report = "\n".join(lines)
    report_path = out_dir / "optimized_report.txt"
    report_path.write_text(report)
    print(report)
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()