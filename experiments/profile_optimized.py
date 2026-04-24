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

    # Wrap eigentriple selection (use engine's own method to capture overrides)
    orig_sel = engine._select_eigentriples  # type: ignore[attr-defined]
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

    window_decomp_times: list[float] = []
    window_bw_times: list[float] = []
    window_svd_times: list[float] = []
    window_sel_times: list[float] = []
    window_recon_times: list[float] = []

    t_total_start = time.perf_counter()

    for sample_idx in range(N):
        window = wm.push(float(signal[sample_idx]))
        if window is None:
            continue

        # Snapshot sub-timing list lengths before this window's fit()
        bw_before = len(bw_times)
        svd_before = len(svd_times)
        sel_before = len(select_times)
        recon_before = len(recon_times)

        t0 = time.perf_counter()
        components = engine.fit(window)
        t1 = time.perf_counter()

        window_decomp_times.append(t1 - t0)
        window_bw_times.append(sum(bw_times[bw_before:]))
        window_svd_times.append(sum(svd_times[svd_before:]))
        window_sel_times.append(sum(select_times[sel_before:]))
        window_recon_times.append(sum(recon_times[recon_before:]))

        components_no_res = components[:-1]
        matching = dict(
            matcher.match_stateful(components_no_res, wm.overlap)
        )
        window_start = sample_idx - wm.window_len + 1
        store.update(window_start, components_no_res, matching, wm.overlap)

    t_total = time.perf_counter() - t_total_start

    def _stats(vals: list[float]) -> dict[str, float]:
        a = np.array(vals)
        return {
            "total": float(a.sum()),
            "mean": float(a.mean()),
            "std": float(a.std()),
            "min": float(a.min()),
            "max": float(a.max()),
            "p95": float(np.percentile(a, 95)),
        }

    decomp_stats = _stats(window_decomp_times)
    other_per_win = [
        max(0.0, d - bw - sv - sl - rc)
        for d, bw, sv, sl, rc in zip(
            window_decomp_times, window_bw_times, window_svd_times,
            window_sel_times, window_recon_times,
        )
    ]

    return {
        "n_windows": len(window_decomp_times),
        "decomposition_s": decomp_stats["total"],
        "total_s": t_total,
        "decomp_per_window": decomp_stats,
        "bandwidth_s": sum(window_bw_times),
        "bandwidth_per_window": _stats(window_bw_times),
        "svd_s": sum(window_svd_times),
        "svd_per_window": _stats(window_svd_times),
        "eigentriple_sel_s": sum(window_sel_times),
        "eigentriple_per_window": _stats(window_sel_times),
        "reconstruction_s": sum(window_recon_times),
        "reconstruction_per_window": _stats(window_recon_times),
        "other_decomp_s": sum(other_per_win),
        "other_per_window": _stats(other_per_win),
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
        N=N, f_sin=50.0, f_start=10.0, f_end=150.0, fs=fs, snr_db=5.0,
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
    lines.append(f"Signal: chirp_plus_sinusoid, N={N}, fs={fs}, SNR=5 dB")
    lines.append(f"Pipeline: window_len=300, stride=150, max_components=10")
    lines.append("=" * 75)

    for label, engine in configs:
        lines.append("")
        lines.append("-" * 75)
        lines.append(f"Engine: {label}")
        lines.append("-" * 75)

        result = _run_pipeline_instrumented(signal, engine, window_len=3000, stride=1500)
        n_win = result["n_windows"]
        decomp = result["decomposition_s"]
        total = result["total_s"]

        pw = result["decomp_per_window"]
        budget_ms = 1500 / fs * 1000  # stride / fs in ms

        lines.append(f"  Windows processed      : {n_win}")
        lines.append(f"  Decomposition time     : {decomp:.4f} s")
        lines.append(f"  Total pipeline time    : {total:.4f} s")
        lines.append(f"  Real-time budget/window: {budget_ms:.1f} ms  (stride={150}/fs={fs:.0f})")
        lines.append(f"")
        lines.append(f"  Per-window decomposition (ms):")
        lines.append(f"    mean ± std : {pw['mean']*1e3:6.2f} ± {pw['std']*1e3:.2f}")
        lines.append(f"    min / max  : {pw['min']*1e3:6.2f} / {pw['max']*1e3:.2f}")
        lines.append(f"    p95        : {pw['p95']*1e3:6.2f}  ({'OK' if pw['p95']*1e3 < budget_ms else 'OVER BUDGET'})")
        lines.append(f"")

        if decomp > 0:
            bw_pct = 100 * result["bandwidth_s"] / decomp
            svd_pct = 100 * result["svd_s"] / decomp
            sel_pct = 100 * result["eigentriple_sel_s"] / decomp
            rec_pct = 100 * result["reconstruction_s"] / decomp
            oth_pct = 100 * result["other_decomp_s"] / decomp
        else:
            bw_pct = svd_pct = sel_pct = rec_pct = oth_pct = 0.0

        def _pw_line(label: str, key: str, pct: float) -> str:
            s = result[key]
            return (
                f"    {label:<22}: "
                f"mean={s['mean']*1e3:5.2f} ms  "
                f"p95={s['p95']*1e3:5.2f} ms  "
                f"({pct:.1f}% of decomp)"
            )

        lines.append(f"  Decomposition breakdown (per-window mean, p95):")
        lines.append(_pw_line("Bandwidth estimation", "bandwidth_per_window", bw_pct))
        lines.append(_pw_line("SVD", "svd_per_window", svd_pct))
        lines.append(_pw_line("Eigentriple selection", "eigentriple_per_window", sel_pct))
        lines.append(_pw_line("Reconstruction", "reconstruction_per_window", rec_pct))
        lines.append(_pw_line("Other", "other_per_window", oth_pct))

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