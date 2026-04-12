"""Profile the streaming SSD pipeline.

Measures wall-clock time breakdown and peak memory for the baseline
pipeline, rSVD-SSD, and IncrementalSSD on a chirp_plus_sinusoid signal.

Usage
-----
    python experiments/profile_pipeline.py
"""

from __future__ import annotations

import cProfile
import io
import pstats
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
from src.engines.ssa import build_trajectory_matrix, svd_decompose
from src.streaming.component_matcher import ComponentMatcher
from src.streaming.trajectory_store import TrajectoryStore
from src.streaming.window_manager import WindowManager


def _run_pipeline(
    signal: np.ndarray,
    engine_name: str,
    fs: float = 1000.0,
    window_len: int = 300,
    stride: int = 150,
    max_components: int = 10,
    **engine_kwargs: object,
) -> dict[str, float]:
    """Run the streaming pipeline and return timing breakdown."""
    N = len(signal)
    wm = WindowManager(window_len=window_len, stride=stride, fs=fs)
    # ssd_rank1 requires stride; pass it through if not already supplied
    if engine_name == "ssd_rank1" and "stride" not in engine_kwargs:
        engine_kwargs = {**engine_kwargs, "stride": stride}
    engine = get_engine(engine_name, fs=fs, **engine_kwargs)
    matcher = ComponentMatcher(
        distance="d_corr", fs=fs, lookback=10,
        max_cost=0.6, max_trajectories=max_components,
    )
    store = TrajectoryStore(max_components=max_components, max_len=N)

    t_decompose = 0.0
    t_match = 0.0
    t_store = 0.0
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

        t0 = time.perf_counter()
        matching = dict(
            matcher.match_stateful(components_no_res, wm.overlap)
        )
        t1 = time.perf_counter()
        t_match += t1 - t0

        window_start = sample_idx - wm.window_len + 1
        t0 = time.perf_counter()
        store.update(
            window_start, components_no_res, matching, wm.overlap,
        )
        t1 = time.perf_counter()
        t_store += t1 - t0

        n_windows += 1

    return {
        "engine": engine_name,
        "n_windows": n_windows,
        "decomposition_s": t_decompose,
        "matching_s": t_match,
        "trajectory_store_s": t_store,
        "total_s": t_decompose + t_match + t_store,
    }


def _profile_with_cprofile(
    signal: np.ndarray,
    engine_name: str,
    **engine_kwargs: object,
) -> str:
    """Run under cProfile and return formatted stats."""
    pr = cProfile.Profile()
    pr.enable()
    _run_pipeline(signal, engine_name, **engine_kwargs)
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(30)
    return s.getvalue()


def _measure_peak_memory(
    signal: np.ndarray,
    engine_name: str,
    **engine_kwargs: object,
) -> float:
    """Measure peak memory in MiB."""
    tracemalloc.start()
    _run_pipeline(signal, engine_name, **engine_kwargs)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024 * 1024)


def main() -> None:
    """Run profiling for all engine variants."""
    out_dir = ROOT / "results" / "profiling"
    out_dir.mkdir(parents=True, exist_ok=True)

    N = 10000
    fs = 1000.0
    signal = chirp_plus_sinusoid(
        N=N, f_sin=50.0, f_start=10.0, f_end=150.0, fs=fs,
    )

    configs = [
        ("ssd", {}),
        ("ssd_incremental", {"use_rsvd": True}),
        ("ssd_incremental", {}),
        ("ssd_rank1", {}),
    ]
    labels = ["Baseline SSD", "rSVD-SSD (IncrementalSSD)", "IncrementalSSD (full SVD)", "Rank-1 IncrementalSSD"]

    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("STREAMING SSD PIPELINE — PROFILING REPORT")
    lines.append(f"Signal: chirp_plus_sinusoid, N={N}, fs={fs}")
    lines.append("=" * 70)

    for (engine_name, kwargs), label in zip(configs, labels):
        lines.append("")
        lines.append("-" * 70)
        lines.append(f"Engine: {label} ({engine_name})")
        lines.append("-" * 70)

        # Timing breakdown
        timings = _run_pipeline(signal, engine_name, **kwargs)
        lines.append(f"  Windows processed : {timings['n_windows']}")
        lines.append(f"  Decomposition     : {timings['decomposition_s']:.4f} s")
        lines.append(f"  Component matching: {timings['matching_s']:.4f} s")
        lines.append(f"  Trajectory store  : {timings['trajectory_store_s']:.4f} s")
        lines.append(f"  Total             : {timings['total_s']:.4f} s")

        if timings['total_s'] > 0:
            pct_decomp = 100 * timings['decomposition_s'] / timings['total_s']
            pct_match = 100 * timings['matching_s'] / timings['total_s']
            pct_store = 100 * timings['trajectory_store_s'] / timings['total_s']
            lines.append(f"  Decomposition %   : {pct_decomp:.1f}%")
            lines.append(f"  Matching %        : {pct_match:.1f}%")
            lines.append(f"  Traj. store %     : {pct_store:.1f}%")

        # Peak memory
        peak_mb = _measure_peak_memory(signal, engine_name, **kwargs)
        lines.append(f"  Peak memory       : {peak_mb:.2f} MiB")

        # cProfile top functions
        lines.append("")
        lines.append("  cProfile (top 30 cumulative):")
        cprofile_output = _profile_with_cprofile(
            signal, engine_name, **kwargs,
        )
        for line in cprofile_output.split("\n"):
            lines.append(f"    {line}")

    report = "\n".join(lines)
    report_path = out_dir / "profile_report.txt"
    report_path.write_text(report)
    print(f"Report saved to {report_path}")
    print(report)


if __name__ == "__main__":
    main()