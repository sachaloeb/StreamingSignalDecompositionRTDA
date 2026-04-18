"""NMSE threshold sensitivity comparison: standard SSD vs streaming SSD.

Runs both decomposition modes across a range of NMSE stopping thresholds
and produces a single figure showing how component count varies with
threshold for each mode.

Usage
-----
    python scripts/plot_nmse_threshold_comparison.py [--show]

Outputs
-------
    results/nmse_threshold_comparison/nmse_comparison.png  (dpi=200)

The ``--show`` flag additionally calls ``plt.show()`` after saving.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ── path setup (identical to run_experiment.py) ───────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.synthetic.generators import chirp_plus_sinusoid
from src.engines.ssd import SSD
from src.streaming.component_matcher import ComponentMatcher
from src.streaming.trajectory_store import TrajectoryStore
from src.streaming.window_manager import WindowManager

# ── signal constants ──────────────────────────────────────────────────────
FS: float = 1000.0
N: int = 3000
F_SIN: float = 50.0
F_START: float = 10.0
F_END: float = 150.0
SNR_DB: float = 20.0
SEED: int = 42

# ── pipeline constants ────────────────────────────────────────────────────
WINDOW_LEN: int = 300
STRIDE: int = 150
MAX_ITER: int = 20
MAX_COMPONENTS: int = 20   # high ceiling; not artificially capped

LOOKBACK: int = 10
MAX_COST: float = 0.10

# ── threshold sweep ───────────────────────────────────────────────────────
NMSE_THRESHOLDS: list[float] = [
    0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50,
]

# ── output ────────────────────────────────────────────────────────────────
OUTPUT_DIR: Path = ROOT / "results" / "nmse_threshold_comparison"


# ── matplotlib style (matches repo conventions) ───────────────────────────
plt.rcParams.update({
    "font.family":          "DejaVu Sans",
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.titlesize":       10,
    "axes.labelsize":       9,
    "xtick.labelsize":      8,
    "ytick.labelsize":      8,
})


# ─────────────────────────────────────────────────────────────────────────
# core functions
# ─────────────────────────────────────────────────────────────────────────

def run_standard_ssd(
    signal: np.ndarray,
    threshold: float,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Run offline SSD on the full signal.

    Parameters
    ----------
    signal : np.ndarray
        Input signal of length N.
    threshold : float
        NMSE stopping threshold passed to ``SSD``.

    Returns
    -------
    tuple[list[np.ndarray], np.ndarray]
        ``(components, residual)`` where *components* excludes the
        residual and *residual* is the final SSD residual.
    """
    engine = SSD(fs=FS, nmse_threshold=threshold, max_iter=MAX_ITER)
    result = engine.fit(signal)
    components = result[:-1]
    residual = result[-1]
    return components, residual


def run_streaming_ssd(
    signal: np.ndarray,
    threshold: float,
) -> dict[int, np.ndarray]:
    """Run the streaming SSD pipeline on *signal*.

    Parameters
    ----------
    signal : np.ndarray
        Input signal of length N, streamed sample-by-sample.
    threshold : float
        NMSE stopping threshold passed to ``SSD``.

    Returns
    -------
    dict[int, np.ndarray]
        All trajectory arrays from ``TrajectoryStore.get_all()``.
        NaN gaps are present for positions never written.

    Notes
    -----
    Pipeline parameters are taken from the module-level constants
    ``WINDOW_LEN``, ``STRIDE``, ``MAX_COMPONENTS``, ``LOOKBACK``,
    and ``MAX_COST``.
    """
    wm = WindowManager(window_len=WINDOW_LEN, stride=STRIDE, fs=FS)
    engine = SSD(fs=FS, nmse_threshold=threshold, max_iter=MAX_ITER)
    matcher = ComponentMatcher(
        distance="d_freq",
        freq_weight=1.0,
        fs=FS,
        lookback=LOOKBACK,
        max_cost=MAX_COST,
        max_trajectories=MAX_COMPONENTS,
    )
    store = TrajectoryStore(max_components=MAX_COMPONENTS, max_len=N)

    for t, sample in enumerate(signal):
        window = wm.push(float(sample))
        if window is None:
            continue
        components = engine.fit(window)
        components_no_res = components[:-1]
        matching = dict(matcher.match_stateful(components_no_res, wm.overlap))
        window_start = t - wm.window_len + 1
        store.update(window_start, components_no_res, matching, wm.overlap)

    return store.get_all()


def sweep_thresholds(
    signal: np.ndarray,
    thresholds: list[float],
) -> tuple[list[int], list[int]]:
    """Run both modes across all thresholds and collect component counts.

    Parameters
    ----------
    signal : np.ndarray
        Input signal used for all runs.
    thresholds : list[float]
        NMSE threshold values to sweep.

    Returns
    -------
    tuple[list[int], list[int]]
        ``(std_counts, stream_counts)`` — number of components produced
        by standard SSD and streaming SSD at each threshold.
    """
    std_counts: list[int] = []
    stream_counts: list[int] = []

    for thr in thresholds:
        # ── standard SSD ──────────────────────────────────────────────
        std_comps, std_residual = run_standard_ssd(signal, thr)
        n_std = len(std_comps)
        std_counts.append(n_std)

        # Reconstruction identity check
        if n_std > 0:
            recon = np.sum(std_comps, axis=0) + std_residual
        else:
            recon = std_residual
        ok = np.allclose(recon, signal, atol=1e-10)
        print(
            f"  threshold={thr:.3f} | std SSD: {n_std} comps | "
            f"recon check: {'PASS' if ok else 'FAIL'}"
        )

        # ── streaming SSD ─────────────────────────────────────────────
        trajs = run_streaming_ssd(signal, thr)
        n_stream = len(trajs)
        if n_stream == 0:
            warnings.warn(
                f"Streaming SSD returned zero trajectories at "
                f"threshold={thr:.3f}. Check signal or pipeline config.",
                RuntimeWarning,
                stacklevel=2,
            )
        stream_counts.append(n_stream)
        print(
            f"  threshold={thr:.3f} | streaming SSD: {n_stream} trajectories"
        )

    return std_counts, stream_counts


def make_figure(
    thresholds: list[float],
    std_counts: list[int],
    stream_counts: list[int],
    save_path: Path,
    show: bool = False,
) -> None:
    """Produce and save the comparison figure.

    Parameters
    ----------
    thresholds : list[float]
        NMSE threshold values (x-axis).
    std_counts : list[int]
        Component counts from standard SSD.
    stream_counts : list[int]
        Trajectory counts from streaming SSD.
    save_path : Path
        Output file path (PNG).
    show : bool, optional
        If ``True``, call ``plt.show()`` after saving.  Default ``False``.

    Notes
    -----
    Both curves share the same axes.  Markers are placed at each
    evaluated threshold to make individual data points identifiable.
    """
    x = np.array(thresholds)
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(
        x, std_counts,
        color=plt.cm.tab10(0),
        linewidth=1.4,
        marker="o",
        markersize=5,
        label="Standard SSD (offline, full signal)",
    )
    ax.plot(
        x, stream_counts,
        color=plt.cm.tab10(1),
        linewidth=1.4,
        marker="s",
        markersize=5,
        label=f"Streaming SSD (window={WINDOW_LEN}, stride={STRIDE})",
    )

    ax.set_xscale("log")
    ax.set_xlabel("NMSE threshold")
    ax.set_ylabel("Number of components / trajectories")
    ax.set_title(
        "Component count vs NMSE threshold\n"
        f"chirp+sinusoid  N={N}, fs={FS:.0f} Hz, SNR={SNR_DB} dB"
    )

    # label each x-tick with the actual threshold value
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in thresholds], rotation=30, ha="right")

    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(fontsize=8)
    fig.tight_layout()

    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {save_path}")

    if show:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
# entry point
# ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Parse CLI arguments, run the sweep, and save the figure."""
    parser = argparse.ArgumentParser(
        description="Compare standard vs streaming SSD component counts "
                    "across NMSE thresholds."
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Call plt.show() after saving.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating signal …")
    signal = chirp_plus_sinusoid(
        N=N, f_sin=F_SIN, f_start=F_START, f_end=F_END,
        fs=FS, snr_db=SNR_DB, seed=SEED,
    )
    print(f"Signal shape: {signal.shape}, energy: {np.dot(signal, signal):.2f}\n")

    print("Sweeping NMSE thresholds …")
    std_counts, stream_counts = sweep_thresholds(signal, NMSE_THRESHOLDS)

    print("\nSummary:")
    print(f"  {'threshold':>12}  {'std_SSD':>8}  {'streaming':>10}")
    for thr, sc, stc in zip(NMSE_THRESHOLDS, std_counts, stream_counts):
        print(f"  {thr:>12.3f}  {sc:>8d}  {stc:>10d}")

    png_path = OUTPUT_DIR / "nmse_comparison.png"
    make_figure(NMSE_THRESHOLDS, std_counts, stream_counts, png_path, show=args.show)


if __name__ == "__main__":
    main()