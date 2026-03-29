"""Component-level visualizations for SSD decomposition results.

Provides stacked decomposition plots, trajectory overlays, spectral
analysis panels, and bipartite matching graphs — all using matplotlib.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch


def plot_decomposition(
    signal: np.ndarray,
    components: list[np.ndarray],
    residual: np.ndarray | None = None,
    fs: float = 1.0,
    title: str = "SSD Decomposition",
    save_path: str | None = None,
) -> None:
    """Plot original signal, extracted components, and optional residual.

    Parameters
    ----------
    signal : np.ndarray
        Original input signal.
    components : list[np.ndarray]
        Extracted SSD components (excluding residual).
    residual : np.ndarray or None, optional
        Final residual after all extractions.
    fs : float, optional
        Sampling frequency in Hz for time-axis scaling.  Default 1.0.
    title : str, optional
        Figure super-title.  Default ``"SSD Decomposition"``.
    save_path : str or None, optional
        If given, save figure to this path and close it.

    Notes
    -----
    Layout is stacked subplots sharing the x-axis: original signal on
    top, one row per component, and the residual (if given) at bottom.
    """
    r = len(components)
    n_rows = 1 + r + (1 if residual is not None else 0)
    fig_h = max(6, 2 * n_rows)

    fig, axes = plt.subplots(
        n_rows, 1, figsize=(12, fig_h), sharex=True,
    )
    if n_rows == 1:
        axes = [axes]

    t = np.arange(len(signal)) / fs
    cmap = plt.cm.tab10

    axes[0].plot(t, signal, color="black", linewidth=0.8)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Original")

    for i, comp in enumerate(components):
        ax = axes[1 + i]
        tc = np.arange(len(comp)) / fs
        ax.plot(tc, comp, color=cmap(i % 10), linewidth=0.8)
        ax.set_ylabel("Amplitude")
        ax.set_title(f"Component {i + 1}")

    if residual is not None:
        ax_res = axes[-1]
        tr = np.arange(len(residual)) / fs
        ax_res.plot(tr, residual, color="grey", linewidth=0.8)
        ax_res.set_ylabel("Amplitude")
        ax_res.set_title("Residual")

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_trajectory_overlay(
    trajectory_store: object,
    signal: np.ndarray,
    fs: float = 1.0,
    alpha_signal: float = 0.25,
    save_path: str | None = None,
) -> None:
    """Overlay component trajectories on the original signal.

    Parameters
    ----------
    trajectory_store : TrajectoryStore
        Instance with ``get_all()`` and ``get(i)`` methods.
    signal : np.ndarray
        Original signal for background reference.
    fs : float, optional
        Sampling frequency in Hz.  Default 1.0.
    alpha_signal : float, optional
        Alpha transparency for the signal trace.  Default 0.25.
    save_path : str or None, optional
        If given, save figure to this path and close it.

    Notes
    -----
    NaN gaps in trajectories are naturally skipped by matplotlib.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    t = np.arange(len(signal)) / fs
    ax.plot(t, signal, color="lightgrey", alpha=alpha_signal,
            label="Signal")

    cmap = plt.cm.tab10
    trajs = trajectory_store.get_all()
    for i in sorted(trajs.keys()):
        arr = trajectory_store.get(i)
        t_traj = np.arange(len(arr)) / fs
        ax.plot(t_traj, arr, color=cmap(i % 10), linewidth=0.8,
                label=f"Component {i + 1}")

    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Component Trajectories")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_component_spectra(
    components: list[np.ndarray],
    fs: float = 1.0,
    nperseg: int = 256,
    save_path: str | None = None,
) -> None:
    """Plot Welch PSD for each component with dominant-frequency marker.

    Parameters
    ----------
    components : list[np.ndarray]
        Extracted SSD components.
    fs : float, optional
        Sampling frequency in Hz.  Default 1.0.
    nperseg : int, optional
        Segment length for Welch's method.  Default 256.
    save_path : str or None, optional
        If given, save figure to this path and close it.

    Notes
    -----
    Subplots are arranged in a grid with ``ceil(sqrt(r))`` columns.
    PSD is displayed in dB (``10 * log10(psd + 1e-12)``).
    """
    r = len(components)
    if r == 0:
        return

    n_cols = max(1, math.ceil(math.sqrt(r)))
    n_rows = max(1, math.ceil(r / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows),
    )
    if r == 1:
        axes = np.array([axes])
    axes_flat = np.asarray(axes).ravel()

    cmap = plt.cm.tab10
    for i, comp in enumerate(components):
        ax = axes_flat[i]
        seg = min(nperseg, len(comp))
        freqs, psd = welch(comp, fs=fs, nperseg=seg)
        psd_db = 10.0 * np.log10(psd + 1e-12)
        ax.plot(freqs, psd_db, color=cmap(i % 10))

        f_dom = float(freqs[np.argmax(psd)])
        ax.axvline(f_dom, color="red", linestyle="--", linewidth=0.8)
        ax.annotate(
            f"{f_dom:.1f} Hz",
            xy=(f_dom, psd_db.max()),
            xytext=(5, -5),
            textcoords="offset points",
            fontsize=7,
            color="red",
        )
        ax.set_title(f"Component {i + 1}")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (dB/Hz)")

    for j in range(r, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_matching_graph(
    prev_components: list[np.ndarray],
    curr_components: list[np.ndarray],
    matching: dict[int, int | None],
    overlap: int,
    cost_matrix: np.ndarray | None = None,
    save_path: str | None = None,
) -> None:
    """Draw a bipartite matching graph between window components.

    Parameters
    ----------
    prev_components : list[np.ndarray]
        Components from the previous window.
    curr_components : list[np.ndarray]
        Components from the current window.
    matching : dict[int, int or None]
        Mapping ``{curr_idx: prev_idx}``.  ``None`` = unmatched.
    overlap : int
        Overlap in samples between the two windows.
    cost_matrix : np.ndarray or None, optional
        If given, confidence labels are drawn on edges.
    save_path : str or None, optional
        If given, save figure to this path and close it.

    Notes
    -----
    Nodes are drawn as circles with matplotlib ``Circle`` patches.
    """
    from matplotlib.patches import Circle

    n_prev = len(prev_components)
    n_curr = len(curr_components)
    n_max = max(n_prev, n_curr, 1)

    fig, ax = plt.subplots(figsize=(8, max(4, n_max * 1.2)))
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, n_max + 0.5)
    ax.set_aspect("equal")
    ax.set_title(
        f"Component Matching (overlap={overlap} samples)",
    )
    ax.set_xticks([])
    ax.set_yticks([])

    radius = 0.25
    left_x, right_x = 0.5, 2.5

    prev_positions = {}
    for j in range(n_prev):
        y = n_max - j - 0.5
        circ = Circle(
            (left_x, y), radius, fill=False,
            edgecolor="steelblue", linewidth=1.5,
        )
        ax.add_patch(circ)
        ax.text(
            left_x, y, f"Prev {j}", ha="center", va="center",
            fontsize=8,
        )
        prev_positions[j] = (left_x, y)

    curr_positions = {}
    for i in range(n_curr):
        y = n_max - i - 0.5
        is_unmatched = matching.get(i) is None
        edge_color = "red" if is_unmatched else "darkorange"
        circ = Circle(
            (right_x, y), radius, fill=False,
            edgecolor=edge_color, linewidth=1.5,
        )
        ax.add_patch(circ)
        label_color = "red" if is_unmatched else "black"
        ax.text(
            right_x, y, f"Curr {i}", ha="center", va="center",
            fontsize=8, color=label_color,
        )
        curr_positions[i] = (right_x, y)

    for curr_i, prev_j in matching.items():
        if prev_j is None:
            continue
        if prev_j not in prev_positions:
            continue
        x0, y0 = prev_positions[prev_j]
        x1, y1 = curr_positions[curr_i]
        ax.plot(
            [x0 + radius, x1 - radius], [y0, y1],
            color="grey", linewidth=1.2,
        )
        if cost_matrix is not None:
            conf = 1.0 - cost_matrix[curr_i, prev_j]
            mid_x = (x0 + x1) / 2
            mid_y = (y0 + y1) / 2
            ax.text(
                mid_x, mid_y + 0.15, f"{conf:.2f}",
                ha="center", fontsize=7, color="green",
            )

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
