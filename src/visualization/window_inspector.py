"""Per-window reconstruction inspection plots.

Provides three visualisation helpers for the streaming SSD pipeline:

* **plot_window_reconstruction** — side-by-side original vs
  reconstruction for a single analysis window.
* **plot_window_grid** — compact multi-window summary grid.
* **plot_nmse_over_time** — NMSE sampled every second over the
  full signal duration.

All metrics follow the definitions in Bonizzi et al. (2014) and
Harmouche et al. (2017).
"""

from __future__ import annotations

import math

import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------------
# Internal metric helpers
# ------------------------------------------------------------------

def _compute_qrf(
    original: np.ndarray,
    reconstruction: np.ndarray,
) -> float:
    """Quality Reconstruction Factor (Harmouche 2017, Eq. 7).

    Parameters
    ----------
    original : np.ndarray
        Reference signal segment.
    reconstruction : np.ndarray
        Reconstructed signal segment.

    Returns
    -------
    float
        QRF in dB.  ``+inf`` when the residual norm is < 1e-12.
    """
    resid_norm = np.linalg.norm(original - reconstruction)
    if resid_norm < 1e-12:
        return float("inf")
    return 20.0 * np.log10(
        np.linalg.norm(original) / resid_norm
    )


def _compute_nmse(
    original: np.ndarray,
    reconstruction: np.ndarray,
) -> float:
    """Normalised Mean Square Error (Bonizzi 2014, Sec. 3.4).

    Parameters
    ----------
    original : np.ndarray
        Reference signal segment.
    reconstruction : np.ndarray
        Reconstructed signal segment.

    Returns
    -------
    float
        NMSE value, or ``NaN`` for silent segments.
    """
    denom = np.dot(original, original)
    if denom < 1e-12:
        return float("nan")
    resid = original - reconstruction
    return float(np.dot(resid, resid) / denom)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def plot_window_reconstruction(
    window_signal: np.ndarray,
    components: list[np.ndarray],
    window_idx: int,
    sample_start: int,
    fs: float = 1.0,
    save_path: str | None = None,
) -> None:
    """Plot original vs reconstruction for a single window.

    Parameters
    ----------
    window_signal : np.ndarray
        Raw signal segment for this window.
    components : list[np.ndarray]
        Extracted signal components (residual excluded).
    window_idx : int
        Window sequence number (used in the title).
    sample_start : int
        Global sample index of the first sample in the window.
    fs : float, optional
        Sampling frequency in Hz (default 1.0).
    save_path : str or None, optional
        If given, save the figure to this path and close it;
        otherwise call ``plt.show()``.

    Notes
    -----
    QRF and NMSE are computed internally and displayed in the
    title of the upper subplot.
    """
    recon = (
        np.sum(components, axis=0)
        if len(components) > 0
        else np.zeros_like(window_signal)
    )
    residual = window_signal - recon
    qrf_val = _compute_qrf(window_signal, recon)
    nmse_val = _compute_nmse(window_signal, recon)

    time_ax = (
        sample_start + np.arange(len(window_signal))
    ) / fs

    fig, axes = plt.subplots(
        2, 1, figsize=(12, 5), sharex=True,
    )

    ax0 = axes[0]
    ax0.plot(
        time_ax, window_signal,
        color="black", linewidth=0.8, label="Original",
    )
    ax0.plot(
        time_ax, recon,
        color="red", linestyle="--", linewidth=0.9,
        label="Reconstruction", alpha=0.85,
    )
    ax0.set_title(
        f"Window {window_idx}  |  "
        f"QRF = {qrf_val:.1f} dB  |  "
        f"NMSE = {nmse_val:.4f}"
    )
    ax0.set_ylabel("Amplitude")
    ax0.legend(loc="upper right", fontsize=8)

    ax1 = axes[1]
    ax1.fill_between(
        time_ax, residual,
        color="grey", linewidth=0.6, label="Residual",
    )
    ax1.set_ylabel("Residual")
    ax1.set_xlabel("Time (s)")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_window_grid(
    pipeline_records: list[dict],
    n_windows: int = 9,
    fs: float = 1.0,
    save_path: str | None = None,
) -> None:
    """Compact grid of original-vs-reconstruction subplots.

    Parameters
    ----------
    pipeline_records : list[dict]
        One dict per processed window with keys
        ``window_idx``, ``sample_start``, ``window_signal``,
        and ``components``.
    n_windows : int, optional
        Number of windows to display (default 9).
    fs : float, optional
        Sampling frequency in Hz (default 1.0).
    save_path : str or None, optional
        If given, save and close; otherwise ``plt.show()``.

    Notes
    -----
    Windows are selected with even spacing across the full
    record list using ``np.linspace``.
    """
    n_records = len(pipeline_records)
    n_show = min(n_windows, n_records)
    indices = np.linspace(
        0, n_records - 1, n_show, dtype=int,
    )

    ncols = min(3, n_show)
    nrows = math.ceil(n_show / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4 * ncols, 3 * nrows),
        squeeze=False,
    )

    for ax in axes.flat:
        ax.set_visible(False)

    for i, rec_idx in enumerate(indices):
        rec = pipeline_records[rec_idx]
        row, col = divmod(i, ncols)
        ax = axes[row][col]
        ax.set_visible(True)

        sig = rec["window_signal"]
        comps = rec["components"]
        recon = (
            np.sum(comps, axis=0)
            if len(comps) > 0
            else np.zeros_like(sig)
        )
        qrf_val = _compute_qrf(sig, recon)

        ax.plot(sig, color="black", linewidth=0.6)
        ax.plot(
            recon, color="red",
            linestyle="--", linewidth=0.7,
        )
        ax.set_title(
            f"W{rec['window_idx']}  "
            f"QRF={qrf_val:.1f}dB",
            fontsize=7,
        )

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color="black", linewidth=0.8,
               label="Original"),
        Line2D([0], [0], color="red", linestyle="--",
               linewidth=0.8, label="Reconstruction"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=8,
    )

    fig.suptitle(
        "Reconstruction Quality Across Windows",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_nmse_over_time(
    signal: np.ndarray,
    reconstruction: np.ndarray,
    fs: float = 1.0,
    save_path: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Plot per-second NMSE over the signal duration.

    Parameters
    ----------
    signal : np.ndarray
        Original 1-D signal.
    reconstruction : np.ndarray
        Reconstructed signal (same length as *signal*).
        ``NaN`` values indicate missing reconstruction and are
        treated as zero when computing the residual.
    fs : float, optional
        Sampling frequency in Hz (default 1.0).
    save_path : str or None, optional
        If given, save and close; otherwise ``plt.show()``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(time_axis, nmse_values)`` arrays with one entry per
        complete second.

    Notes
    -----
    NMSE follows Bonizzi et al. (2014), Sec. 3.4, computed on
    non-overlapping one-second segments.  Seconds that contain
    NaN in the reconstruction are flagged with a different
    marker on the plot.
    """
    samples_per_sec = int(round(fs))
    n_seconds = len(signal) // samples_per_sec

    times = np.empty(n_seconds, dtype=np.float64)
    nmse_vals = np.empty(n_seconds, dtype=np.float64)
    has_nan_flags = np.zeros(n_seconds, dtype=bool)

    for k in range(n_seconds):
        a = k * samples_per_sec
        b = a + samples_per_sec
        seg_orig = signal[a:b]
        seg_recon_raw = reconstruction[a:b]

        has_nan_flags[k] = np.any(np.isnan(seg_recon_raw))
        seg_recon = np.nan_to_num(seg_recon_raw, nan=0.0)

        denom = np.dot(seg_orig, seg_orig)
        if denom < 1e-12:
            nmse_vals[k] = np.nan
        else:
            resid = seg_orig - seg_recon
            nmse_vals[k] = np.dot(resid, resid) / denom

        times[k] = k + 0.5

    # --- plotting ---
    fig, ax = plt.subplots(figsize=(12, 4))

    valid_mask = np.isfinite(nmse_vals)
    nan_mask = ~valid_mask

    ax.plot(
        times, nmse_vals,
        color="blue", linewidth=1.2,
    )

    nan_recon_mask = has_nan_flags & valid_mask
    if np.any(nan_recon_mask):
        ax.plot(
            times[nan_recon_mask],
            nmse_vals[nan_recon_mask],
            "D", color="orange", markersize=5,
            label="NaN in recon", linestyle="none",
        )

    if np.any(nan_mask):
        ax.plot(
            times[nan_mask],
            np.zeros(np.sum(nan_mask)),
            "x", color="grey", markersize=5,
            label="Silent segment", linestyle="none",
        )

    ax.axhline(
        1.0, color="red", linestyle="--",
        alpha=0.5, label="NMSE = 1.0 baseline",
    )
    ax.axhline(
        0.0, color="green", linestyle="--",
        alpha=0.3, label="Perfect reconstruction",
    )

    valid_nmse = nmse_vals[valid_mask]
    if len(valid_nmse) > 0:
        y_lo = max(0, float(np.min(valid_nmse)) - 0.05)
        ax.set_ylim(bottom=y_lo)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("NMSE")
    ax.set_title(
        "Reconstruction NMSE Over Time "
        "(sampled every 1 s)"
    )
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return times, nmse_vals
