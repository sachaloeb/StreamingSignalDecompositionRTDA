"""Plot streaming SSA metrics from a results directory.

Produces a 3-panel figure:
  1. Singular Value Drift over Windows  (cross-window)
  2. Dominant Frequency per Component   (per-window)
  3. Post-Hoc freq_drift Aggregate      (bar chart)

Usage
-----
    python src/visualization/plot_metrics.py results/demo_run
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# 1. Data loading
# ------------------------------------------------------------------


def _load_data(
    results_dir: Path,
) -> tuple[pd.DataFrame, dict]:
    """Load metrics.csv and run_summary.json from *results_dir*.

    Parameters
    ----------
    results_dir : Path
        Directory containing ``metrics.csv`` and optionally
        ``run_summary.json``.

    Returns
    -------
    df : pd.DataFrame
        The metrics dataframe.
    summary : dict
        Contents of ``run_summary.json``, or ``{}`` if absent.

    Raises
    ------
    FileNotFoundError
        If ``metrics.csv`` does not exist.
    ValueError
        If ``window_index`` column is missing.
    """
    csv_path = results_dir / "metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"metrics.csv not found in {results_dir}"
        )
    df = pd.read_csv(csv_path)
    if "window_index" not in df.columns:
        raise ValueError(
            "'window_index' column missing from metrics.csv"
        )
    df["window_index"] = df["window_index"].astype(int)

    summary: dict = {}
    json_path = results_dir / "run_summary.json"
    if json_path.exists():
        with open(json_path) as fh:
            summary = json.load(fh)
    else:
        logger.warning(
            "run_summary.json not found in %s", results_dir
        )
    return df, summary


# ------------------------------------------------------------------
# 2. Panel 1: Singular Value Drift
# ------------------------------------------------------------------


def _plot_sv_drift(
    ax: plt.Axes,
    df: pd.DataFrame,
) -> None:
    """Render panel 1: singular_value_drift over windows.

    Parameters
    ----------
    ax : matplotlib Axes
    df : pd.DataFrame
        Must contain ``window_index`` and optionally
        ``singular_value_drift``.
    """
    if "singular_value_drift" not in df.columns:
        ax.text(
            0.5, 0.5,
            "Column 'singular_value_drift' not found in metrics.csv",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=9, color="gray", fontstyle="italic",
        )
        ax.set_title(
            "Singular Value Drift  "
            "(cross-window, NaN at t=0 by design)"
        )
        return

    sub = df[df["window_index"] >= 1].copy()
    wins = sub["window_index"].values
    vals = pd.to_numeric(
        sub["singular_value_drift"], errors="coerce"
    ).values.astype(float)
    finite = np.isfinite(vals)

    if not finite.any():
        ax.text(
            0.5, 0.5, "No data available",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=9, color="gray", fontstyle="italic",
        )
    else:
        ax.plot(
            wins[finite], vals[finite],
            color="#2d6a9f", linewidth=1.5,
        )
        if finite.sum() >= 2:
            mu = float(np.nanmean(vals[finite]))
            sd = float(np.nanstd(vals[finite]))
            ax.axhspan(
                mu - sd, mu + sd,
                color="#2d6a9f", alpha=0.15,
            )

    ax.axhline(
        0, color="gray", linewidth=0.8,
        linestyle="--", alpha=0.5,
    )
    ax.set_ylabel("‖S_t − S_{t-1}‖_F")
    ax.set_title(
        "Singular Value Drift  "
        "(cross-window, NaN at t=0 by design)"
    )
    ax.grid(axis="y", alpha=0.3, linestyle=":")


# ------------------------------------------------------------------
# 3. Panel 2: Dominant Frequency Trajectories
# ------------------------------------------------------------------


def _plot_freq_trajectories(
    ax: plt.Axes,
    df: pd.DataFrame,
) -> None:
    """Render panel 2: f_max_cK columns over all windows.

    Parameters
    ----------
    ax : matplotlib Axes
    df : pd.DataFrame
        Must contain ``window_index`` and zero or more
        ``f_max_cK`` columns.
    """
    comp_cols = sorted(
        [c for c in df.columns
         if re.fullmatch(r"f_max_c\d+", c)],
        key=lambda c: int(c.split("f_max_c")[1]),
    )

    if not comp_cols:
        ax.text(
            0.5, 0.5, "No f_max columns found",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=9, color="gray", fontstyle="italic",
        )
        ax.set_title(
            "Dominant Frequency per Component over Windows"
        )
        ax.set_xlabel("Window index")
        ax.set_ylabel("Dominant frequency (Hz)")
        return

    wins = df["window_index"].values
    n = len(df)
    step = max(1, n // 40)
    cmap = plt.cm.tab10

    for i, col in enumerate(comp_cols):
        k = int(col.split("f_max_c")[1])
        vals = pd.to_numeric(
            df[col], errors="coerce"
        ).values.astype(float)
        ax.plot(
            wins, vals,
            color=cmap(k % 10),
            label=f"Component {k}",
            marker="o", markersize=3,
            markevery=step, linewidth=1.0,
        )

    ax.set_xlabel("Window index")
    ax.set_ylabel("Dominant frequency (Hz)")
    ax.grid(axis="both", alpha=0.3, linestyle=":")

    if len(comp_cols) == 1:
        k0 = int(comp_cols[0].split("f_max_c")[1])
        ax.set_title(
            "Dominant Frequency over Windows  "
            f"(Component {k0})"
        )
    else:
        ax.set_title(
            "Dominant Frequency per Component over Windows"
        )
        ax.legend(
            loc="upper right", fontsize=8, framealpha=0.7,
        )


# ------------------------------------------------------------------
# 4. Panel 3: Post-Hoc freq_drift Aggregate
# ------------------------------------------------------------------


def _plot_freq_drift_bar(
    ax: plt.Axes,
    summary: dict,
) -> None:
    """Render panel 3: horizontal bar chart of freq_drift_cK.

    Parameters
    ----------
    ax : matplotlib Axes
    summary : dict
        Contents of ``run_summary.json``.
    """
    drift_keys = sorted(
        [k for k in summary
         if re.fullmatch(r"freq_drift_c\d+", k)],
        key=lambda k: int(k.split("freq_drift_c")[1]),
    )

    if not drift_keys:
        ax.text(
            0.5, 0.5,
            "run_summary.json not found or contains no\n"
            "freq_drift entries — run the experiment first.",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=9, color="gray", fontstyle="italic",
        )
        ax.set_title(
            "Post-Hoc freq_drift Aggregate  "
            "(global Var_t[f_max] per component)"
        )
        return

    labels: list[str] = []
    values: list[float] = []
    k_indices: list[int] = []
    for key in drift_keys:
        k = int(key.split("freq_drift_c")[1])
        k_indices.append(k)
        labels.append(f"C{k}")
        raw = summary[key]
        if raw is None:
            values.append(float("nan"))
        else:
            values.append(float(raw))

    cmap = plt.cm.tab10
    y_pos = np.arange(len(labels))

    finite_vals = [v for v in values if np.isfinite(v)]
    max_val = max(finite_vals) if finite_vals else 1.0
    if max_val <= 0:
        max_val = 1.0

    for i, (lbl, val, k) in enumerate(
        zip(labels, values, k_indices)
    ):
        if np.isfinite(val):
            ax.barh(
                i, val, height=0.5,
                color=cmap(k % 10),
            )
        else:
            ax.barh(
                i, 0.0, height=0.5,
                color="none", edgecolor="gray",
                hatch="//",
            )

    ax.set_xlim(0, max_val * 1.25)
    offset = 0.02 * max_val * 1.25

    for i, val in enumerate(values):
        if np.isfinite(val):
            ax.annotate(
                f"{val:.3g} Hz²",
                xy=(val + offset, i),
                fontsize=8, va="center", ha="left",
            )
        else:
            ax.annotate(
                "NaN",
                xy=(offset, i),
                fontsize=8, va="center", ha="left",
                color="gray",
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.axvline(
        0, color="gray", linewidth=0.8, alpha=0.5,
    )
    ax.set_xlabel("Freq drift  Var_t[f_max] (Hz²)")
    ax.set_ylabel("Component")
    ax.set_title(
        "Post-Hoc freq_drift Aggregate  "
        "(global Var_t[f_max] per component)"
    )
    ax.grid(axis="x", alpha=0.3, linestyle=":")


# ------------------------------------------------------------------
# 5. Main entry point
# ------------------------------------------------------------------


def plot_metrics(
    results_dir: str | Path,
    *,
    show: bool = False,
) -> Path:
    """Load data, build figure, save to PDF and PNG.

    Parameters
    ----------
    results_dir : str or Path
        Path to the results directory containing
        ``metrics.csv`` and optionally ``run_summary.json``.
    show : bool
        If ``True``, call ``plt.show()`` after saving.

    Returns
    -------
    Path
        Absolute path to the saved PDF file.

    Raises
    ------
    FileNotFoundError
        If ``metrics.csv`` is not found.
    """
    results_dir = Path(results_dir)
    df, summary = _load_data(results_dir)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    })

    fig, axes = plt.subplots(
        3, 1, figsize=(10, 11),
        gridspec_kw={"height_ratios": [1.0, 1.5, 0.8]},
    )

    axes[1].sharex(axes[0])
    axes[0].tick_params(labelbottom=False)

    _plot_sv_drift(axes[0], df)
    _plot_freq_trajectories(axes[1], df)
    _plot_freq_drift_bar(axes[2], summary)

    output_name = results_dir.name
    fig.suptitle(
        f"Streaming SSA Metrics — {output_name}",
        fontsize=12, y=1.01, fontweight="semibold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    fig.subplots_adjust(hspace=0.35)

    pdf_path = results_dir / "metrics_plot.pdf"
    png_path = results_dir / "metrics_plot.png"
    fig.savefig(
        pdf_path, bbox_inches="tight",
    )
    fig.savefig(
        png_path, dpi=200, bbox_inches="tight",
    )
    logger.info("Saved: %s", pdf_path)

    if show:
        plt.show()
    plt.close(fig)

    return pdf_path.resolve()


# ------------------------------------------------------------------
# 6. CLI entry point
# ------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Plot streaming SSA metrics from a results"
            " directory."
        ),
    )
    parser.add_argument(
        "results_dir",
        type=str,
        help=(
            "Path to the results directory "
            "(must contain metrics.csv)."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively after saving.",
    )
    args = parser.parse_args()
    try:
        out = plot_metrics(args.results_dir, show=args.show)
        print(f"Plot saved to: {out}")
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
