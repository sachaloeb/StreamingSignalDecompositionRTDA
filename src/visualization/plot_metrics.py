"""Plot streaming SSA metrics from a results directory.  # WEEK6-PLOT

Produces a 3-panel figure:                                 # WEEK6-PLOT
  1. Singular Value Drift over Windows  (cross-window)     # WEEK6-PLOT
  2. Dominant Frequency per Component   (per-window)       # WEEK6-PLOT
  3. Post-Hoc freq_drift Aggregate      (bar chart)        # WEEK6-PLOT

Usage                                                      # WEEK6-PLOT
-----                                                      # WEEK6-PLOT
    python src/visualization/plot_metrics.py results/demo_run  # WEEK6-PLOT
"""  # WEEK6-PLOT

from __future__ import annotations  # WEEK6-PLOT

import argparse  # WEEK6-PLOT
import json  # WEEK6-PLOT
import logging  # WEEK6-PLOT
import re  # WEEK6-PLOT
import sys  # WEEK6-PLOT
from pathlib import Path  # WEEK6-PLOT

import matplotlib.pyplot as plt  # WEEK6-PLOT
import numpy as np  # WEEK6-PLOT
import pandas as pd  # WEEK6-PLOT

logger = logging.getLogger(__name__)  # WEEK6-PLOT


# ------------------------------------------------------------------ # WEEK6-PLOT
# 1. Data loading                                                     # WEEK6-PLOT
# ------------------------------------------------------------------ # WEEK6-PLOT


def _load_data(  # WEEK6-PLOT
    results_dir: Path,  # WEEK6-PLOT
) -> tuple[pd.DataFrame, dict]:  # WEEK6-PLOT
    """Load metrics.csv and run_summary.json from *results_dir*.  # WEEK6-PLOT

    Parameters  # WEEK6-PLOT
    ----------  # WEEK6-PLOT
    results_dir : Path  # WEEK6-PLOT
        Directory containing ``metrics.csv`` and optionally  # WEEK6-PLOT
        ``run_summary.json``.  # WEEK6-PLOT

    Returns  # WEEK6-PLOT
    -------  # WEEK6-PLOT
    df : pd.DataFrame  # WEEK6-PLOT
        The metrics dataframe.  # WEEK6-PLOT
    summary : dict  # WEEK6-PLOT
        Contents of ``run_summary.json``, or ``{}`` if absent.  # WEEK6-PLOT

    Raises  # WEEK6-PLOT
    ------  # WEEK6-PLOT
    FileNotFoundError  # WEEK6-PLOT
        If ``metrics.csv`` does not exist.  # WEEK6-PLOT
    ValueError  # WEEK6-PLOT
        If ``window_index`` column is missing.  # WEEK6-PLOT
    """  # WEEK6-PLOT
    csv_path = results_dir / "metrics.csv"  # WEEK6-PLOT
    if not csv_path.exists():  # WEEK6-PLOT
        raise FileNotFoundError(  # WEEK6-PLOT
            f"metrics.csv not found in {results_dir}"  # WEEK6-PLOT
        )  # WEEK6-PLOT
    df = pd.read_csv(csv_path)  # WEEK6-PLOT
    if "window_index" not in df.columns:  # WEEK6-PLOT
        raise ValueError(  # WEEK6-PLOT
            "'window_index' column missing from metrics.csv"  # WEEK6-PLOT
        )  # WEEK6-PLOT
    df["window_index"] = df["window_index"].astype(int)  # WEEK6-PLOT

    summary: dict = {}  # WEEK6-PLOT
    json_path = results_dir / "run_summary.json"  # WEEK6-PLOT
    if json_path.exists():  # WEEK6-PLOT
        with open(json_path) as fh:  # WEEK6-PLOT
            summary = json.load(fh)  # WEEK6-PLOT
    else:  # WEEK6-PLOT
        logger.warning(  # WEEK6-PLOT
            "run_summary.json not found in %s", results_dir  # WEEK6-PLOT
        )  # WEEK6-PLOT
    return df, summary  # WEEK6-PLOT


# ------------------------------------------------------------------ # WEEK6-PLOT
# 2. Panel 1: Singular Value Drift                                    # WEEK6-PLOT
# ------------------------------------------------------------------ # WEEK6-PLOT


def _plot_sv_drift(  # WEEK6-PLOT
    ax: plt.Axes,  # WEEK6-PLOT
    df: pd.DataFrame,  # WEEK6-PLOT
) -> None:  # WEEK6-PLOT
    """Render panel 1: singular_value_drift over windows.  # WEEK6-PLOT

    Parameters  # WEEK6-PLOT
    ----------  # WEEK6-PLOT
    ax : matplotlib Axes  # WEEK6-PLOT
    df : pd.DataFrame  # WEEK6-PLOT
        Must contain ``window_index`` and optionally  # WEEK6-PLOT
        ``singular_value_drift``.  # WEEK6-PLOT
    """  # WEEK6-PLOT
    if "singular_value_drift" not in df.columns:  # WEEK6-PLOT
        ax.text(  # WEEK6-PLOT
            0.5, 0.5,  # WEEK6-PLOT
            "Column 'singular_value_drift' not found in metrics.csv",  # WEEK6-PLOT
            transform=ax.transAxes, ha="center", va="center",  # WEEK6-PLOT
            fontsize=9, color="gray", fontstyle="italic",  # WEEK6-PLOT
        )  # WEEK6-PLOT
        ax.set_title(  # WEEK6-PLOT
            "Singular Value Drift  "  # WEEK6-PLOT
            "(cross-window, NaN at t=0 by design)"  # WEEK6-PLOT
        )  # WEEK6-PLOT
        return  # WEEK6-PLOT

    sub = df[df["window_index"] >= 1].copy()  # WEEK6-PLOT
    wins = sub["window_index"].values  # WEEK6-PLOT
    vals = pd.to_numeric(  # WEEK6-PLOT
        sub["singular_value_drift"], errors="coerce"  # WEEK6-PLOT
    ).values.astype(float)  # WEEK6-PLOT
    finite = np.isfinite(vals)  # WEEK6-PLOT

    if not finite.any():  # WEEK6-PLOT
        ax.text(  # WEEK6-PLOT
            0.5, 0.5, "No data available",  # WEEK6-PLOT
            transform=ax.transAxes, ha="center", va="center",  # WEEK6-PLOT
            fontsize=9, color="gray", fontstyle="italic",  # WEEK6-PLOT
        )  # WEEK6-PLOT
    else:  # WEEK6-PLOT
        ax.plot(  # WEEK6-PLOT
            wins[finite], vals[finite],  # WEEK6-PLOT
            color="#2d6a9f", linewidth=1.5,  # WEEK6-PLOT
        )  # WEEK6-PLOT
        if finite.sum() >= 2:  # WEEK6-PLOT
            mu = float(np.nanmean(vals[finite]))  # WEEK6-PLOT
            sd = float(np.nanstd(vals[finite]))  # WEEK6-PLOT
            ax.axhspan(  # WEEK6-PLOT
                mu - sd, mu + sd,  # WEEK6-PLOT
                color="#2d6a9f", alpha=0.15,  # WEEK6-PLOT
            )  # WEEK6-PLOT

    ax.axhline(  # WEEK6-PLOT
        0, color="gray", linewidth=0.8,  # WEEK6-PLOT
        linestyle="--", alpha=0.5,  # WEEK6-PLOT
    )  # WEEK6-PLOT
    ax.set_ylabel("‖S_t − S_{t-1}‖_F")  # WEEK6-PLOT
    ax.set_title(  # WEEK6-PLOT
        "Singular Value Drift  "  # WEEK6-PLOT
        "(cross-window, NaN at t=0 by design)"  # WEEK6-PLOT
    )  # WEEK6-PLOT
    ax.grid(axis="y", alpha=0.3, linestyle=":")  # WEEK6-PLOT


# ------------------------------------------------------------------ # WEEK6-PLOT
# 3. Panel 2: Dominant Frequency Trajectories                         # WEEK6-PLOT
# ------------------------------------------------------------------ # WEEK6-PLOT


def _plot_freq_trajectories(  # WEEK6-PLOT
    ax: plt.Axes,  # WEEK6-PLOT
    df: pd.DataFrame,  # WEEK6-PLOT
) -> None:  # WEEK6-PLOT
    """Render panel 2: f_max_cK columns over all windows.  # WEEK6-PLOT

    Parameters  # WEEK6-PLOT
    ----------  # WEEK6-PLOT
    ax : matplotlib Axes  # WEEK6-PLOT
    df : pd.DataFrame  # WEEK6-PLOT
        Must contain ``window_index`` and zero or more  # WEEK6-PLOT
        ``f_max_cK`` columns.  # WEEK6-PLOT
    """  # WEEK6-PLOT
    comp_cols = sorted(  # WEEK6-PLOT
        [c for c in df.columns  # WEEK6-PLOT
         if re.fullmatch(r"f_max_c\d+", c)],  # WEEK6-PLOT
        key=lambda c: int(c.split("f_max_c")[1]),  # WEEK6-PLOT
    )  # WEEK6-PLOT

    if not comp_cols:  # WEEK6-PLOT
        ax.text(  # WEEK6-PLOT
            0.5, 0.5, "No f_max columns found",  # WEEK6-PLOT
            transform=ax.transAxes, ha="center", va="center",  # WEEK6-PLOT
            fontsize=9, color="gray", fontstyle="italic",  # WEEK6-PLOT
        )  # WEEK6-PLOT
        ax.set_title(  # WEEK6-PLOT
            "Dominant Frequency per Component over Windows"  # WEEK6-PLOT
        )  # WEEK6-PLOT
        ax.set_xlabel("Window index")  # WEEK6-PLOT
        ax.set_ylabel("Dominant frequency (Hz)")  # WEEK6-PLOT
        return  # WEEK6-PLOT

    wins = df["window_index"].values  # WEEK6-PLOT
    n = len(df)  # WEEK6-PLOT
    step = max(1, n // 40)  # WEEK6-PLOT
    cmap = plt.cm.tab10  # WEEK6-PLOT

    for i, col in enumerate(comp_cols):  # WEEK6-PLOT
        k = int(col.split("f_max_c")[1])  # WEEK6-PLOT
        vals = pd.to_numeric(  # WEEK6-PLOT
            df[col], errors="coerce"  # WEEK6-PLOT
        ).values.astype(float)  # WEEK6-PLOT
        ax.plot(  # WEEK6-PLOT
            wins, vals,  # WEEK6-PLOT
            color=cmap(k % 10),  # WEEK6-PLOT
            label=f"Component {k}",  # WEEK6-PLOT
            marker="o", markersize=3,  # WEEK6-PLOT
            markevery=step, linewidth=1.0,  # WEEK6-PLOT
        )  # WEEK6-PLOT

    ax.set_xlabel("Window index")  # WEEK6-PLOT
    ax.set_ylabel("Dominant frequency (Hz)")  # WEEK6-PLOT
    ax.grid(axis="both", alpha=0.3, linestyle=":")  # WEEK6-PLOT

    if len(comp_cols) == 1:  # WEEK6-PLOT
        k0 = int(comp_cols[0].split("f_max_c")[1])  # WEEK6-PLOT
        ax.set_title(  # WEEK6-PLOT
            "Dominant Frequency over Windows  "  # WEEK6-PLOT
            f"(Component {k0})"  # WEEK6-PLOT
        )  # WEEK6-PLOT
    else:  # WEEK6-PLOT
        ax.set_title(  # WEEK6-PLOT
            "Dominant Frequency per Component over Windows"  # WEEK6-PLOT
        )  # WEEK6-PLOT
        ax.legend(  # WEEK6-PLOT
            loc="upper right", fontsize=8, framealpha=0.7,  # WEEK6-PLOT
        )  # WEEK6-PLOT


# ------------------------------------------------------------------ # WEEK6-PLOT
# 4. Panel 3: Post-Hoc freq_drift Aggregate                           # WEEK6-PLOT
# ------------------------------------------------------------------ # WEEK6-PLOT


def _plot_freq_drift_bar(  # WEEK6-PLOT
    ax: plt.Axes,  # WEEK6-PLOT
    summary: dict,  # WEEK6-PLOT
) -> None:  # WEEK6-PLOT
    """Render panel 3: horizontal bar chart of freq_drift_cK.  # WEEK6-PLOT

    Parameters  # WEEK6-PLOT
    ----------  # WEEK6-PLOT
    ax : matplotlib Axes  # WEEK6-PLOT
    summary : dict  # WEEK6-PLOT
        Contents of ``run_summary.json``.  # WEEK6-PLOT
    """  # WEEK6-PLOT
    drift_keys = sorted(  # WEEK6-PLOT
        [k for k in summary  # WEEK6-PLOT
         if re.fullmatch(r"freq_drift_c\d+", k)],  # WEEK6-PLOT
        key=lambda k: int(k.split("freq_drift_c")[1]),  # WEEK6-PLOT
    )  # WEEK6-PLOT

    if not drift_keys:  # WEEK6-PLOT
        ax.text(  # WEEK6-PLOT
            0.5, 0.5,  # WEEK6-PLOT
            "run_summary.json not found or contains no\n"  # WEEK6-PLOT
            "freq_drift entries — run the experiment first.",  # WEEK6-PLOT
            transform=ax.transAxes, ha="center", va="center",  # WEEK6-PLOT
            fontsize=9, color="gray", fontstyle="italic",  # WEEK6-PLOT
        )  # WEEK6-PLOT
        ax.set_title(  # WEEK6-PLOT
            "Post-Hoc freq_drift Aggregate  "  # WEEK6-PLOT
            "(global Var_t[f_max] per component)"  # WEEK6-PLOT
        )  # WEEK6-PLOT
        return  # WEEK6-PLOT

    labels: list[str] = []  # WEEK6-PLOT
    values: list[float] = []  # WEEK6-PLOT
    k_indices: list[int] = []  # WEEK6-PLOT
    for key in drift_keys:  # WEEK6-PLOT
        k = int(key.split("freq_drift_c")[1])  # WEEK6-PLOT
        k_indices.append(k)  # WEEK6-PLOT
        labels.append(f"C{k}")  # WEEK6-PLOT
        raw = summary[key]  # WEEK6-PLOT
        if raw is None:  # WEEK6-PLOT
            values.append(float("nan"))  # WEEK6-PLOT
        else:  # WEEK6-PLOT
            values.append(float(raw))  # WEEK6-PLOT

    cmap = plt.cm.tab10  # WEEK6-PLOT
    y_pos = np.arange(len(labels))  # WEEK6-PLOT

    finite_vals = [v for v in values if np.isfinite(v)]  # WEEK6-PLOT
    max_val = max(finite_vals) if finite_vals else 1.0  # WEEK6-PLOT
    if max_val <= 0:  # WEEK6-PLOT
        max_val = 1.0  # WEEK6-PLOT

    for i, (lbl, val, k) in enumerate(  # WEEK6-PLOT
        zip(labels, values, k_indices)  # WEEK6-PLOT
    ):  # WEEK6-PLOT
        if np.isfinite(val):  # WEEK6-PLOT
            ax.barh(  # WEEK6-PLOT
                i, val, height=0.5,  # WEEK6-PLOT
                color=cmap(k % 10),  # WEEK6-PLOT
            )  # WEEK6-PLOT
        else:  # WEEK6-PLOT
            ax.barh(  # WEEK6-PLOT
                i, 0.0, height=0.5,  # WEEK6-PLOT
                color="none", edgecolor="gray",  # WEEK6-PLOT
                hatch="//",  # WEEK6-PLOT
            )  # WEEK6-PLOT

    ax.set_xlim(0, max_val * 1.25)  # WEEK6-PLOT
    offset = 0.02 * max_val * 1.25  # WEEK6-PLOT

    for i, val in enumerate(values):  # WEEK6-PLOT
        if np.isfinite(val):  # WEEK6-PLOT
            ax.annotate(  # WEEK6-PLOT
                f"{val:.3g} Hz²",  # WEEK6-PLOT
                xy=(val + offset, i),  # WEEK6-PLOT
                fontsize=8, va="center", ha="left",  # WEEK6-PLOT
            )  # WEEK6-PLOT
        else:  # WEEK6-PLOT
            ax.annotate(  # WEEK6-PLOT
                "NaN",  # WEEK6-PLOT
                xy=(offset, i),  # WEEK6-PLOT
                fontsize=8, va="center", ha="left",  # WEEK6-PLOT
                color="gray",  # WEEK6-PLOT
            )  # WEEK6-PLOT

    ax.set_yticks(y_pos)  # WEEK6-PLOT
    ax.set_yticklabels(labels)  # WEEK6-PLOT
    ax.invert_yaxis()  # WEEK6-PLOT
    ax.axvline(  # WEEK6-PLOT
        0, color="gray", linewidth=0.8, alpha=0.5,  # WEEK6-PLOT
    )  # WEEK6-PLOT
    ax.set_xlabel("Component")  # WEEK6-PLOT
    ax.set_ylabel("Freq drift  Var_t[f_max] (Hz²)")  # WEEK6-PLOT
    ax.set_title(  # WEEK6-PLOT
        "Post-Hoc freq_drift Aggregate  "  # WEEK6-PLOT
        "(global Var_t[f_max] per component)"  # WEEK6-PLOT
    )  # WEEK6-PLOT
    ax.grid(axis="x", alpha=0.3, linestyle=":")  # WEEK6-PLOT


# ------------------------------------------------------------------ # WEEK6-PLOT
# 5. Main entry point                                                  # WEEK6-PLOT
# ------------------------------------------------------------------ # WEEK6-PLOT


def plot_metrics(  # WEEK6-PLOT
    results_dir: str | Path,  # WEEK6-PLOT
    *,  # WEEK6-PLOT
    show: bool = False,  # WEEK6-PLOT
) -> Path:  # WEEK6-PLOT
    """Load data, build figure, save to PDF and PNG.  # WEEK6-PLOT

    Parameters  # WEEK6-PLOT
    ----------  # WEEK6-PLOT
    results_dir : str or Path  # WEEK6-PLOT
        Path to the results directory containing  # WEEK6-PLOT
        ``metrics.csv`` and optionally ``run_summary.json``.  # WEEK6-PLOT
    show : bool  # WEEK6-PLOT
        If ``True``, call ``plt.show()`` after saving.  # WEEK6-PLOT

    Returns  # WEEK6-PLOT
    -------  # WEEK6-PLOT
    Path  # WEEK6-PLOT
        Absolute path to the saved PDF file.  # WEEK6-PLOT

    Raises  # WEEK6-PLOT
    ------  # WEEK6-PLOT
    FileNotFoundError  # WEEK6-PLOT
        If ``metrics.csv`` is not found.  # WEEK6-PLOT
    """  # WEEK6-PLOT
    results_dir = Path(results_dir)  # WEEK6-PLOT
    df, summary = _load_data(results_dir)  # WEEK6-PLOT

    plt.rcParams.update({  # WEEK6-PLOT
        "font.family": "DejaVu Sans",  # WEEK6-PLOT
        "axes.spines.top": False,  # WEEK6-PLOT
        "axes.spines.right": False,  # WEEK6-PLOT
        "axes.titlesize": 10,  # WEEK6-PLOT
        "axes.labelsize": 9,  # WEEK6-PLOT
        "xtick.labelsize": 8,  # WEEK6-PLOT
        "ytick.labelsize": 8,  # WEEK6-PLOT
        "legend.fontsize": 8,  # WEEK6-PLOT
    })  # WEEK6-PLOT

    fig, axes = plt.subplots(  # WEEK6-PLOT
        3, 1, figsize=(10, 11),  # WEEK6-PLOT
        gridspec_kw={"height_ratios": [1.0, 1.5, 0.8]},  # WEEK6-PLOT
    )  # WEEK6-PLOT

    axes[1].sharex(axes[0])  # WEEK6-PLOT
    axes[0].tick_params(labelbottom=False)  # WEEK6-PLOT

    _plot_sv_drift(axes[0], df)  # WEEK6-PLOT
    _plot_freq_trajectories(axes[1], df)  # WEEK6-PLOT
    _plot_freq_drift_bar(axes[2], summary)  # WEEK6-PLOT

    output_name = results_dir.name  # WEEK6-PLOT
    fig.suptitle(  # WEEK6-PLOT
        f"Streaming SSA Metrics — {output_name}",  # WEEK6-PLOT
        fontsize=12, y=1.01, fontweight="semibold",  # WEEK6-PLOT
    )  # WEEK6-PLOT
    plt.tight_layout(rect=[0, 0, 1, 0.99])  # WEEK6-PLOT
    fig.subplots_adjust(hspace=0.35)  # WEEK6-PLOT

    pdf_path = results_dir / "metrics_plot.pdf"  # WEEK6-PLOT
    png_path = results_dir / "metrics_plot.png"  # WEEK6-PLOT
    fig.savefig(  # WEEK6-PLOT
        pdf_path, bbox_inches="tight",  # WEEK6-PLOT
    )  # WEEK6-PLOT
    fig.savefig(  # WEEK6-PLOT
        png_path, dpi=200, bbox_inches="tight",  # WEEK6-PLOT
    )  # WEEK6-PLOT
    logger.info("Saved: %s", pdf_path)  # WEEK6-PLOT

    if show:  # WEEK6-PLOT
        plt.show()  # WEEK6-PLOT
    plt.close(fig)  # WEEK6-PLOT

    return pdf_path.resolve()  # WEEK6-PLOT


# ------------------------------------------------------------------ # WEEK6-PLOT
# 6. CLI entry point                                                   # WEEK6-PLOT
# ------------------------------------------------------------------ # WEEK6-PLOT


if __name__ == "__main__":  # WEEK6-PLOT
    parser = argparse.ArgumentParser(  # WEEK6-PLOT
        description=(  # WEEK6-PLOT
            "Plot streaming SSA metrics from a results"  # WEEK6-PLOT
            " directory."  # WEEK6-PLOT
        ),  # WEEK6-PLOT
    )  # WEEK6-PLOT
    parser.add_argument(  # WEEK6-PLOT
        "results_dir",  # WEEK6-PLOT
        type=str,  # WEEK6-PLOT
        help=(  # WEEK6-PLOT
            "Path to the results directory "  # WEEK6-PLOT
            "(must contain metrics.csv)."  # WEEK6-PLOT
        ),  # WEEK6-PLOT
    )  # WEEK6-PLOT
    parser.add_argument(  # WEEK6-PLOT
        "--show",  # WEEK6-PLOT
        action="store_true",  # WEEK6-PLOT
        help="Display the figure interactively after saving.",  # WEEK6-PLOT
    )  # WEEK6-PLOT
    args = parser.parse_args()  # WEEK6-PLOT
    try:  # WEEK6-PLOT
        out = plot_metrics(args.results_dir, show=args.show)  # WEEK6-PLOT
        print(f"Plot saved to: {out}")  # WEEK6-PLOT
    except FileNotFoundError as exc:  # WEEK6-PLOT
        print(f"ERROR: {exc}", file=sys.stderr)  # WEEK6-PLOT
        sys.exit(1)  # WEEK6-PLOT
