"""CLI entry point for running streaming decomposition experiments.

Usage
-----
    python experiments/run_experiment.py \\
        --config experiments/configs/baseline.yaml \\
        --output-dir results/baseline
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.synthetic.generators import (
    chirp_plus_sinusoid,
    component_onset,
    rossler,
    two_sinusoids,
)
from src.engines import build_trajectory_matrix, get_engine, svd_decompose
from src.metrics.stability import (
    dominant_frequency,
    energy_continuity,
    freq_drift_aggregate,
    matching_confidence,
    qrf,
    singular_value_drift,
)
from src.streaming.component_matcher import ComponentMatcher
from src.streaming.trajectory_store import TrajectoryStore
from src.streaming.window_manager import WindowManager

logger = logging.getLogger(__name__)


_GENERATORS = {
    "two_sinusoids": two_sinusoids,
    "chirp_plus_sinusoid": chirp_plus_sinusoid,
    "rossler": rossler,
    "component_onset": component_onset,
}


def _generate_signal(cfg: dict) -> np.ndarray:
    """Dispatch to the appropriate signal generator."""
    sig_type = cfg.pop("type")
    gen = _GENERATORS.get(sig_type)
    if gen is None:
        raise ValueError(
            f"Unknown signal type '{sig_type}'. "
            f"Available: {list(_GENERATORS)}"
        )
    return gen(**cfg)


def build_pipeline(cfg: dict, signal_length: int) -> tuple:
    """Instantiate the streaming pipeline objects from a config dict.

    Returns
    -------
    tuple
        ``(window_manager, engine, matcher, store)``.
    """
    fs = cfg["signal"]["fs"]
    engine_cfg = dict(cfg["engine"])
    engine_name = engine_cfg.pop("name")

    wm = WindowManager(
        window_len=cfg["streaming"]["window_len"],
        stride=cfg["streaming"]["stride"],
        fs=fs,
    )
    engine = get_engine(engine_name, fs=fs, **engine_cfg)
    matcher = ComponentMatcher(
        distance=cfg["matcher"]["distance"],
        freq_weight=cfg["matcher"]["freq_weight"],
        fs=fs,
        lookback=cfg["matcher"]["lookback"],
        max_cost=cfg["matcher"]["max_cost"],
        max_trajectories=cfg["streaming"]["max_components"],
    )
    store = TrajectoryStore(
        max_components=cfg["streaming"]["max_components"],
        max_len=signal_length,
    )
    return wm, engine, matcher, store


def run(
    config_path: str | None = None,
    output_dir: str = "results/default",
    config_dict: dict | None = None,
) -> None:
    """Execute a full streaming experiment from a YAML config."""
    if config_dict is not None:
        cfg = config_dict
    elif config_path is not None:
        with open(config_path, "r") as fh:
            cfg = yaml.safe_load(fh)
    else:
        raise ValueError("Either config_path or config_dict required")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if config_path is not None:
        shutil.copy2(config_path, out / "config_used.yaml")

    sig_cfg = dict(cfg["signal"])
    signal = _generate_signal(sig_cfg)
    N = len(signal)
    fs = cfg["signal"]["fs"]

    wm, engine, matcher, store = build_pipeline(cfg, signal_length=N)

    prev_components: list[np.ndarray] | None = None
    prev_S: np.ndarray | None = None
    window_idx = 0
    overlap = wm.overlap
    max_components = cfg["streaming"]["max_components"]

    metrics_rows: list[dict] = []

    logger.info(
        "Starting experiment: N=%d, window=%d, stride=%d",
        N, wm.window_len, wm.stride,
    )

    for sample_idx in tqdm(range(N), desc="Streaming"):
        window = wm.push(float(signal[sample_idx]))
        if window is None:
            continue

        components = engine.fit(window)
        components_no_res = components[:-1]

        matching: dict[int, int | None] = dict(
            matcher.match_stateful(components_no_res, overlap)
        )
        prev_window_matching = matcher.previous_window_mapping()

        window_start = sample_idx - wm.window_len + 1
        store.update(
            window_start, components_no_res,
            matching, overlap,
        )

        recon = (
            np.sum(components_no_res, axis=0)
            if components_no_res
            else np.zeros_like(window)
        )
        qrf_val = qrf(window, recon)

        # Engine-agnostic SVD drift: decompose a Hankel trajectory
        # matrix of the raw window.
        L = max(2, len(window) // 3)
        X_win = build_trajectory_matrix(window, L)
        _, S_curr, _ = svd_decompose(X_win)
        svd_drift = singular_value_drift(S_curr, prev_S)
        prev_S = S_curr

        ec_val = energy_continuity(
            components_no_res, prev_components, prev_window_matching,
        )

        if prev_components is not None and prev_window_matching:
            cost = matcher.build_cost_matrix(
                prev_components, components_no_res,
                overlap,
            )
            mc_val = matching_confidence(cost, prev_window_matching)
        else:
            mc_val = float("nan")

        fmax_row: dict[str, float] = {}
        for ci, comp in enumerate(components_no_res):
            traj_id = matching.get(ci, -1)
            if traj_id >= 0:
                fmax_row[f"f_max_t{traj_id}"] = dominant_frequency(
                    comp, fs=fs
                )

        row: dict[str, object] = {
            "window_index": window_idx,
            "qrf": qrf_val,
            "singular_value_drift": svd_drift,
            "energy_continuity": ec_val,
            "matching_confidence": mc_val,
        }
        row.update(fmax_row)
        metrics_rows.append(row)

        prev_components = components_no_res
        window_idx += 1

    summary: dict[str, float] = {}
    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows)
        traj_cols = [
            c for c in metrics_df.columns if c.startswith("f_max_t")
        ]
        for col in traj_cols:
            tid = int(col[len("f_max_t"):])
            summary[f"freq_drift_t{tid}"] = freq_drift_aggregate(
                metrics_df[col].values,
            )

    summary_path = out / "run_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(
            {k: (None if np.isnan(v) else v)
             for k, v in summary.items()},
            fh, indent=2,
        )
    logger.info("Saved summary to %s", summary_path)

    if cfg["output"].get("save_metrics", True) and metrics_rows:
        csv_path = out / "metrics.csv"
        all_keys: dict[str, None] = {}
        for r in metrics_rows:
            for k in r:
                all_keys[k] = None
        fieldnames = list(all_keys)
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=fieldnames, extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(metrics_rows)
        logger.info("Saved metrics to %s", csv_path)

    if cfg["output"].get("save_trajectories", True):
        trajs = store.get_all()
        npz_path = out / "trajectories.npz"
        np.savez(
            npz_path,
            **{str(k): v for k, v in trajs.items()},
        )
        logger.info("Saved trajectories to %s", npz_path)

    logger.info(
        "Experiment complete: %d windows processed.", window_idx,
    )


def main() -> None:
    """Parse CLI arguments and run the experiment."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Run a streaming decomposition experiment.",
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()
    run(args.config, args.output_dir)


if __name__ == "__main__":
    main()