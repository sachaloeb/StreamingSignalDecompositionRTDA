"""CLI entry point for running streaming SSD experiments.

Usage
-----
    python experiments/run_experiment.py \\
        --config experiments/configs/baseline.yaml \\
        --output-dir results/baseline
"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
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
from src.metrics.stability import (
    energy_continuity,
    frequency_drift,
    matching_confidence,
    qrf,
    singular_value_drift,
)
from src.ssd.core import SSD
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
    """Dispatch to the appropriate signal generator.

    Parameters
    ----------
    cfg : dict
        ``signal`` section of the YAML config.

    Returns
    -------
    np.ndarray
        Generated signal.
    """
    sig_type = cfg.pop("type")
    gen = _GENERATORS.get(sig_type)
    if gen is None:
        raise ValueError(
            f"Unknown signal type '{sig_type}'. "
            f"Available: {list(_GENERATORS)}"
        )
    return gen(**cfg)


def run(
    config_path: str | None = None,
    output_dir: str = "results/default",
    config_dict: dict | None = None,  # WEEK6-FIX: accept pre-built config dict
) -> None:
    """Execute a full streaming experiment from a YAML config.

    Parameters
    ----------
    config_path : str or None
        Path to the YAML configuration file.
    output_dir : str
        Directory to write results into.
    config_dict : dict or None, optional
        Pre-built configuration dictionary.  Takes precedence over
        *config_path* when both are given.
    """
    # WEEK6-FIX: support config_dict kwarg for programmatic use
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

    wm = WindowManager(
        window_len=cfg["streaming"]["window_len"],
        stride=cfg["streaming"]["stride"],
        fs=fs,
    )
    ssd = SSD(
        fs=fs,
        nmse_threshold=cfg["ssd"]["nmse_threshold"],
        max_iter=cfg["ssd"]["max_iter"],
    )
    matcher = ComponentMatcher(
        distance=cfg["matcher"]["distance"],
        freq_weight=cfg["matcher"]["freq_weight"],
        fs=fs,
    )
    store = TrajectoryStore(
        max_components=cfg["streaming"]["max_components"],
        max_len=N,
    )

    prev_components: list[np.ndarray] = []
    prev_S: np.ndarray | None = None
    prev_window_energy: float | None = None
    window_idx = 0
    overlap = wm.overlap

    metrics_rows: list[dict] = []
    freq_trajectory: list[float] = []

    logger.info(
        "Starting experiment: N=%d, window=%d, stride=%d",
        N, wm.window_len, wm.stride,
    )

    for sample_idx in tqdm(range(N), desc="Streaming"):
        window = wm.push(float(signal[sample_idx]))
        if window is None:
            continue

        components = ssd.fit(window)
        residual = components[-1]
        components_no_res = components[:-1]

        if len(prev_components) == 0:
            matching = {i: None for i in range(len(components_no_res))}
        else:
            matching = matcher.match(
                prev_components, components_no_res, overlap,
            )

        window_start = sample_idx - wm.window_len + 1
        store.update(
            window_start, components_no_res, matching, overlap,
        )

        recon = sum(components_no_res) + residual
        qrf_val = qrf(window, recon)

        # Track dominant frequency per window; aggregate drift globally
        # after the streaming run.
        fqs_win = np.fft.rfftfreq(len(window), d=1.0 / fs)
        mag_win = np.abs(np.fft.rfft(window))
        freq_trajectory.append(float(fqs_win[np.argmax(mag_win)]))
        fd_val = np.nan

        # Cross-window energy continuity from total component energy.
        curr_window_energy = float(
            sum(np.dot(c, c) for c in components_no_res)
        )
        ec_val = 0.0
        if prev_window_energy is not None:
            ec_val = energy_continuity(
                [prev_window_energy, curr_window_energy]
            )
        prev_window_energy = curr_window_energy

        from src.ssd.ssa import svd_decompose

        M_win = ssd._choose_window_length(window)
        X_win = SSD._build_trajectory_matrix(window, M_win)
        _, S_curr, _ = svd_decompose(X_win)
        svd_drift = 0.0
        if prev_S is not None:
            svd_drift = singular_value_drift(prev_S, S_curr)
        prev_S = S_curr

        cost = matcher.build_cost_matrix(
            prev_components if prev_components else components_no_res,
            components_no_res,
            overlap,
        )
        mc_val = matching_confidence(cost, matching)

        metrics_rows.append({
            "window": window_idx,
            "qrf": qrf_val,
            "freq_drift": fd_val,
            "energy_continuity": ec_val,
            "singular_value_drift": svd_drift,
            "matching_confidence": mc_val,
        })

        prev_components = components_no_res
        window_idx += 1

    global_freq_drift = frequency_drift(freq_trajectory)
    for row in metrics_rows:
        row["freq_drift"] = global_freq_drift

    if cfg["output"].get("save_metrics", True) and metrics_rows:
        csv_path = out / "metrics.csv"
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=list(metrics_rows[0].keys()),
            )
            writer.writeheader()
            writer.writerows(metrics_rows)
        logger.info("Saved metrics to %s", csv_path)

    if cfg["output"].get("save_trajectories", True):
        trajs = store.get_all()
        npz_path = out / "trajectories.npz"
        np.savez(npz_path, **{str(k): v for k, v in trajs.items()})
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
        description="Run a streaming SSD experiment.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save results.",
    )
    args = parser.parse_args()
    run(args.config, args.output_dir)


if __name__ == "__main__":
    main()
