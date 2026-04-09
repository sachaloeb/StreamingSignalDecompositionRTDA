"""Run the streaming SSD pipeline and produce window-inspection plots.

Generates a chirp + sinusoid test signal, processes it through the
full streaming pipeline, and saves per-window reconstruction
comparisons plus an NMSE-over-time plot to
``results/window_inspection/``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.synthetic.generators import chirp_plus_sinusoid
from src.ssd.core import SSD
from src.streaming.component_matcher import ComponentMatcher
from src.streaming.trajectory_store import TrajectoryStore
from src.streaming.window_manager import WindowManager
from src.visualization.window_inspector import (
    plot_nmse_over_time,
    plot_window_grid,
    plot_window_reconstruction,
)

CONFIG_PATH = "experiments/configs/baseline.yaml"


def main() -> None:
    """Entry point for the window-inspection experiment."""
    # Load configuration from baseline config
    with open(CONFIG_PATH) as fh:
        cfg = yaml.safe_load(fh)

    sig_cfg = cfg["signal"]
    stream_cfg = cfg["streaming"]
    ssd_cfg = cfg["ssd"]
    matcher_cfg = cfg["matcher"]

    # Generate signal using config parameters
    signal = chirp_plus_sinusoid(
        N=sig_cfg["N"],
        f_sin=sig_cfg["f_sin"],
        f_start=sig_cfg["f_start"],
        f_end=sig_cfg["f_end"],
        fs=sig_cfg["fs"],
        snr_db=sig_cfg["snr_db"],
        seed=42,
    )

    wm = WindowManager(
        window_len=stream_cfg["window_len"],
        stride=stream_cfg["stride"],
        fs=sig_cfg["fs"],
    )
    ssd = SSD(
        fs=sig_cfg["fs"],
        nmse_threshold=ssd_cfg["nmse_threshold"],
        max_iter=ssd_cfg["max_iter"],
    )
    store = TrajectoryStore(max_components=stream_cfg["max_components"])
    matcher = ComponentMatcher(
        distance=matcher_cfg["distance"],
        freq_weight=matcher_cfg["freq_weight"],
        fs=sig_cfg["fs"],
        lookback=matcher_cfg["lookback"],
        max_cost=matcher_cfg["max_cost"],
        max_trajectories= stream_cfg["max_components"],
    )

    pipeline_records: list[dict] = []
    prev_components: list[np.ndarray] = []

    for t, sample in enumerate(signal):
        window = wm.push(sample)
        if window is None:
            continue

        sample_start = t - wm.window_len + 1
        window_idx = len(pipeline_records)
        components = ssd.fit(window)
        signal_components = components[:-1]

        overlap = wm.overlap
        matching = matcher.match_stateful(signal_components, overlap)
        store.update(
            sample_start, signal_components, matching, overlap,
        )

        pipeline_records.append({
            "window_idx": window_idx,
            "sample_start": sample_start,
            "window_signal": window.copy(),
            "components": [
                c.copy() for c in signal_components
            ],
        })
        prev_components = signal_components

    # ---- assemble full reconstruction from TrajectoryStore ----
    reconstruction = np.full(len(signal), np.nan)
    all_trajs = store.get_all()
    for _comp_idx, traj in all_trajs.items():
        valid = np.isfinite(traj)
        n_traj = len(traj)
        reconstruction[:n_traj] = np.where(
            valid,
            np.nan_to_num(
                reconstruction[:n_traj], nan=0.0,
            ) + traj,
            reconstruction[:n_traj],
        )

    # ---- save outputs ----
    out_dir = os.path.join("results", "window_inspection")
    os.makedirs(out_dir, exist_ok=True)

    plot_window_grid(
        pipeline_records, n_windows=9, fs=sig_cfg["fs"],
        save_path=os.path.join(out_dir, "window_grid.png"),
    )

    _t_axis, nmse_vals = plot_nmse_over_time(
        signal, reconstruction, fs=sig_cfg["fs"],
        save_path=os.path.join(
            out_dir, "nmse_over_time.png",
        ),
    )

    rep_indices = [
        0,
        len(pipeline_records) // 2,
        len(pipeline_records) - 1,
    ]
    for idx in rep_indices:
        rec = pipeline_records[idx]
        plot_window_reconstruction(
            rec["window_signal"],
            rec["components"],
            rec["window_idx"],
            rec["sample_start"],
            fs=sig_cfg["fs"],
            save_path=os.path.join(
                out_dir,
                f"window_{rec['window_idx']:04d}.png",
            ),
        )

    # ---- summary ----
    valid_nmse = nmse_vals[np.isfinite(nmse_vals)]
    print(f"Windows processed:  {len(pipeline_records)}")
    print(f"Trajectories stored: {len(all_trajs)}")
    if len(valid_nmse) > 0:
        print(
            f"NMSE range: "
            f"[{np.min(valid_nmse):.4f}, "
            f"{np.max(valid_nmse):.4f}] "
            f"(ignoring NaN)"
        )
    else:
        print("NMSE range: no valid values")
    print(f"Outputs saved to: {out_dir}/")


if __name__ == "__main__":
    main()
