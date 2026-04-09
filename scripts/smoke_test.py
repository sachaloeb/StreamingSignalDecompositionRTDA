#!/usr/bin/env python
"""Smoke-test script: imports every module, runs baseline, checks results.

Exits with code 0 on full pass, code 1 on any failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os

os.chdir(ROOT)

import numpy as np


def _import_check() -> tuple[bool, str]:
    """Import all public modules and report failures."""
    modules = [
        ("src.engines.ssd", "SSD"),
        ("src.engines.ssa", "auto_ssa"),
        ("src.engines.svd_update", "RankOneUpdater"),
        ("src.streaming.window_manager", "WindowManager"),
        ("src.streaming.component_matcher", "ComponentMatcher"),
        ("src.streaming.trajectory_store", "TrajectoryStore"),
        ("src.metrics.similarity", "d_corr"),
        ("src.metrics.stability", "qrf"),
        ("src.visualization", "plot_decomposition"),
        ("experiments.synthetic.generators", "two_sinusoids"),
        ("experiments.run_experiment", "run"),
    ]
    for mod_path, symbol in modules:
        try:
            mod = __import__(mod_path, fromlist=[symbol])
            if not hasattr(mod, symbol):
                return False, (
                    f"{mod_path} has no attribute '{symbol}'"
                )
        except Exception as exc:
            return False, f"Import {mod_path} failed: {exc}"
    return True, "All modules imported successfully"


def _baseline_experiment_check() -> tuple[bool, str]:
    """Run baseline experiment and verify output."""
    from experiments.run_experiment import run

    out_dir = "results/smoke_test"
    try:
        run(
            config_path="experiments/configs/baseline.yaml",
            output_dir=out_dir,
        )
    except Exception as exc:
        return False, f"run() raised: {exc}"

    csv_path = Path(out_dir) / "metrics.csv"
    if not csv_path.exists():
        return False, f"{csv_path} does not exist"

    import csv as csv_mod

    with open(csv_path) as fh:
        reader = csv_mod.DictReader(fh)
        rows = list(reader)
    if len(rows) == 0:
        return False, "metrics.csv has 0 rows"

    return True, f"Baseline OK — {len(rows)} metric rows"


def _trajectory_check() -> tuple[bool, str]:
    """Check TrajectoryStore after baseline run has populated it."""
    traj_npz = Path("results/smoke_test/trajectories.npz")
    if not traj_npz.exists():
        return False, "trajectories.npz not found"

    data = np.load(traj_npz)
    if len(data.files) == 0:
        return False, "No trajectories stored"

    for key in data.files:
        arr = data[key]
        if np.all(np.isnan(arr)):
            return False, f"Trajectory '{key}' is all-NaN"

    return True, f"{len(data.files)} trajectories, none all-NaN"


def main() -> int:
    """Run all smoke-test checks and print summary."""
    checks = [
        ("Module imports", _import_check),
        ("Baseline experiment", _baseline_experiment_check),
        ("Trajectory store", _trajectory_check),
    ]

    results: list[tuple[str, bool, str]] = []
    for name, fn in checks:
        ok, msg = fn()
        results.append((name, ok, msg))

    col_w = max(len(r[0]) for r in results) + 2
    print()
    print("=" * (col_w + 40))
    print("SMOKE TEST SUMMARY")
    print("=" * (col_w + 40))
    all_pass = True
    for name, ok, msg in results:
        icon = "\u2705" if ok else "\u274c"
        status = "PASS" if ok else "FAIL"
        print(f"  {icon} {name:<{col_w}} {status}  {msg}")
        if not ok:
            all_pass = False
    print("=" * (col_w + 40))

    if all_pass:
        print("\nAll checks passed.")
        return 0
    else:
        print("\nSome checks FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
