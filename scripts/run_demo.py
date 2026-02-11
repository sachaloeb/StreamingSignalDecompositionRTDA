#!/usr/bin/env python3
"""Run end-to-end demo: generate signal → sliding windows → decomposition → tracking → metrics."""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from experiments.synthetic import generate_synthetic_signal
from streamssd.engines import SSABatchEngine, SSDBonizziEngine
from streamssd.metrics import compute_pair_metrics
from streamssd.tracking import align_components, fix_component_sign
from streamssd.utils import set_seed, setup_logging
from streamssd.window import SlidingWindow


def load_config(config_path: str) -> dict[str, Any]:
    """Load YAML configuration file.
    
    Args:
        config_path: Path to YAML config file.
        
    Returns:
        Configuration dictionary.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_engine(config: dict[str, Any]) -> Any:
    """Create decomposition engine from config.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        Engine instance (SSA or SSD).
    """
    engine_type = config["engine"]["type"].lower()
    L = config["embedding"]["L"]
    
    if engine_type == "ssa":
        r = config["engine"]["r_ssa"]
        return SSABatchEngine(L=L, r=r)
    elif engine_type == "ssd":
        M = config["engine"]["M_ssd"]
        fmin = config["engine"].get("fmin")
        fmax = config["engine"].get("fmax")
        return SSDBonizziEngine(L=L, M=M, fmin=fmin, fmax=fmax)
    else:
        raise ValueError(f"Unknown engine type: {engine_type}")


def main():
    parser = argparse.ArgumentParser(description="Run streaming signal decomposition demo")
    parser.add_argument(
        "--config", type=str, default="configs/demo.yaml", help="Path to config file"
    )
    args = parser.parse_args()
    
    # Setup
    setup_logging()
    config = load_config(args.config)
    set_seed(config["output"]["seed"])
    
    # Create output directory
    output_dir = Path(config["output"]["dir"])
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("Streaming Signal Decomposition Demo")
    print("=" * 60)
    
    # Generate synthetic signal
    print("\n1. Generating synthetic signal...")
    signal, components = generate_synthetic_signal(
        fs=config["signal"]["fs"],
        duration=config["signal"]["duration"],
        noise_level=config["signal"]["noise_level"],
        seed=config["output"]["seed"],
    )
    print(f"   Signal length: {len(signal)} samples")
    print(f"   Sampling frequency: {config['signal']['fs']} Hz")
    
    # Create engine
    print(f"\n2. Creating {config['engine']['type'].upper()} engine...")
    engine = create_engine(config)
    print(f"   Embedding dimension L: {config['embedding']['L']}")
    
    # Create sliding window
    print("\n3. Setting up sliding windows...")
    window_length = config["window"]["W"]
    stride = config["window"]["stride"]
    sliding_window = SlidingWindow(window_length=window_length, stride=stride)
    num_windows = sliding_window.num_windows(len(signal))
    print(f"   Window length W: {window_length}")
    print(f"   Stride s: {stride}")
    print(f"   Number of windows: {num_windows}")
    
    # Process windows
    print("\n4. Processing windows...")
    all_results = []
    all_components = []
    overlap_len = config["tracking"]["overlap_len"]
    fs = config["signal"]["fs"]
    
    for window_idx, window_data in enumerate(sliding_window.extract_windows(signal)):
        if window_idx % 10 == 0:
            print(f"   Processing window {window_idx + 1}/{num_windows}...")
        
        # Decompose window
        result = engine.fit_window(window_data, fs=fs)
        all_results.append(result)
        all_components.append(result.components)
    
    print(f"   Completed {num_windows} windows")
    
    # Track components across windows
    print("\n5. Tracking components across windows...")
    tracks = {}  # track_id -> list of (window_idx, component_idx)
    next_track_id = 0
    pair_metrics_list = []
    
    similarity_threshold = config["tracking"]["similarity_threshold"]
    
    for k in range(num_windows - 1):
        components_k = all_components[k]
        components_k1 = all_components[k + 1]
        
        # Align components
        matches, unmatched_k, unmatched_k1 = align_components(
            components_k,
            components_k1,
            overlap_len,
            similarity_threshold=similarity_threshold,
            fs=fs,
        )
        
        # Update tracks
        for i, j, sim in matches:
            # Find track for component i in window k
            track_id = None
            for tid, track in tracks.items():
                if track and track[-1] == (k, i):
                    track_id = tid
                    break
            
            if track_id is None:
                # New track
                track_id = next_track_id
                next_track_id += 1
                tracks[track_id] = [(k, i)]
            
            # Add to track
            tracks[track_id].append((k + 1, j))
        
        # Compute metrics for matched pairs
        for i, j, sim in matches:
            comp_k = components_k[i]
            comp_k1 = components_k1[j]
            
            # Fix sign
            comp_k1_fixed = fix_component_sign(comp_k, comp_k1, overlap_len)
            
            metrics = compute_pair_metrics(comp_k, comp_k1_fixed, overlap_len, fs=fs)
            metrics["window_k"] = k
            metrics["window_k1"] = k + 1
            metrics["component_k"] = i
            metrics["component_k1"] = j
            metrics["similarity"] = sim
            pair_metrics_list.append(metrics)
    
    print(f"   Found {len(tracks)} component tracks")
    print(f"   Computed {len(pair_metrics_list)} pair metrics")
    
    # Save metrics
    print("\n6. Saving results...")
    if pair_metrics_list:
        df_metrics = pd.DataFrame(pair_metrics_list)
        metrics_path = output_dir / "metrics.csv"
        df_metrics.to_csv(metrics_path, index=False)
        print(f"   Saved metrics to {metrics_path}")
    
    # Save tracks
    tracks_serializable = {
        str(tid): [(int(w), int(c)) for w, c in track] for tid, track in tracks.items()
    }
    tracks_path = output_dir / "tracks.json"
    with open(tracks_path, "w") as f:
        json.dump(tracks_serializable, f, indent=2)
    print(f"   Saved tracks to {tracks_path}")
    
    # Create plots
    print("\n7. Creating plots...")
    
    # Plot 1: Original signal and components
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    t = components["t"]
    
    axes[0].plot(t, signal, "k-", linewidth=0.5, label="Total signal")
    axes[0].set_title("Original Signal")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Show components for first window
    window_0_data = list(sliding_window.extract_windows(signal))[0]
    result_0 = all_results[0]
    t_window = np.arange(len(window_0_data)) / fs
    
    axes[1].plot(t_window, window_0_data, "k-", linewidth=1, label="Window 0")
    axes[1].plot(t_window, sum(result_0.components), "r--", linewidth=1, label="Reconstructed")
    axes[1].set_title("Window 0: Original vs Reconstructed")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Show individual components for window 0
    for i, comp in enumerate(result_0.components[:3]):  # Show first 3
        axes[2].plot(t_window, comp, linewidth=1, label=f"Component {i+1}")
    axes[2].set_title("Window 0: Individual Components (first 3)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Amplitude")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    plot1_path = output_dir / "signal_and_components.png"
    plt.savefig(plot1_path, dpi=150, bbox_inches="tight")
    print(f"   Saved plot to {plot1_path}")
    plt.close()
    
    # Plot 2: Stability metrics over time
    if pair_metrics_list:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        df = pd.DataFrame(pair_metrics_list)
        
        axes[0, 0].plot(df["window_k"], df["corr"], "o", markersize=3, alpha=0.6)
        axes[0, 0].set_title("Correlation Across Windows")
        axes[0, 0].set_xlabel("Window k")
        axes[0, 0].set_ylabel("Correlation")
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(df["window_k"], df["overlap_l2"], "o", markersize=3, alpha=0.6)
        axes[0, 1].set_title("L2 Difference Across Windows")
        axes[0, 1].set_xlabel("Window k")
        axes[0, 1].set_ylabel("Normalized L2")
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(df["window_k"], df["energy_delta"], "o", markersize=3, alpha=0.6)
        axes[1, 0].set_title("Energy Delta Across Windows")
        axes[1, 0].set_xlabel("Window k")
        axes[1, 0].set_ylabel("Relative Energy Change")
        axes[1, 0].grid(True, alpha=0.3)
        
        if not df["freq_delta"].isna().all():
            axes[1, 1].plot(df["window_k"], df["freq_delta"], "o", markersize=3, alpha=0.6)
            axes[1, 1].set_title("Frequency Delta Across Windows")
            axes[1, 1].set_xlabel("Window k")
            axes[1, 1].set_ylabel("Frequency Difference (Hz)")
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot2_path = output_dir / "stability_metrics.png"
        plt.savefig(plot2_path, dpi=150, bbox_inches="tight")
        print(f"   Saved plot to {plot2_path}")
        plt.close()
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
