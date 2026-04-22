#!/usr/bin/env python3
"""
Compute normalization statistics for LeRobot dataset.

This script reads a LeRobot dataset and computes normalization statistics
(mean, std, q01, q99) for state and actions, saving them in the format
expected by pi05.

Usage:
    python compute_lerobot_norm_stats.py \
        --data_path /path/to/lerobot/dataset \
        --output_path /path/to/output/norm_stats.json
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def load_episodes_from_parquet(data_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load all episodes from parquet files and return concatenated states and actions."""
    data_dir = Path(data_path) / "data"
    
    # Find all parquet files
    parquet_files = sorted(data_dir.glob("chunk-*/episode_*.parquet"))
    
    if not parquet_files:
        raise ValueError(f"No parquet files found in {data_dir}")
    
    print(f"Found {len(parquet_files)} episode files")
    
    all_states = []
    all_actions = []
    
    for parquet_file in tqdm(parquet_files, desc="Loading episodes"):
        # Read parquet file
        df = pd.read_parquet(parquet_file)
        
        # Extract state and action columns
        # LeRobot format: observation.state and action
        if "observation.state" in df.columns:
            state_col = "observation.state"
        else:
            raise ValueError(f"Cannot find observation.state in {parquet_file}")
        
        if "action" in df.columns:
            action_col = "action"
        else:
            raise ValueError(f"Cannot find action in {parquet_file}")
        
        # Stack the arrays (each cell contains an array)
        states = np.stack(df[state_col].values)
        actions = np.stack(df[action_col].values)
        
        all_states.append(states)
        all_actions.append(actions)
    
    # Concatenate all episodes
    all_states = np.concatenate(all_states, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)
    
    print(f"Loaded {len(all_states)} total frames")
    print(f"State shape: {all_states.shape}")
    print(f"Action shape: {all_actions.shape}")
    
    return all_states, all_actions


def compute_statistics(data: np.ndarray) -> dict:
    """Compute normalization statistics for a dataset."""
    # Compute mean and std
    mean = np.mean(data, axis=0).tolist()
    std = np.std(data, axis=0).tolist()
    
    # Compute percentiles (q01 and q99)
    q01 = np.percentile(data, 1, axis=0).tolist()
    q99 = np.percentile(data, 99, axis=0).tolist()
    
    return {
        "mean": mean,
        "std": std,
        "q01": q01,
        "q99": q99
    }


def main():
    parser = argparse.ArgumentParser(description="Compute norm stats for LeRobot dataset")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to LeRobot dataset directory (contains meta/ and data/)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output path for norm_stats.json (default: <data_path>/norm_stats.json)"
    )
    
    args = parser.parse_args()
    
    data_path = Path(args.data_path)
    
    # Set default output path
    if args.output_path is None:
        output_path = data_path / "norm_stats.json"
    else:
        output_path = Path(args.output_path)
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from: {data_path}")
    states, actions = load_episodes_from_parquet(str(data_path))
    
    print("\nComputing statistics...")
    state_stats = compute_statistics(states)
    action_stats = compute_statistics(actions)
    
    # Build norm_stats in the format expected by pi05
    norm_stats = {
        "norm_stats": {
            "state": state_stats,
            "actions": action_stats
        }
    }
    
    # Save to JSON
    print(f"\nSaving to: {output_path}")
    with open(output_path, "w") as f:
        json.dump(norm_stats, f, indent=2)
    
    # Print summary
    print("\n=== Normalization Statistics ===")
    print("\nState statistics:")
    print(f"  Mean: {state_stats['mean']}")
    print(f"  Std:  {state_stats['std']}")
    print(f"  Q01:  {state_stats['q01']}")
    print(f"  Q99:  {state_stats['q99']}")
    
    print("\nAction statistics:")
    print(f"  Mean: {action_stats['mean']}")
    print(f"  Std:  {action_stats['std']}")
    print(f"  Q01:  {action_stats['q01']}")
    print(f"  Q99:  {action_stats['q99']}")
    
    print(f"\n✓ Norm stats saved to {output_path}")


if __name__ == "__main__":
    main()
