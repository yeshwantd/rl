#!/usr/bin/env python3
"""
Merge multiple LunarLander .npz datasets into one.

Usage:
  python merge_npz.py -o merged_dataset.npz file1.npz file2.npz file3.npz ...
"""

import argparse
import os
import numpy as np
from typing import Dict, List

def load_npz(path: str) -> Dict[str, np.ndarray]:
    return dict(np.load(path, allow_pickle=True))

def merge_npz(files: List[str], out_path: str):
    # Collect all datasets
    datasets = [load_npz(f) for f in files]

    # Keys that are per-step (same length as number of steps)
    step_keys = ["observations", "actions", "rewards", "terminateds", "truncateds", "episode_starts", "infos"]
    # Keys that are per-episode
    ep_keys = ["episode_returns"]

    # Check if any file has frames
    has_frames = any("frames" in d for d in datasets)
    if has_frames:
        step_keys.append("frames")

    merged: Dict[str, List[np.ndarray]] = {k: [] for k in step_keys + ep_keys}

    for d in datasets:
        for k in step_keys:
            if k in d:
                merged[k].append(d[k])
        for k in ep_keys:
            if k in d:
                merged[k].append(d[k])

    # Concatenate
    final = {}
    for k in merged:
        if merged[k]:  # skip if empty
            final[k] = np.concatenate(merged[k], axis=0)

    np.savez_compressed(out_path, **final)
    print(f"[saved] {out_path} with {len(final['rewards'])} steps and {len(final['episode_returns'])} episodes")

def main():
    parser = argparse.ArgumentParser(description="Merge multiple NPZ datasets into one")
    parser.add_argument("files", nargs="+", help="Input .npz files")
    parser.add_argument("-o", "--out", required=True, help="Output .npz file")
    args = parser.parse_args()

    merge_npz(args.files, args.out)

if __name__ == "__main__":
    main()