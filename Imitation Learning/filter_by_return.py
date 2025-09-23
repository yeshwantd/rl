#!/usr/bin/env python3
"""
Filter NPZ datasets to keep only episodes with returns above a threshold.

Usage:
  # Keep only episodes with returns >= 0 and save alongside originals
  python filter_by_return.py --min-return 0 file1.npz file2.npz

  # Keep strictly >200, add custom suffix
  python filter_by_return.py --min-return 200 --strict --suffix "_gt200" file1.npz

  # Keep >=0, save all outputs into a specific directory
  python filter_by_return.py --min-return 0 --output-dir ./filtered file1.npz file2.npz
"""

import os
import sys
import argparse
import numpy as np
from typing import List, Tuple

def compute_episode_ranges(terminateds: np.ndarray, truncations: np.ndarray) -> List[Tuple[int, int]]:
    """Return a list of (start_idx, end_idx) inclusive, using only end flags."""
    term = np.asarray(terminateds, dtype=bool)
    trunc = np.asarray(truncations, dtype=bool)
    end_idx = np.flatnonzero(term | trunc)

    ranges: List[Tuple[int, int]] = []
    start = 0
    for e in end_idx:
        ranges.append((start, int(e)))
        start = int(e) + 1
    return ranges

def filter_file(path: str, min_return: float, strict: bool, output_dir: str = None, suffix: str = None) -> str:
    data = np.load(path, allow_pickle=True)

    observations    = data["observations"]
    actions         = data["actions"]
    rewards         = data["rewards"]
    terminateds     = data["terminateds"]
    truncations     = data["truncateds"]
    infos           = data["infos"]
    episode_returns = data["episode_returns"]
    frames          = data["frames"] if "frames" in data else None

    # Build episode ranges purely from end flags
    ranges = compute_episode_ranges(terminateds, truncations)

    # Align returns to ranges
    num_eps = min(len(ranges), len(episode_returns))
    ranges = ranges[:num_eps]
    returns = np.asarray(episode_returns[:num_eps], dtype=float)

    # Decide which episodes to keep
    keep_mask = returns > min_return if strict else returns >= min_return
    kept_ranges = [rng for rng, keep in zip(ranges, keep_mask) if keep]
    kept_returns = returns[keep_mask]

    # Build step indices
    step_indices: List[int] = []
    new_episode_starts: List[bool] = []
    for (s, e) in kept_ranges:
        step_indices.extend(range(s, e + 1))
        new_episode_starts.extend([True] + [False] * (e - s))

    step_indices = np.array(step_indices, dtype=int)
    new_episode_starts = np.array(new_episode_starts, dtype=bool)

    # Slice arrays
    out = {
        "observations": observations[step_indices],
        "actions": actions[step_indices],
        "rewards": rewards[step_indices],
        "terminateds": terminateds[step_indices],
        "truncateds": truncations[step_indices],
        "episode_starts": new_episode_starts,
        "infos": infos[step_indices],
        "episode_returns": kept_returns.astype(np.float32),
    }
    if frames is not None:
        out["frames"] = frames[step_indices]

    # Build output path
    base, ext = os.path.splitext(os.path.basename(path))
    op_tag = "GT" if strict else "GE"
    auto_suffix = f"_ret{op_tag}{int(min_return) if float(min_return).is_integer() else min_return}"
    final_suffix = suffix if suffix is not None else auto_suffix
    out_name = base + final_suffix + ext
    out_dir = output_dir if output_dir is not None else os.path.dirname(path)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)

    np.savez_compressed(out_path, **out)
    return out_path

def main():
    p = argparse.ArgumentParser(description="Filter NPZ datasets by episode return threshold.")
    p.add_argument("files", nargs="+", help="One or more .npz dataset files")
    p.add_argument("--min-return", type=float, default=0.0, help="Minimum episode return to keep")
    p.add_argument("--strict", action="store_true", help="Use '>' instead of '>='")
    p.add_argument("--output-dir", type=str, default=None, help="Directory to save filtered files")
    p.add_argument("--suffix", type=str, default=None, help="Custom suffix for output files (e.g., '_filtered')")
    args = p.parse_args()

    for f in args.files:
        if not os.path.isfile(f):
            print(f"[skip] Not a file: {f}")
            continue
        out = filter_file(f, args.min_return, args.strict, args.output_dir, args.suffix)
        print(f"[saved] {out}")

if __name__ == "__main__":
    main()