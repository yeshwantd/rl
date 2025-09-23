#!/usr/bin/env python3
"""
Filter NPZ datasets to keep only episodes whose **final step reward**
meets a threshold. Reports how many episodes were saved vs discarded.

Usage:
  # Keep episodes with final reward >= 0 and save alongside originals
  python filter_by_final_reward.py --min-final-reward 0 file1.npz file2.npz

  # Keep strictly > 10.0, add custom suffix
  python filter_by_final_reward.py --min-final-reward 10.0 --strict --suffix "_final_gt10" file1.npz

  # Keep >= 0, save all outputs into a specific directory
  python filter_by_final_reward.py --min-final-reward 0 --output-dir ./filtered file1.npz file2.npz
"""

import os
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


def filter_file(path: str, min_final_reward: float, strict: bool,
                output_dir: str = None, suffix: str = None) -> str:
    data = np.load(path, allow_pickle=True)

    observations    = data["observations"]
    actions         = data["actions"]
    rewards         = data["rewards"]
    terminateds     = data["terminateds"]
    truncations     = data["truncateds"]
    infos           = data["infos"]
    frames          = data["frames"] if "frames" in data else None

    # Optional: keep episode_returns if present for reference (not used for filtering)
    episode_returns = data["episode_returns"] if "episode_returns" in data else None

    # Build episode ranges purely from end flags
    ranges = compute_episode_ranges(terminateds, truncations)

    # Final reward for each episode = reward at the end index
    end_indices = np.array([e for (_, e) in ranges], dtype=int)
    final_rewards = rewards[end_indices].astype(float)

    # Decide which episodes to keep based on FINAL reward
    keep_mask = final_rewards > min_final_reward if strict else final_rewards >= min_final_reward
    kept_idx = np.flatnonzero(keep_mask)

    kept_ranges = [ranges[i] for i in kept_idx]
    kept_final_rewards = final_rewards[kept_idx]

    num_total = len(ranges)
    num_kept = len(kept_ranges)
    num_discarded = num_total - num_kept

    # Build step indices to slice per-step arrays
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
        # Save final rewards used for filtering (per-episode)
        "episode_final_rewards": kept_final_rewards.astype(np.float32),
    }
    # If original had episode_returns, pass through the kept ones for reference
    if episode_returns is not None:
        out["episode_returns"] = np.asarray(episode_returns)[kept_idx].astype(np.float32)

    if frames is not None:
        out["frames"] = frames[step_indices]

    # Build output path
    base, ext = os.path.splitext(os.path.basename(path))
    op_tag = "GT" if strict else "GE"
    auto_suffix = f"_final{op_tag}{int(min_final_reward) if float(min_final_reward).is_integer() else min_final_reward}"
    final_suffix = suffix if suffix is not None else auto_suffix
    out_name = base + final_suffix + ext
    out_dir = output_dir if output_dir is not None else os.path.dirname(path)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)

    np.savez_compressed(out_path, **out)

    print(f"[saved] {out_path} | episodes kept={num_kept} discarded={num_discarded} total={num_total}")
    return out_path


def main():
    p = argparse.ArgumentParser(description="Filter NPZ datasets by FINAL step reward threshold.")
    p.add_argument("files", nargs="+", help="One or more .npz dataset files")
    p.add_argument("--min-final-reward", type=float, default=0.0,
                   help="Minimum FINAL step reward to keep an episode")
    p.add_argument("--strict", action="store_true", help="Use '>' instead of '>=' for the threshold")
    p.add_argument("--output-dir", type=str, default=None, help="Directory to save filtered files")
    p.add_argument("--suffix", type=str, default=None, help="Custom suffix for output files (e.g., '_filtered')")
    args = p.parse_args()

    for f in args.files:
        if not os.path.isfile(f):
            print(f"[skip] Not a file: {f}")
            continue
        filter_file(f, args.min_final_reward, args.strict, args.output_dir, args.suffix)


if __name__ == "__main__":
    main()