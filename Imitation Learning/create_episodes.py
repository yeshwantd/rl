#!/usr/bin/env python3
"""
Split a LunarLander dataset into episodes and save them.

- Uses ONLY (terminateds | truncateds) to define episode boundaries.
- Always drops any trailing partial episode (no terminal at the end).
- Saves an object-array of episodes (each is a list of (obs, act, rew) tuples)
  plus the aligned episode_returns.

Usage:
  python create_episodes.py input.npz
  python create_episodes.py input.npz -o output_episodes.npz
"""

import argparse
import os
from typing import Any, Dict, List, Tuple

import numpy as np


def load_npz(path: str) -> Dict[str, Any]:
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def compute_episode_ranges(terminateds: np.ndarray, truncations: np.ndarray) -> List[Tuple[int, int]]:
    """
    Return a list of (start_idx, end_idx) inclusive using only end flags.
    Drops any trailing partial episode (i.e., last step is non-terminal).
    """
    term = np.asarray(terminateds, dtype=bool)
    trunc = np.asarray(truncations, dtype=bool)
    end_idxs = np.flatnonzero(term | trunc)

    ranges: List[Tuple[int, int]] = []
    start = 0
    for end in end_idxs:
        ranges.append((start, int(end)))
        start = int(end) + 1
    return ranges  # no trailing partial included


def split_episodes(arr: Dict[str, Any]) -> List[List[Tuple[np.ndarray, np.ndarray, float]]]:
    """
    Split dataset into episodes of (observation, action, reward) using end flags only.
    Always drops a trailing partial episode.
    """
    observations = arr["observations"]
    actions = arr["actions"]
    rewards = arr["rewards"]
    terminations = arr["terminateds"]
    truncations = arr["truncateds"]

    N = len(rewards)
    assert len(observations) == N and len(actions) == N and len(terminations) == N and len(truncations) == N, \
        "Length mismatch among step arrays."

    ranges = compute_episode_ranges(terminations, truncations)

    episodes: List[List[Tuple[np.ndarray, np.ndarray, float]]] = []
    for (s, e) in ranges:
        ep = [(observations[i], actions[i], float(rewards[i])) for i in range(s, e + 1)]
        episodes.append(ep)
    return episodes


def save_episodes(path: str,
                  episodes: List[List[Tuple[np.ndarray, np.ndarray, float]]],
                  episode_returns: np.ndarray):
    """
    Save episodes and their returns to a .npz file.
    """
    np.savez_compressed(
        path,
        episodes=np.array(episodes, dtype=object),
        episode_returns=np.asarray(episode_returns, dtype=np.float32),
    )
    print(f"[saved] {path}  episodes={len(episodes)}")


def main():
    parser = argparse.ArgumentParser(description="Split a LunarLander dataset into episodes and save them.")
    parser.add_argument("input", help="Input .npz dataset file")
    parser.add_argument("-o", "--out", default=None, help="Output .npz path (default: <input>_episodes.npz)")
    args = parser.parse_args()

    data = load_npz(args.input)

    # Split into episodes and align with episode_returns (drop extra if mismatch)
    episodes = split_episodes(data)
    ep_returns = np.asarray(data["episode_returns"], dtype=float)

    # Align (defensive): use the minimum count between computed episodes and provided returns
    n = min(len(episodes), len(ep_returns))
    episodes = episodes[:n]
    ep_returns = ep_returns[:n]

    # Default output path
    out_path = args.out
    if out_path is None:
        base, ext = os.path.splitext(args.input)
        out_path = f"{base}_episodes.npz"

    print(f"[info] input_steps={len(data['rewards'])}  episodes={len(episodes)}  aligned_returns={len(ep_returns)}")
    save_episodes(out_path, episodes, ep_returns)


if __name__ == "__main__":
    main()