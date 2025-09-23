#!/usr/bin/env python3
"""
Replay one episode from a teleop-collected LunarLander dataset (.npz).

The dataset must include:
  - observations, actions, rewards, terminateds, truncateds  (per-step arrays)
  - episode_returns (per-episode)
  - episode_seeds (per-episode; produced by the updated teleop.py)

Usage:
  python replay_from_npz.py --data lunar_bc_discrete.npz --ep-idx 0 --env-id LunarLander-v3
  python replay_from_npz.py --data lunar_bc_cont.npz --ep-idx 3 --env-id LunarLanderContinuous-v3 --target-fps 30 --slowdown 2
"""

import argparse
from typing import List, Tuple

import numpy as np
import gymnasium as gym
import pygame


# ---------- Episode utilities ----------

def compute_episode_ranges(terminateds: np.ndarray, truncateds: np.ndarray) -> List[Tuple[int, int]]:
    """
    Return a list of (start_idx, end_idx) inclusive for each completed episode,
    using ONLY end flags. Drops any trailing partial episode.
    """
    term = np.asarray(terminateds, dtype=bool)
    trunc = np.asarray(truncateds, dtype=bool)
    ends = np.flatnonzero(term | trunc).astype(int)

    ranges: List[Tuple[int, int]] = []
    start = 0
    for e in ends:
        ranges.append((start, e))
        start = e + 1
    return ranges


# ---------- Pygame helpers ----------

def init_pygame_window(width=960, height=720):
    pygame.init()
    pygame.display.set_caption("LunarLander Episode Replay")
    screen = pygame.display.set_mode((width, height), flags=pygame.SCALED | pygame.DOUBLEBUF)
    font = pygame.font.SysFont(None, 20)
    clock = pygame.time.Clock()
    return screen, font, clock

def frame_to_surface(frame: np.ndarray) -> pygame.Surface:
    # H x W x 3 -> W x H x 3 for pygame
    return pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))

def draw_overlay(screen, font, lines: List[str]):
    y = 8
    for line in lines:
        surf = font.render(line, True, (240, 240, 240))
        screen.blit(surf, (8, y))
        y += 18


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(description="Replay one episode from a teleop .npz using saved seed.")
    ap.add_argument("--data", required=True, help="Path to teleop-collected .npz (with episode_seeds)")
    ap.add_argument("--ep-idx", type=int, default=0, help="Episode index to replay (0-based)")
    ap.add_argument("--env-id", type=str, default="LunarLander-v3",
                    help="Gymnasium env id (e.g., LunarLander-v3 or LunarLanderContinuous-v3)")
    ap.add_argument("--target-fps", type=int, default=30, help="Visual FPS cap")
    ap.add_argument("--slowdown", type=float, default=1.0, help="Wall-clock slowdown factor (1=normal, 2=2x slower)")
    ap.add_argument("--window-size", type=int, nargs=2, default=[960, 720], help="Window size W H")
    ap.add_argument("--check-obs", action="store_true",
                    help="Compare env observations to stored ones and print mean abs error periodically")
    args = ap.parse_args()

    # Load dataset
    d = np.load(args.data, allow_pickle=True)
    observations  = d["observations"]
    actions       = d["actions"]
    rewards       = d["rewards"]
    terminateds   = d["terminateds"]
    truncateds    = d["truncateds"]
    episode_returns = d["episode_returns"]
    if "episode_seeds" not in d:
        raise KeyError("Dataset is missing 'episode_seeds'. Re-collect with the updated teleop.py.")
    episode_seeds = d["episode_seeds"]

    # Compute completed episode ranges and align with seeds/returns defensively
    ranges = compute_episode_ranges(terminateds, truncateds)
    num_eps = min(len(ranges), len(episode_returns), len(episode_seeds))
    if len(ranges) != num_eps:
        # Drop any extra ranges beyond what we have returns/seeds for
        ranges = ranges[:num_eps]

    if not (0 <= args.ep_idx < num_eps):
        raise IndexError(f"--ep-idx {args.ep_idx} out of range [0, {num_eps-1}]")

    (s_idx, e_idx) = ranges[args.ep_idx]
    ep_seed = int(episode_seeds[args.ep_idx])
    ep_len = e_idx - s_idx + 1
    print(f"[info] episode {args.ep_idx}: steps={ep_len}  return={float(episode_returns[args.ep_idx]):.2f}  seed={ep_seed}")

    # Determine action space type
    is_discrete = "Continuous" not in args.env_id

    # Make env & reset with recorded episode seed
    env = gym.make(args.env_id, render_mode="rgb_array", disable_env_checker=True)
    obs, info = env.reset(seed=ep_seed)

    # Pygame window
    screen, font, clock = init_pygame_window(*args.window_size)
    effective_fps = max(1, int(round(args.target_fps / max(1.0, args.slowdown))))

    # Replay loop
    total_reward = 0.0
    step = s_idx
    running = True
    while running and step <= e_idx:
        # Events
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN and e.key in (pygame.K_ESCAPE, pygame.K_q):
                running = False

        # Extract stored step data
        s_stored = observations[step]
        a_stored = actions[step]
        r_stored = float(rewards[step])

        # Convert stored action for discrete envs (floats back to ints)
        if is_discrete:
            # actions were stored as shape (1,) float32 or scalar float
            action = int(a_stored[0]) if np.ndim(a_stored) > 0 else int(a_stored)
        else:
            action = a_stored  # continuous: shape (2,), float

        # Step
        obs, rew, terminated, truncated, info = env.step(action)
        total_reward += rew

        # Optional obs check
        if args.check_obs and (step - s_idx) % 20 == 0:
            try:
                mae = float(np.mean(np.abs(obs - s_stored)))
                print(f"[step {step - s_idx:4d}/{ep_len}] obs MAE vs stored: {mae:.4f}")
            except Exception:
                pass

        # Render
        frame = env.render()
        if frame is not None:
            surf = frame_to_surface(frame)
            screen.blit(pygame.transform.smoothscale(surf, screen.get_size()), (0, 0))
        draw_overlay(screen, font, [
            f"{args.env_id} | ep={args.ep_idx} | step {step - s_idx + 1}/{ep_len}",
            f"stored r_t={r_stored:.2f}   env sum R={total_reward:.2f}",
            "Q/ESC: quit",
        ])
        pygame.display.flip()
        clock.tick(effective_fps)

        step += 1

        # If env ends early, reset to same seed so the visuals keep going.
        if terminated or truncated:
            obs, info = env.reset(seed=ep_seed)

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()