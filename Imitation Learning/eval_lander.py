#!/usr/bin/env python3
"""
Evaluate a trained BasicPolicy on Gymnasium LunarLander-v3.

- Loads a checkpoint (saved from training) and runs N episodes.
- Reports per-episode success/failure and overall success percentage.
- "Success" = episode terminated (not truncated) with BOTH legs in contact
  at the end (lander settled between the flag posts).

Usage:
  python eval_lander.py --ckpt ./checkpoints/policy_epoch_040.pt --episodes 5 --display --no-greedy
"""

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

from models import BasicPolicy

# --------- Helpers --------- #

def load_checkpoint(model: nn.Module, ckpt_path: str, device: torch.device) -> None:
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)


import torch
import torch.nn.functional as F

def select_action(logits: torch.Tensor, greedy: bool = True) -> int:
    if greedy:
        return int(torch.argmax(logits, dim=-1).item())
    else:
        probs = F.softmax(logits, dim=-1)  # [1, num_actions]
        dist = torch.distributions.Categorical(probs)
        return int(dist.sample().item())


def is_success(final_obs: np.ndarray, terminated: bool, truncated: bool) -> bool:
    """
    Define success as: episode terminated (not truncated) AND both legs in contact.
    In LunarLander, observation[-2:] are (left_leg_contact, right_leg_contact) ∈ {0,1}.
    """
    if not terminated or truncated:
        return False
    left_contact = final_obs[6] > 0.5
    right_contact = final_obs[7] > 0.5
    return bool(left_contact and right_contact)


# --------- Main --------- #

def main():
    parser = argparse.ArgumentParser(description="Evaluate a BasicPolicy on LunarLander-v3.")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint .pt/.pth saved during training")
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes to run")
    parser.add_argument("--env-id", type=str, default="LunarLander-v3", help="Gymnasium env id (discrete)")
    parser.add_argument("--seed", type=int, default=None, help="Optional base seed for env.reset")
    parser.add_argument("--display", action="store_true", help="Render the environment window")
    parser.add_argument("--device", type=str, default=None, help="Force device (cpu/cuda). Default: auto")
    parser.add_argument("--no-greedy", action="store_true", help="Use stochastic policy (no greedy action)")
    args = parser.parse_args()

    if "Continuous" in args.env_id:
        raise ValueError("This script expects a DISCRETE LunarLander policy (4 actions). Use LunarLander-v3.")

    # Device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Model
    model = BasicPolicy().to(device)
    load_checkpoint(model, args.ckpt, device)
    model.eval()

    # Env
    render_mode = "human" if args.display else None
    env = gym.make(args.env_id, render_mode=render_mode, disable_env_checker=True)

    success_count = 0
    total_return = 0.0

    for ep in range(args.episodes):
        # Seed per episode if provided
        ep_seed = None if args.seed is None else int(args.seed + ep)
        obs, info = env.reset(seed=ep_seed)
        done = False
        ep_ret = 0.0

        while not done:
            # Prepare obs
            obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(obs_t)
            action = select_action(logits, greedy=not args.no_greedy)

            # Step
            next_obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += float(reward)
            done = terminated or truncated
            obs = next_obs

            # Render if requested (env handles it with render_mode="human")
            if args.display and render_mode == "human":
                # no explicit sleep; Gym handles frame pacing
                pass

        # Success check uses the final observation we ended on
        ok = is_success(obs, terminated, truncated)
        success_count += int(ok)
        total_return += ep_ret
        print(f"Episode {ep+1:3d}/{args.episodes}: "
              f"{'SUCCESS' if ok else 'FAIL   '} | return={ep_ret:.1f}")

    success_pct = 100.0 * success_count / max(1, args.episodes)
    avg_return = total_return / max(1, args.episodes)
    print(f"\nSummary over {args.episodes} episodes:")
    print(f"  Success: {success_count}/{args.episodes} ({success_pct:.1f}%)")
    print(f"  Average return: {avg_return:.1f}")

    env.close()


if __name__ == "__main__":
    main()