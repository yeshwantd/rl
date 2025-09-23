#!/usr/bin/env python3
"""
Manual data collection for Gymnasium LunarLander (single-window pygame, smooth rendering).

Examples:
  # Discrete, smooth 30 FPS
  python teleop.py --env-id LunarLander-v3 --out lunar_bc_discrete.npz --target-fps 30

  # Discrete + slower visuals (wall-clock only; data unchanged)
  python teleop.py --env-id LunarLander-v3 --out lunar_bc_slowpilot.npz --target-fps 30 --slowdown 3

  # Heavier control feel (changes action cadence)
  python teleop.py --env-id LunarLander-v3 --out lunar_bc_heavy.npz --target-fps 30 --action-repeat 2

  # Continuous + record frames
  python teleop.py --env-id LunarLanderContinuous-v3 --out lunar_bc_cont.npz --target-fps 24 --action-repeat 2 --record-frames
"""

import argparse
import time
from typing import List, Dict, Any, Tuple

import numpy as np
import gymnasium as gym
import pygame


# ---------------------------- Controller ------------------------------------- #

class KeyboardController:
    """
    Keyboard → action mapping for both discrete and continuous variants.

    Discrete (LunarLander-v3):
      0: no-op, 1: left, 2: main, 3: right (main has priority)
      Keys: UP, LEFT, RIGHT

    Continuous (LunarLanderContinuous-v3):
      action = [torque, main] in [-1, 1]
      UP/DOWN adjusts main, LEFT/RIGHT adjusts torque
      Hold SHIFT for finer adjustments; BACKSPACE to center.
    """
    def __init__(self, env_id: str):
        self.discrete = "Continuous" not in env_id
        # Continuous control state
        self._a = np.zeros(2, dtype=np.float32)
        self.step = 0.10
        self.step_fine = 0.03
        self.decay = 0.02

    def poll(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        keys = pygame.key.get_pressed()
        info: Dict[str, Any] = {}
        if self.discrete:
            act = 0
            if keys[pygame.K_UP]:
                act = 2
            elif keys[pygame.K_LEFT]:
                act = 3
            elif keys[pygame.K_RIGHT]:
                act = 1
            return np.array([float(act)], dtype=np.float32), info
        else:
            step = self.step_fine if (keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]) else self.step
            # torque (index 0)
            if keys[pygame.K_LEFT]:
                self._a[0] -= step
            if keys[pygame.K_RIGHT]:
                self._a[0] += step
            # main (index 1)
            if keys[pygame.K_UP]:
                self._a[1] += step
            if keys[pygame.K_DOWN]:
                self._a[1] -= step
            # quick center
            if keys[pygame.K_BACKSPACE]:
                self._a[:] = 0.0
            # decay toward 0 when no key is held
            lr = keys[pygame.K_LEFT] or keys[pygame.K_RIGHT]
            ud = keys[pygame.K_UP] or keys[pygame.K_DOWN]
            if not lr:
                self._a[0] -= np.sign(self._a[0]) * min(abs(self._a[0]), self.decay)
            if not ud:
                self._a[1] -= np.sign(self._a[1]) * min(abs(self._a[1]), self.decay)
            self._a = np.clip(self._a, -1.0, 1.0)
            info["continuous_action"] = self._a.copy()
            return self._a.copy(), info


# --------------------------- Pygame & Drawing -------------------------------- #

def init_pygame() -> Tuple[pygame.Surface, pygame.font.Font, pygame.time.Clock]:
    pygame.init()
    pygame.display.set_caption("LunarLander Teleop (single-window, smooth)")
    # SCALED handles HiDPI nicely; DOUBLEBUF reduces flicker/tearing.
    screen = pygame.display.set_mode((1200, 900), flags=pygame.SCALED | pygame.DOUBLEBUF)
    font = pygame.font.SysFont(None, 20)
    clock = pygame.time.Clock()
    return screen, font, clock


def frame_to_surface(frame: np.ndarray) -> pygame.Surface:
    # Gym returns HxWx3; pygame expects WxH for surfarray.make_surface
    return pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))


def draw_overlay(screen: pygame.Surface, font: pygame.font.Font, lines: List[str]) -> None:
    # draw simple text overlay in top-left
    y = 8
    for line in lines:
        surf = font.render(line, True, (240, 240, 240))
        screen.blit(surf, (8, y))
        y += 18


# ------------------------------ Main ----------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="LunarLander-v3",
                        help="LunarLander-v3 or LunarLanderContinuous-v3")
    parser.add_argument("--seed", type=int, default=None,
                        help="Base seed. If set, episode k uses seed=(base+k). If omitted, random per episode.")
    parser.add_argument("--out", type=str, required=True, help="Output .npz path")
    parser.add_argument("--append", action="store_true", help="Append to existing file if present")
    parser.add_argument("--target-fps", type=int, default=30, help="Visual FPS cap (lower = slower appearance)")
    parser.add_argument("--action-repeat", type=int, default=1, help="Repeat each action this many env steps")
    parser.add_argument("--record-frames", action="store_true", help="Also save RGB frames (warning: large)")
    parser.add_argument("--slowdown", type=float, default=1.0,
                        help="Wall-clock slowdown factor (does NOT change recorded data). 1.0=normal, 2.0=2x slower, etc.")
    args = parser.parse_args()

    # RNG for per-episode seeds (when no base seed is provided)
    rng = np.random.default_rng()

    def episode_seed_for(idx: int) -> int:
        # Deterministic sequence if base seed provided; else random uint31
        return int(args.seed + idx) if args.seed is not None else int(rng.integers(0, 2**31 - 1))

    # Wall-clock pacing only
    slowdown = max(1.0, args.slowdown)
    effective_fps = max(1, int(round(args.target_fps / slowdown)))

    # Create env that yields frames as arrays; we render those in our pygame window.
    env = gym.make(args.env_id, render_mode="rgb_array", disable_env_checker=True)

    # I/O and control surfaces
    screen, font, clock = init_pygame()
    controller = KeyboardController(args.env_id)

    # Buffers
    observations: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    rewards: List[float] = []
    terminateds: List[bool] = []
    truncateds: List[bool] = []
    infos: List[dict] = []
    episode_starts: List[bool] = []
    episode_returns: List[float] = []
    episode_seeds: List[int] = []
    frames: List[np.ndarray] = []

    # Episode state
    ep_idx = 0
    seed0 = episode_seed_for(ep_idx)
    obs, info = env.reset(seed=seed0)
    episode_starts.append(True)
    episode_seeds.append(seed0)
    ep_return = 0.0
    running = True
    paused = False
    last_frame = None
    pending_reset = False  # defer manual resets to keep arrays aligned

    last_save_time = time.time()
    autosave_every_sec = 30

    def pack_and_save(final=False):
        save_dict = {
            "observations": np.asarray(observations, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.float32),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "terminateds": np.asarray(terminateds, dtype=np.bool_),
            "truncateds": np.asarray(truncateds, dtype=np.bool_),
            "episode_starts": np.asarray(episode_starts, dtype=np.bool_),
            "episode_returns": np.asarray(episode_returns, dtype=np.float32),
            "episode_seeds": np.asarray(episode_seeds, dtype=np.int64),
            "infos": np.asarray(infos, dtype=object),  # requires allow_pickle to load
        }
        if frames:
            save_dict["frames"] = np.asarray(frames, dtype=np.uint8)

        if args.append:
            try:
                old = np.load(args.out, allow_pickle=True)
                for k, v in save_dict.items():
                    if k in old:
                        save_dict[k] = np.concatenate([old[k], v], axis=0)
            except FileNotFoundError:
                pass

        np.savez_compressed(args.out, **save_dict)
        if final:
            print(f"[saved] {args.out}  steps={len(rewards)}  episodes={len(episode_returns)}")

    while running:
        # --- Events ---
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif e.key == pygame.K_SPACE:
                    paused = not paused
                elif e.key == pygame.K_r:
                    # Defer manual reset to keep logging aligned
                    pending_reset = True

        if paused:
            if last_frame is not None:
                surf = frame_to_surface(last_frame)
                screen.blit(pygame.transform.smoothscale(surf, screen.get_size()), (0, 0))
            draw_overlay(screen, font, [
                f"[PAUSED] EP {ep_idx}  steps={len(rewards)}  episodes={len(episode_returns)}",
                "Controls: UP=main, LEFT/RIGHT=side (discrete) | arrows adjust (continuous)",
                "SHIFT=fine adjust, BACKSPACE=center, R=reset, SPACE=pause, Q/ESC=quit",
            ])
            pygame.display.flip()
            clock.tick(10)
            continue

        # Handle deferred manual reset (aligned)
        if pending_reset:
            if ep_return != 0.0:
                episode_returns.append(ep_return)
                ep_idx += 1
                ep_return = 0.0
            next_seed = episode_seed_for(ep_idx)
            obs, info = env.reset(seed=next_seed)
            episode_starts.append(True)
            episode_seeds.append(next_seed)
            last_frame = env.render()
            pending_reset = False

        # --- Action selection from keyboard ---
        act, ctrl_info = controller.poll()
        step_action = int(act[0]) if controller.discrete else act

        # --- Step env (with optional action repeat) ---
        for _ in range(max(1, args.action_repeat)):
            next_obs, reward, terminated, truncated, step_info = env.step(step_action)

            # Log transition BEFORE we potentially reset
            observations.append(np.asarray(obs, dtype=np.float32))
            actions.append(np.asarray(act, dtype=np.float32))
            rewards.append(float(reward))
            terminateds.append(bool(terminated))
            truncateds.append(bool(truncated))
            info_to_store = dict(step_info)
            info_to_store.update(ctrl_info)
            infos.append(info_to_store)

            ep_return += reward
            episode_starts.append(False)

            obs = next_obs

            if terminated or truncated:
                episode_returns.append(ep_return)
                ep_idx += 1
                ep_return = 0.0
                next_seed = episode_seed_for(ep_idx)
                obs, info = env.reset(seed=next_seed)
                episode_starts.append(True)
                episode_seeds.append(next_seed)
                break

        # --- Render current frame into our pygame window ---
        frame = env.render()  # np.ndarray (H, W, 3)
        if frame is not None:
            last_frame = frame
            surf = frame_to_surface(frame)
            screen.blit(pygame.transform.smoothscale(surf, screen.get_size()), (0, 0))

        # --- HUD ---
        draw_overlay(screen, font, [
            f"EP {ep_idx}  steps={len(rewards)}  episodes={len(episode_returns)}",
            f"Return (this ep): {ep_return:.1f}",
            f"Action: {step_action}",
            "R=reset  SPACE=pause  Q/ESC=quit",
        ])

        pygame.display.flip()
        clock.tick(effective_fps)

        # --- Optional frame recording ---
        if args.record_frames and last_frame is not None:
            frames.append(last_frame)

        # --- Autosave safety ---
        if time.time() - last_save_time > autosave_every_sec:
            pack_and_save(final=False)
            last_save_time = time.time()

    # Final save
    pack_and_save(final=True)

    pygame.quit()
    env.close()


if __name__ == "__main__":
    main()