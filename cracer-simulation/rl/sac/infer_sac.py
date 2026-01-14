#!/usr/bin/env python3
"""
SAC Inference Script for Cracer Sim.

Runs a trained SAC agent and optionally records video.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from collections import deque
from typing import Deque, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cracer_sim import CracerGymEnv  # noqa: E402


class FrameStack:
    """Stacks multiple frames for temporal context."""
    def __init__(self, num_frames: int, obs_size: int):
        self.num_frames = num_frames
        self.obs_size = obs_size
        self.frames: Deque[np.ndarray] = deque(maxlen=num_frames)

    def reset(self, obs: np.ndarray) -> np.ndarray:
        for _ in range(self.num_frames):
            self.frames.append(obs.copy())
        return self.get()

    def push(self, obs: np.ndarray) -> np.ndarray:
        self.frames.append(obs.copy())
        return self.get()

    def get(self) -> np.ndarray:
        return np.concatenate(list(self.frames), axis=0)


class ObservationNormalizer:
    """Running mean/std normalization for observations (inference mode)."""
    def __init__(self, obs_size: int, clip: float = 10.0, epsilon: float = 1e-8):
        self.obs_size = obs_size
        self.clip = clip
        self.epsilon = epsilon
        self.mean = np.zeros(obs_size, dtype=np.float64)
        self.var = np.ones(obs_size, dtype=np.float64)
        self.count = 0

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation using stored statistics."""
        normalized = (obs - self.mean.astype(np.float32)) / (np.sqrt(self.var.astype(np.float32)) + self.epsilon)
        return np.clip(normalized, -self.clip, self.clip)

    def load_state(self, state: dict):
        """Load normalizer state."""
        self.mean = state["mean"]
        self.var = state["var"]
        self.count = state["count"]


class PolicyNetwork(nn.Module):
    """Policy network for SAC-Discrete (must match training architecture)."""
    def __init__(self, obs_size: int, num_actions: int, hidden_sizes: Tuple[int, ...],
                 use_layer_norm: bool = True):
        super().__init__()
        self.num_actions = num_actions

        layers = []
        last_size = obs_size
        for hidden in hidden_sizes:
            layers.append(nn.Linear(last_size, hidden))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden))
            layers.append(nn.ReLU())
            last_size = hidden
        layers.append(nn.Linear(last_size, num_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns action logits."""
        return self.net(x)

    def get_action(self, x: torch.Tensor, deterministic: bool = True) -> int:
        """Get action from policy."""
        logits = self(x)
        if deterministic:
            return int(torch.argmax(logits, dim=-1).item())
        else:
            probs = F.softmax(logits, dim=-1)
            return int(torch.multinomial(probs, 1).item())


def pick_device(choice: str) -> str:
    if choice != "auto":
        return choice
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_checkpoint(path: str, device: str):
    return torch.load(path, map_location=device, weights_only=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a SAC checkpoint and record video")
    default_checkpoint = os.path.join(ROOT, "rl", "sac", "checkpoints", "best.pt")
    parser.add_argument("--checkpoint", type=str, default=default_checkpoint)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-episode-steps", type=int, default=10_000)
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--record-fps", type=float, default=0.0)
    parser.add_argument("--env-fps", type=int, default=60)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=None, help="Random seed (None = random each episode)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--no-render", action="store_true", help="Run without video recording")
    parser.add_argument("--render-live", action="store_true", help="Show game window during inference")
    parser.add_argument("--random-seeds", action="store_true", help="Use random seed for each episode")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic action selection")
    return parser.parse_args()


def pick_output_path(base: str) -> str:
    if base:
        path = base
        if not os.path.isabs(path):
            path = os.path.join(ROOT, path)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(ROOT, "rl", "sac", "runs", f"sac_{timestamp}.mp4")

    base_dir = os.path.dirname(path)
    if base_dir:
        os.makedirs(base_dir, exist_ok=True)

    if not os.path.exists(path):
        return path

    root, ext = os.path.splitext(path)
    for idx in range(1, 1000):
        candidate = f"{root}_{idx}{ext}"
        if not os.path.exists(candidate):
            return candidate
    raise RuntimeError("Could not find a free filename for output video")


def make_writer(path: str, fps: float):
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError("imageio is required to write video files") from exc

    try:
        return imageio.get_writer(path, fps=fps, codec="libx264")
    except Exception:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".gif":
            return imageio.get_writer(path, fps=fps)
        raise RuntimeError(
            "Failed to open video writer. Install imageio-ffmpeg or use --output with .gif extension."
        )


def resolve_model_config(args, checkpoint) -> dict:
    """Extract model configuration from checkpoint or use defaults."""
    model_config = checkpoint.get("model_config") if isinstance(checkpoint, dict) else None

    config = {
        "hidden_sizes": (512, 512, 256),
        "use_layer_norm": True,
        "frame_stack": 1,
        "normalize_obs": False,
    }

    if args.hidden_sizes:
        config["hidden_sizes"] = tuple(args.hidden_sizes)
    elif model_config and "hidden_sizes" in model_config:
        config["hidden_sizes"] = tuple(model_config["hidden_sizes"])

    if model_config:
        config["use_layer_norm"] = model_config.get("use_layer_norm", True)
        config["frame_stack"] = model_config.get("frame_stack", 1)
        config["normalize_obs"] = model_config.get("normalize_obs", False)

    return config


def create_network(obs_size: int, num_actions: int, config: dict, device: str) -> nn.Module:
    """Create the PolicyNetwork based on config."""
    net = PolicyNetwork(
        obs_size=obs_size,
        num_actions=num_actions,
        hidden_sizes=config["hidden_sizes"],
        use_layer_norm=config["use_layer_norm"],
    )
    return net.to(device)


def main() -> None:
    args = parse_args()
    if args.frame_skip < 1:
        raise ValueError("--frame-skip must be >= 1")

    device = pick_device(args.device)
    print(f"Device: {device}")

    render_mode = "human" if args.render_live else ("rgb_array" if not args.no_render else None)

    initial_seed = args.seed if args.seed is not None else random.randint(0, 1_000_000)

    env = CracerGymEnv(
        render_mode=render_mode,
        obs_mode="state",
        action_mode="discrete",
        fps=args.env_fps,
        seed=initial_seed,
    )
    obs, info = env.reset(seed=initial_seed)

    checkpoint_path = args.checkpoint
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(ROOT, checkpoint_path)

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = load_checkpoint(checkpoint_path, device)

    model_config = resolve_model_config(args, checkpoint)
    frame_stack_size = model_config.get("frame_stack", 1)
    normalize_obs = model_config.get("normalize_obs", False)

    print(f"Model config: hidden_sizes={model_config['hidden_sizes']}, "
          f"layer_norm={model_config['use_layer_norm']}, "
          f"frame_stack={frame_stack_size}, normalize_obs={normalize_obs}")

    if "alpha" in checkpoint:
        print(f"Trained alpha (entropy coef): {checkpoint['alpha']:.4f}")

    base_obs_size = env.observation_space.shape[0]
    num_actions = env.action_space.n
    obs_size = base_obs_size * frame_stack_size

    # Frame stacker
    frame_stacker = FrameStack(frame_stack_size, base_obs_size) if frame_stack_size > 1 else None

    # Observation normalizer
    obs_normalizer = None
    if normalize_obs and "obs_normalizer" in checkpoint:
        obs_normalizer = ObservationNormalizer(obs_size)
        obs_normalizer.load_state(checkpoint["obs_normalizer"])
        print(f"Loaded observation normalizer (trained on {obs_normalizer.count} samples)")

    policy = create_network(obs_size, num_actions, model_config, device)
    policy.load_state_dict(checkpoint["policy"])
    policy.eval()

    writer = None
    output_path = None
    if not args.no_render and not args.render_live:
        output_path = pick_output_path(args.output)
        record_fps = args.record_fps if args.record_fps > 0 else float(args.env_fps) / float(args.frame_skip)
        writer = make_writer(output_path, record_fps)

    total_rewards = []
    total_steps = []
    total_scores = []
    stages_reached = []

    deterministic = not args.stochastic
    print(f"Action selection: {'deterministic' if deterministic else 'stochastic'}")

    for episode in range(1, args.episodes + 1):
        # Use random seed for generalization testing
        if args.random_seeds or args.seed is None:
            episode_seed = random.randint(0, 1_000_000)
        else:
            episode_seed = args.seed

        obs, info = env.reset(seed=episode_seed)
        obs = np.asarray(obs, dtype=np.float32)

        # Initialize frame stacker
        if frame_stacker:
            stacked_obs = frame_stacker.reset(obs)
        else:
            stacked_obs = obs

        episode_reward = 0.0
        episode_steps = 0
        max_stage = 1

        if writer:
            frame = env.render()
            if frame is not None:
                writer.append_data(frame)

        while episode_steps < args.max_episode_steps:
            # Normalize observation if normalizer is available
            obs_for_network = obs_normalizer.normalize(stacked_obs) if obs_normalizer else stacked_obs

            with torch.no_grad():
                obs_tensor = torch.tensor(obs_for_network, dtype=torch.float32, device=device).unsqueeze(0)
                action = policy.get_action(obs_tensor, deterministic=deterministic)

            next_obs, reward, terminated, truncated, info = env.step(action)
            next_obs = np.asarray(next_obs, dtype=np.float32)

            # Update frame stacker
            if frame_stacker:
                stacked_obs = frame_stacker.push(next_obs)
            else:
                stacked_obs = next_obs

            episode_reward += float(reward)
            episode_steps += 1

            current_stage = info.get("stage", 1)
            max_stage = max(max_stage, current_stage)

            if writer and episode_steps % args.frame_skip == 0:
                frame = env.render()
                if frame is not None:
                    writer.append_data(frame)

            if args.render_live:
                env.render()

            if terminated or truncated:
                break

        final_score = info.get("score", 0)
        total_rewards.append(episode_reward)
        total_steps.append(episode_steps)
        total_scores.append(final_score)
        stages_reached.append(max_stage)

        print(f"Episode {episode} (seed={episode_seed}): steps={episode_steps}, reward={episode_reward:.1f}, "
              f"score={final_score:.0f}, stage={max_stage}")

    if writer:
        writer.close()
    env.close()

    mean_reward = sum(total_rewards) / max(1, len(total_rewards))
    mean_steps = sum(total_steps) / max(1, len(total_steps))
    mean_score = sum(total_scores) / max(1, len(total_scores))
    max_stage_reached = max(stages_reached) if stages_reached else 1
    mean_stage = sum(stages_reached) / max(1, len(stages_reached))

    print("-" * 60)
    print(f"Summary ({len(total_rewards)} episodes):")
    print(f"  Mean reward: {mean_reward:.1f}")
    print(f"  Mean score:  {mean_score:.0f}")
    print(f"  Mean steps:  {mean_steps:.0f}")
    print(f"  Mean stage:  {mean_stage:.1f}")
    print(f"  Max stage:   {max_stage_reached}")

    if output_path:
        print(f"  Video saved: {output_path}")


if __name__ == "__main__":
    main()
