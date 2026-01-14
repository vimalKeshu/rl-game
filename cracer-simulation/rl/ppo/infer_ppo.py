#!/usr/bin/env python3
"""
PPO Inference Script for Cracer Sim.

Runs a trained PPO agent and optionally records video.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from collections import deque
from typing import Deque, Tuple

import numpy as np
import torch
import torch.nn as nn

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


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO (must match training architecture)."""
    def __init__(
        self,
        obs_size: int,
        num_actions: int,
        hidden_sizes: Tuple[int, ...],
        use_layer_norm: bool = False,
        shared_backbone: bool = False
    ):
        super().__init__()
        self.num_actions = num_actions
        self.shared_backbone = shared_backbone

        def build_mlp(input_size: int, output_size: int, hidden: Tuple[int, ...]) -> nn.Module:
            layers = []
            last_size = input_size
            for h in hidden:
                layers.append(nn.Linear(last_size, h))
                if use_layer_norm:
                    layers.append(nn.LayerNorm(h))
                layers.append(nn.Tanh())
                last_size = h
            layers.append(nn.Linear(last_size, output_size))
            return nn.Sequential(*layers)

        if shared_backbone:
            self.backbone = build_mlp(obs_size, hidden_sizes[-1], hidden_sizes[:-1])
            self.actor_head = nn.Linear(hidden_sizes[-1], num_actions)
            self.critic_head = nn.Linear(hidden_sizes[-1], 1)
        else:
            self.actor = build_mlp(obs_size, num_actions, hidden_sizes)
            self.critic = build_mlp(obs_size, 1, hidden_sizes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns action logits and value estimate."""
        if self.shared_backbone:
            features = self.backbone(x)
            features = torch.tanh(features)
            action_logits = self.actor_head(features)
            value = self.critic_head(features)
        else:
            action_logits = self.actor(x)
            value = self.critic(x)
        return action_logits, value.squeeze(-1)

    def get_action(self, x: torch.Tensor, deterministic: bool = True) -> int:
        """Get action from policy."""
        action_logits, _ = self(x)
        if deterministic:
            return int(torch.argmax(action_logits, dim=1).item())
        else:
            probs = torch.softmax(action_logits, dim=1)
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
    parser = argparse.ArgumentParser(description="Run inference with a PPO checkpoint and record video")
    default_checkpoint = os.path.join(ROOT, "rl", "ppo", "checkpoints", "best.pt")
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
        path = os.path.join(ROOT, "rl", "ppo", "runs", f"ppo_{timestamp}.mp4")

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
        "hidden_sizes": (256, 256),
        "use_layer_norm": False,
        "shared_backbone": False,
        "frame_stack": 1,
        "normalize_obs": False,
    }

    if args.hidden_sizes:
        config["hidden_sizes"] = tuple(args.hidden_sizes)
    elif model_config and "hidden_sizes" in model_config:
        config["hidden_sizes"] = tuple(model_config["hidden_sizes"])

    if model_config:
        config["use_layer_norm"] = model_config.get("use_layer_norm", False)
        config["shared_backbone"] = model_config.get("shared_backbone", False)
        config["frame_stack"] = model_config.get("frame_stack", 1)
        config["normalize_obs"] = model_config.get("normalize_obs", False)

    return config


def create_network(obs_size: int, num_actions: int, config: dict, device: str) -> nn.Module:
    """Create the ActorCritic network based on config."""
    net = ActorCritic(
        obs_size=obs_size,
        num_actions=num_actions,
        hidden_sizes=config["hidden_sizes"],
        use_layer_norm=config["use_layer_norm"],
        shared_backbone=config["shared_backbone"],
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
          f"shared_backbone={model_config['shared_backbone']}, "
          f"frame_stack={frame_stack_size}, normalize_obs={normalize_obs}")

    base_obs_size = env.observation_space.shape[0]
    num_actions = env.action_space.n
    obs_size = base_obs_size * frame_stack_size

    # Frame stacker
    frame_stacker = FrameStack(frame_stack_size, base_obs_size) if frame_stack_size > 1 else None

    # Observation normalizer (load from checkpoint if available)
    obs_normalizer = None
    if normalize_obs and "obs_normalizer" in checkpoint:
        obs_normalizer = ObservationNormalizer(obs_size)
        obs_normalizer.load_state(checkpoint["obs_normalizer"])
        print(f"Loaded observation normalizer (trained on {obs_normalizer.count} samples)")

    actor_critic = create_network(obs_size, num_actions, model_config, device)
    actor_critic.load_state_dict(checkpoint["actor_critic"])
    actor_critic.eval()

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
                action = actor_critic.get_action(obs_tensor, deterministic=deterministic)

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
