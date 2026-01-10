from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cracer_sim import CracerGymEnv  # noqa: E402


class QNetwork(nn.Module):
    """Standard Q-Network (for backwards compatibility)."""
    def __init__(
        self,
        obs_size: int,
        num_actions: int,
        hidden_sizes: Tuple[int, ...],
        use_layer_norm: bool = False
    ) -> None:
        super().__init__()
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
        return self.net(x)


class DuelingQNetwork(nn.Module):
    """Dueling DQN architecture with optional layer normalization."""
    def __init__(
        self,
        obs_size: int,
        num_actions: int,
        hidden_sizes: Tuple[int, ...],
        use_layer_norm: bool = True
    ) -> None:
        super().__init__()
        self.num_actions = num_actions

        # Shared feature extraction layers
        layers = []
        last_size = obs_size
        for hidden in hidden_sizes[:-1]:  # All but last hidden layer
            layers.append(nn.Linear(last_size, hidden))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden))
            layers.append(nn.ReLU())
            last_size = hidden
        self.feature_net = nn.Sequential(*layers)

        # Value stream
        value_layers = [nn.Linear(last_size, hidden_sizes[-1])]
        if use_layer_norm:
            value_layers.append(nn.LayerNorm(hidden_sizes[-1]))
        value_layers.extend([nn.ReLU(), nn.Linear(hidden_sizes[-1], 1)])
        self.value_stream = nn.Sequential(*value_layers)

        # Advantage stream
        adv_layers = [nn.Linear(last_size, hidden_sizes[-1])]
        if use_layer_norm:
            adv_layers.append(nn.LayerNorm(hidden_sizes[-1]))
        adv_layers.extend([nn.ReLU(), nn.Linear(hidden_sizes[-1], num_actions)])
        self.advantage_stream = nn.Sequential(*adv_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_net(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        # Combine using mean advantage baseline
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values


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
    parser = argparse.ArgumentParser(description="Run inference with a DQN checkpoint and record video")
    default_checkpoint = os.path.join(ROOT, "rl", "checkpoints", "best.pt")
    parser.add_argument("--checkpoint", type=str, default=default_checkpoint)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-episode-steps", type=int, default=5_000)
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--record-fps", type=float, default=0.0)
    parser.add_argument("--env-fps", type=int, default=60)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--no-render", action="store_true", help="Run without video recording")
    parser.add_argument("--render-live", action="store_true", help="Show game window during inference")
    return parser.parse_args()


def pick_output_path(base: str) -> str:
    if base:
        path = base
        if not os.path.isabs(path):
            path = os.path.join(ROOT, path)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(ROOT, "rl", "runs", f"cracer_{timestamp}.mp4")

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
        "use_dueling": False,
        "use_layer_norm": False,
    }

    if args.hidden_sizes:
        config["hidden_sizes"] = tuple(args.hidden_sizes)
    elif model_config and "hidden_sizes" in model_config:
        config["hidden_sizes"] = tuple(model_config["hidden_sizes"])

    if model_config:
        config["use_dueling"] = model_config.get("use_dueling", False)
        config["use_layer_norm"] = model_config.get("use_layer_norm", False)

    return config


def create_network(obs_size: int, num_actions: int, config: dict, device: str) -> nn.Module:
    """Create the appropriate network architecture based on config."""
    hidden_sizes = config["hidden_sizes"]
    use_dueling = config["use_dueling"]
    use_layer_norm = config["use_layer_norm"]

    if use_dueling:
        net = DuelingQNetwork(obs_size, num_actions, hidden_sizes, use_layer_norm)
    else:
        net = QNetwork(obs_size, num_actions, hidden_sizes, use_layer_norm)

    return net.to(device)


def main() -> None:
    args = parse_args()
    if args.frame_skip < 1:
        raise ValueError("--frame-skip must be >= 1")

    device = pick_device(args.device)
    print(f"Device: {device}")

    render_mode = "human" if args.render_live else ("rgb_array" if not args.no_render else None)

    env = CracerGymEnv(
        render_mode=render_mode,
        obs_mode="state",
        action_mode="discrete",
        fps=args.env_fps,
        seed=args.seed,
    )
    obs, info = env.reset(seed=args.seed)
    obs = np.asarray(obs, dtype=np.float32)

    checkpoint_path = args.checkpoint
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(ROOT, checkpoint_path)

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = load_checkpoint(checkpoint_path, device)

    model_config = resolve_model_config(args, checkpoint)
    print(f"Model config: hidden_sizes={model_config['hidden_sizes']}, "
          f"dueling={model_config['use_dueling']}, layer_norm={model_config['use_layer_norm']}")

    obs_size = env.observation_space.shape[0]
    num_actions = env.action_space.n

    q_net = create_network(obs_size, num_actions, model_config, device)
    q_net.load_state_dict(checkpoint["q_net"])
    q_net.eval()

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

    for episode in range(1, args.episodes + 1):
        episode_reward = 0.0
        episode_steps = 0
        max_stage = 1

        if writer:
            frame = env.render()
            if frame is not None:
                writer.append_data(frame)

        while episode_steps < args.max_episode_steps:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                q_values = q_net(obs_tensor)
                action = int(torch.argmax(q_values, dim=1).item())

            next_obs, reward, terminated, truncated, info = env.step(action)
            obs = np.asarray(next_obs, dtype=np.float32)
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

        print(f"Episode {episode}: steps={episode_steps}, reward={episode_reward:.1f}, "
              f"score={final_score:.0f}, stage={max_stage}")

        obs, info = env.reset()
        obs = np.asarray(obs, dtype=np.float32)

    if writer:
        writer.close()
    env.close()

    mean_reward = sum(total_rewards) / max(1, len(total_rewards))
    mean_steps = sum(total_steps) / max(1, len(total_steps))
    mean_score = sum(total_scores) / max(1, len(total_scores))
    max_stage_reached = max(stages_reached) if stages_reached else 1

    print("-" * 60)
    print(f"Summary ({len(total_rewards)} episodes):")
    print(f"  Mean reward: {mean_reward:.1f}")
    print(f"  Mean score:  {mean_score:.0f}")
    print(f"  Mean steps:  {mean_steps:.0f}")
    print(f"  Max stage:   {max_stage_reached}")

    if output_path:
        print(f"  Video saved: {output_path}")


if __name__ == "__main__":
    main()
