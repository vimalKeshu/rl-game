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
    def __init__(self, obs_size: int, num_actions: int, hidden_sizes: Tuple[int, ...]) -> None:
        super().__init__()
        layers = []
        last_size = obs_size
        for hidden in hidden_sizes:
            layers.append(nn.Linear(last_size, hidden))
            layers.append(nn.ReLU())
            last_size = hidden
        layers.append(nn.Linear(last_size, num_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def pick_device(choice: str) -> str:
    if choice != "auto":
        return choice
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_checkpoint(path: str, device: str):
    return torch.load(path, map_location=device)


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


def resolve_hidden_sizes(args, checkpoint) -> Tuple[int, ...]:
    if args.hidden_sizes:
        return tuple(args.hidden_sizes)
    model_config = checkpoint.get("model_config") if isinstance(checkpoint, dict) else None
    if model_config and "hidden_sizes" in model_config:
        return tuple(model_config["hidden_sizes"])
    return (256, 256)


def main() -> None:
    args = parse_args()
    if args.frame_skip < 1:
        raise ValueError("--frame-skip must be >= 1")

    device = pick_device(args.device)

    env = CracerGymEnv(
        render_mode="rgb_array",
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
    checkpoint = load_checkpoint(checkpoint_path, device)
    hidden_sizes = resolve_hidden_sizes(args, checkpoint)

    obs_size = env.observation_space.shape[0]
    num_actions = env.action_space.n

    q_net = QNetwork(obs_size, num_actions, hidden_sizes).to(device)
    q_net.load_state_dict(checkpoint["q_net"])
    q_net.eval()

    output_path = pick_output_path(args.output)
    record_fps = args.record_fps if args.record_fps > 0 else float(args.env_fps) / float(args.frame_skip)
    writer = make_writer(output_path, record_fps)

    total_rewards = []
    total_steps = []

    for episode in range(1, args.episodes + 1):
        episode_reward = 0.0
        episode_steps = 0

        frame = env.render()
        if frame is not None:
            writer.append_data(frame)

        while True:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                q_values = q_net(obs_tensor)
                action = int(torch.argmax(q_values, dim=1).item())

            next_obs, reward, terminated, truncated, info = env.step(action)
            obs = np.asarray(next_obs, dtype=np.float32)
            episode_reward += float(reward)
            episode_steps += 1

            if episode_steps % args.frame_skip == 0:
                frame = env.render()
                if frame is not None:
                    writer.append_data(frame)

            if terminated or truncated:
                break

        total_rewards.append(episode_reward)
        total_steps.append(episode_steps)

        obs, info = env.reset()
        obs = np.asarray(obs, dtype=np.float32)

    writer.close()
    env.close()

    mean_reward = sum(total_rewards) / max(1, len(total_rewards))
    mean_steps = sum(total_steps) / max(1, len(total_steps))
    print(f"Saved video to {output_path}")
    print(f"Episodes: {len(total_rewards)} mean_reward={mean_reward:.1f} mean_steps={mean_steps:.1f}")


if __name__ == "__main__":
    main()
