#!/usr/bin/env python3
"""
SAC (Soft Actor-Critic) Training Script for Cracer Sim.

SAC is an off-policy actor-critic algorithm that maximizes both expected reward
and entropy, encouraging exploration while learning optimal policies.

This implementation uses SAC-Discrete for discrete action spaces.

Key features:
- Off-policy: Uses replay buffer for sample efficiency
- Entropy regularization: Automatic temperature tuning
- Twin Q-networks: Reduces overestimation bias
- Soft updates: Stable target network updates
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from game import CracerGymEnv  # noqa: E402

try:
    import yaml
except ImportError:
    yaml = None


@dataclass
class TrainConfig:
    """SAC Training configuration."""
    # Environment
    env_fps: int = 60
    max_episode_steps: int = 10_000

    # SAC Hyperparameters
    learning_rate: float = 3e-4
    gamma: float = 0.99  # Discount factor
    tau: float = 0.005  # Soft update coefficient
    alpha: float = 0.2  # Initial entropy coefficient
    auto_alpha: bool = True  # Automatically tune alpha
    target_entropy_ratio: float = 0.98  # Target entropy as ratio of max entropy

    # Training
    buffer_size: int = 100_000  # Replay buffer size
    batch_size: int = 256
    learning_starts: int = 1000  # Steps before training starts
    train_freq: int = 1  # Train every N steps
    gradient_steps: int = 1  # Gradient steps per train call
    total_timesteps: int = 1_000_000

    # Network architecture
    hidden_sizes: Tuple[int, ...] = (512, 512, 256)
    use_layer_norm: bool = True

    # Observation normalization
    normalize_obs: bool = True
    obs_clip: float = 10.0

    # Frame stacking
    frame_stack: int = 1

    # Reward shaping
    reward_speed_scale: float = 0.05
    reward_fuel_bonus: float = 30.0
    reward_crash_penalty: float = 50.0
    reward_stage_bonus: float = 500.0
    reward_survival_bonus: float = 0.1
    reward_distance_scale: float = 0.01
    reward_pothole_penalty: float = 5.0

    # Generalization
    randomize_seed: bool = True
    seed_range: int = 100

    # Logging
    log_interval: int = 1000  # Log every N steps
    save_interval: int = 10000  # Save checkpoint every N steps
    checkpoint_dir: str = "rl/sac/checkpoints"

    # Device
    device: str = "auto"

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        if yaml is None:
            raise ImportError("PyYAML required. pip install pyyaml")
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        if "hidden_sizes" in data:
            data["hidden_sizes"] = tuple(data["hidden_sizes"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class ReplayBuffer:
    """Experience replay buffer for SAC."""
    def __init__(self, buffer_size: int, obs_size: int, device: str):
        self.buffer_size = buffer_size
        self.obs_size = obs_size
        self.device = device
        self.pos = 0
        self.size = 0

        self.observations = np.zeros((buffer_size, obs_size), dtype=np.float32)
        self.next_observations = np.zeros((buffer_size, obs_size), dtype=np.float32)
        self.actions = np.zeros(buffer_size, dtype=np.int64)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)

    def add(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool):
        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_observations[self.pos] = next_obs
        self.dones[self.pos] = done

        self.pos = (self.pos + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size: int):
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.tensor(self.observations[indices], device=self.device),
            torch.tensor(self.actions[indices], device=self.device),
            torch.tensor(self.rewards[indices], device=self.device),
            torch.tensor(self.next_observations[indices], device=self.device),
            torch.tensor(self.dones[indices], device=self.device),
        )

    def __len__(self):
        return self.size


class ObservationNormalizer:
    """Running mean/std normalization for observations."""
    def __init__(self, obs_size: int, clip: float = 10.0, epsilon: float = 1e-8):
        self.obs_size = obs_size
        self.clip = clip
        self.epsilon = epsilon
        self.mean = np.zeros(obs_size, dtype=np.float64)
        self.var = np.ones(obs_size, dtype=np.float64)
        self.count = 0

    def update(self, obs: np.ndarray):
        """Update running statistics with new observation."""
        batch_mean = obs.astype(np.float64)
        batch_var = np.zeros_like(batch_mean)
        batch_count = 1

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        self.var = M2 / total_count
        self.count = total_count

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation using running statistics."""
        normalized = (obs - self.mean.astype(np.float32)) / (np.sqrt(self.var.astype(np.float32)) + self.epsilon)
        return np.clip(normalized, -self.clip, self.clip)

    def get_state(self) -> dict:
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": self.count}

    def load_state(self, state: dict):
        self.mean = state["mean"]
        self.var = state["var"]
        self.count = state["count"]


class SoftQNetwork(nn.Module):
    """Soft Q-Network for SAC."""
    def __init__(self, obs_size: int, num_actions: int, hidden_sizes: Tuple[int, ...],
                 use_layer_norm: bool = True):
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


class PolicyNetwork(nn.Module):
    """Policy network for SAC-Discrete."""
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

    def get_action_probs(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action probabilities and log probabilities."""
        logits = self(x)
        # Use softmax for probabilities
        probs = F.softmax(logits, dim=-1)
        # Add small epsilon for numerical stability
        log_probs = F.log_softmax(logits, dim=-1)
        return probs, log_probs

    def sample_action(self, x: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Sample action from policy."""
        probs, log_probs = self.get_action_probs(x)
        action = torch.multinomial(probs, 1).squeeze(-1)
        return action.item(), probs, log_probs


class FrameStack:
    """Stacks multiple frames for temporal context."""
    def __init__(self, num_frames: int, obs_size: int):
        self.num_frames = num_frames
        self.obs_size = obs_size
        self.frames: List[np.ndarray] = []

    def reset(self, obs: np.ndarray) -> np.ndarray:
        self.frames = [obs.copy() for _ in range(self.num_frames)]
        return self.get()

    def push(self, obs: np.ndarray) -> np.ndarray:
        self.frames.pop(0)
        self.frames.append(obs.copy())
        return self.get()

    def get(self) -> np.ndarray:
        return np.concatenate(self.frames, axis=0)


class TrainingLogger:
    """Logs training metrics to CSV and console."""
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.fieldnames = [
            "timestep", "episodes", "mean_reward", "mean_score", "mean_stage",
            "mean_length", "q1_loss", "q2_loss", "policy_loss", "alpha", "entropy"
        ]

        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def log(self, metrics: Dict):
        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow({k: metrics.get(k, 0) for k in self.fieldnames})


def pick_device(choice: str) -> str:
    if choice != "auto":
        return choice
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def shape_reward(reward: float, info: dict, prev_info: dict, config: TrainConfig) -> float:
    """Apply reward shaping based on game events."""
    shaped_reward = reward

    # Survival bonus
    shaped_reward += config.reward_survival_bonus

    # Speed bonus
    speed = info.get("speed", 0)
    shaped_reward += speed * config.reward_speed_scale * 0.01

    # Distance bonus
    distance_current = info.get("distance", 0)
    distance_prev = prev_info.get("distance", 0)
    distance_delta = distance_current - distance_prev
    if distance_delta > 0:
        shaped_reward += distance_delta * config.reward_distance_scale

    # Fuel pickup bonus
    fuel_current = info.get("fuel_collected", 0)
    fuel_prev = prev_info.get("fuel_collected", 0)
    if fuel_current > fuel_prev:
        shaped_reward += config.reward_fuel_bonus

    # Stage completion bonus
    stage_current = info.get("stage", 1)
    stage_prev = prev_info.get("stage", 1)
    if stage_current > stage_prev:
        stage_multiplier = stage_current
        shaped_reward += config.reward_stage_bonus * stage_multiplier

    # Pothole penalty
    potholes_current = info.get("potholes_hit", 0)
    potholes_prev = prev_info.get("potholes_hit", 0)
    if potholes_current > potholes_prev:
        shaped_reward -= config.reward_pothole_penalty

    # Crash penalty
    if info.get("crashed", False):
        shaped_reward -= config.reward_crash_penalty

    return shaped_reward


def train(config: TrainConfig) -> None:
    """Main SAC training loop."""
    device = pick_device(config.device)
    print(f"Device: {device}")
    print(f"Config: {config}")

    # Create environment
    initial_seed = random.randint(0, config.seed_range) if config.randomize_seed else 42
    env = CracerGymEnv(
        render_mode=None,
        obs_mode="state",
        action_mode="discrete",
        fps=config.env_fps,
        seed=initial_seed,
    )

    base_obs_size = env.observation_space.shape[0]
    num_actions = env.action_space.n
    obs_size = base_obs_size * config.frame_stack

    print(f"Observation space: {base_obs_size} (stacked: {obs_size})")
    print(f"Action space: {num_actions}")

    # Create networks
    policy = PolicyNetwork(obs_size, num_actions, config.hidden_sizes, config.use_layer_norm).to(device)
    q1 = SoftQNetwork(obs_size, num_actions, config.hidden_sizes, config.use_layer_norm).to(device)
    q2 = SoftQNetwork(obs_size, num_actions, config.hidden_sizes, config.use_layer_norm).to(device)
    q1_target = SoftQNetwork(obs_size, num_actions, config.hidden_sizes, config.use_layer_norm).to(device)
    q2_target = SoftQNetwork(obs_size, num_actions, config.hidden_sizes, config.use_layer_norm).to(device)

    # Copy weights to target networks
    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())

    # Optimizers
    policy_optimizer = optim.Adam(policy.parameters(), lr=config.learning_rate)
    q1_optimizer = optim.Adam(q1.parameters(), lr=config.learning_rate)
    q2_optimizer = optim.Adam(q2.parameters(), lr=config.learning_rate)

    # Automatic entropy tuning
    if config.auto_alpha:
        target_entropy = -config.target_entropy_ratio * np.log(1.0 / num_actions)
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha_optimizer = optim.Adam([log_alpha], lr=config.learning_rate)
        alpha = log_alpha.exp().item()
    else:
        alpha = config.alpha
        log_alpha = None
        alpha_optimizer = None

    # Frame stacker
    frame_stacker = FrameStack(config.frame_stack, base_obs_size) if config.frame_stack > 1 else None

    # Observation normalizer
    obs_normalizer = ObservationNormalizer(obs_size, config.obs_clip) if config.normalize_obs else None

    # Replay buffer
    buffer = ReplayBuffer(config.buffer_size, obs_size, device)

    # Logging
    checkpoint_dir = os.path.join(ROOT, config.checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = TrainingLogger(os.path.join(checkpoint_dir, "training_log.csv"))

    # Training state
    global_step = 0
    episode_count = 0
    episode_rewards: List[float] = []
    episode_scores: List[float] = []
    episode_stages: List[int] = []
    episode_lengths: List[int] = []

    # Loss tracking
    q1_losses: List[float] = []
    q2_losses: List[float] = []
    policy_losses: List[float] = []
    entropies: List[float] = []

    # Current episode state
    obs, info = env.reset(seed=initial_seed)
    obs = np.asarray(obs, dtype=np.float32)
    if frame_stacker:
        obs = frame_stacker.reset(obs)

    current_episode_reward = 0.0
    current_episode_length = 0
    max_stage = 1
    prev_info = info.copy()

    best_mean_reward = float("-inf")
    start_time = time.time()

    print(f"\nStarting SAC training for {config.total_timesteps} timesteps...")
    print(f"Buffer size: {config.buffer_size}, Batch size: {config.batch_size}")
    print(f"Auto alpha: {config.auto_alpha}, Initial alpha: {alpha:.4f}")

    while global_step < config.total_timesteps:
        global_step += 1

        # Update normalizer
        if obs_normalizer:
            obs_normalizer.update(obs)

        # Normalize observation
        obs_normalized = obs_normalizer.normalize(obs) if obs_normalizer else obs

        # Select action
        if global_step < config.learning_starts:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs_normalized, dtype=torch.float32, device=device).unsqueeze(0)
                action, _, _ = policy.sample_action(obs_tensor)

        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_obs = np.asarray(next_obs, dtype=np.float32)

        # Shape reward
        shaped_reward = shape_reward(reward, info, prev_info, config)
        prev_info = info.copy()

        done = terminated or truncated
        current_episode_reward += shaped_reward
        current_episode_length += 1
        max_stage = max(max_stage, info.get("stage", 1))

        # Update observation
        if frame_stacker:
            next_obs_stacked = frame_stacker.push(next_obs) if not done else next_obs
        else:
            next_obs_stacked = next_obs

        # Normalize next observation for storage
        next_obs_normalized = obs_normalizer.normalize(next_obs_stacked) if obs_normalizer else next_obs_stacked

        # Store transition
        buffer.add(obs_normalized, action, shaped_reward, next_obs_normalized, done)

        obs = next_obs_stacked

        if done:
            # Log episode
            episode_rewards.append(current_episode_reward)
            episode_scores.append(info.get("score", 0))
            episode_stages.append(max_stage)
            episode_lengths.append(current_episode_length)
            episode_count += 1

            # Reset
            new_seed = random.randint(0, config.seed_range) if config.randomize_seed else 42
            obs, info = env.reset(seed=new_seed)
            obs = np.asarray(obs, dtype=np.float32)
            if frame_stacker:
                obs = frame_stacker.reset(obs)

            current_episode_reward = 0.0
            current_episode_length = 0
            max_stage = 1
            prev_info = info.copy()

        # Training
        if global_step >= config.learning_starts and global_step % config.train_freq == 0:
            for _ in range(config.gradient_steps):
                # Sample batch
                b_obs, b_actions, b_rewards, b_next_obs, b_dones = buffer.sample(config.batch_size)

                with torch.no_grad():
                    # Get next action probabilities
                    next_probs, next_log_probs = policy.get_action_probs(b_next_obs)

                    # Compute target Q values
                    q1_next = q1_target(b_next_obs)
                    q2_next = q2_target(b_next_obs)
                    min_q_next = torch.min(q1_next, q2_next)

                    # Soft Q target (expectation over actions)
                    soft_q_next = (next_probs * (min_q_next - alpha * next_log_probs)).sum(dim=-1)
                    target_q = b_rewards + config.gamma * (1 - b_dones) * soft_q_next

                # Update Q networks
                q1_values = q1(b_obs).gather(1, b_actions.unsqueeze(-1)).squeeze(-1)
                q2_values = q2(b_obs).gather(1, b_actions.unsqueeze(-1)).squeeze(-1)

                q1_loss = F.mse_loss(q1_values, target_q)
                q2_loss = F.mse_loss(q2_values, target_q)

                q1_optimizer.zero_grad()
                q1_loss.backward()
                q1_optimizer.step()

                q2_optimizer.zero_grad()
                q2_loss.backward()
                q2_optimizer.step()

                # Update policy
                probs, log_probs = policy.get_action_probs(b_obs)

                with torch.no_grad():
                    q1_values_all = q1(b_obs)
                    q2_values_all = q2(b_obs)
                    min_q_values = torch.min(q1_values_all, q2_values_all)

                # Policy loss: maximize Q - alpha * log_prob
                policy_loss = (probs * (alpha * log_probs - min_q_values)).sum(dim=-1).mean()

                policy_optimizer.zero_grad()
                policy_loss.backward()
                policy_optimizer.step()

                # Update alpha
                if config.auto_alpha:
                    # Entropy of current policy
                    entropy = -(probs * log_probs).sum(dim=-1).mean()

                    alpha_loss = (log_alpha * (entropy - target_entropy).detach()).mean()

                    alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    alpha_optimizer.step()

                    alpha = log_alpha.exp().item()

                # Soft update target networks
                for param, target_param in zip(q1.parameters(), q1_target.parameters()):
                    target_param.data.copy_(config.tau * param.data + (1 - config.tau) * target_param.data)
                for param, target_param in zip(q2.parameters(), q2_target.parameters()):
                    target_param.data.copy_(config.tau * param.data + (1 - config.tau) * target_param.data)

                # Track losses
                q1_losses.append(q1_loss.item())
                q2_losses.append(q2_loss.item())
                policy_losses.append(policy_loss.item())
                with torch.no_grad():
                    ent = -(probs * log_probs).sum(dim=-1).mean().item()
                    entropies.append(ent)

        # Logging
        if global_step % config.log_interval == 0 and episode_rewards:
            recent_rewards = episode_rewards[-100:] if len(episode_rewards) >= 100 else episode_rewards
            recent_scores = episode_scores[-100:] if len(episode_scores) >= 100 else episode_scores
            recent_stages = episode_stages[-100:] if len(episode_stages) >= 100 else episode_stages
            recent_lengths = episode_lengths[-100:] if len(episode_lengths) >= 100 else episode_lengths

            mean_reward = np.mean(recent_rewards)
            mean_score = np.mean(recent_scores)
            mean_stage = np.mean(recent_stages)
            mean_length = np.mean(recent_lengths)

            elapsed = time.time() - start_time
            fps = global_step / elapsed

            print(f"Step {global_step}/{config.total_timesteps} | Episodes: {episode_count} | FPS: {fps:.0f}")
            print(f"  Mean reward: {mean_reward:.1f} | Score: {mean_score:.0f} | "
                  f"Stage: {mean_stage:.1f} | Length: {mean_length:.0f}")

            if q1_losses:
                print(f"  Q1 loss: {np.mean(q1_losses[-100:]):.4f} | Q2 loss: {np.mean(q2_losses[-100:]):.4f} | "
                      f"Policy loss: {np.mean(policy_losses[-100:]):.4f}")
                print(f"  Alpha: {alpha:.4f} | Entropy: {np.mean(entropies[-100:]):.4f}")

            logger.log({
                "timestep": global_step,
                "episodes": episode_count,
                "mean_reward": mean_reward,
                "mean_score": mean_score,
                "mean_stage": mean_stage,
                "mean_length": mean_length,
                "q1_loss": np.mean(q1_losses[-100:]) if q1_losses else 0,
                "q2_loss": np.mean(q2_losses[-100:]) if q2_losses else 0,
                "policy_loss": np.mean(policy_losses[-100:]) if policy_losses else 0,
                "alpha": alpha,
                "entropy": np.mean(entropies[-100:]) if entropies else 0,
            })

            # Save best model
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                save_checkpoint(policy, q1, q2, config, global_step, episode_count,
                               checkpoint_dir, "best.pt", obs_normalizer, alpha)

        # Periodic checkpoint
        if global_step % config.save_interval == 0:
            save_checkpoint(policy, q1, q2, config, global_step, episode_count,
                           checkpoint_dir, f"checkpoint_{global_step}.pt", obs_normalizer, alpha)

    # Final save
    save_checkpoint(policy, q1, q2, config, global_step, episode_count,
                   checkpoint_dir, "final.pt", obs_normalizer, alpha)

    env.close()
    print(f"\nTraining complete! Total timesteps: {global_step}, Episodes: {episode_count}")
    print(f"Best mean reward: {best_mean_reward:.1f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")


def save_checkpoint(policy: nn.Module, q1: nn.Module, q2: nn.Module, config: TrainConfig,
                   global_step: int, episode_count: int, checkpoint_dir: str, filename: str,
                   obs_normalizer: Optional[ObservationNormalizer] = None, alpha: float = 0.2):
    """Save training checkpoint."""
    path = os.path.join(checkpoint_dir, filename)
    checkpoint_data = {
        "policy": policy.state_dict(),
        "q1": q1.state_dict(),
        "q2": q2.state_dict(),
        "global_step": global_step,
        "episode_count": episode_count,
        "alpha": alpha,
        "model_config": {
            "hidden_sizes": config.hidden_sizes,
            "use_layer_norm": config.use_layer_norm,
            "frame_stack": config.frame_stack,
            "normalize_obs": config.normalize_obs,
        },
    }
    if obs_normalizer is not None:
        checkpoint_data["obs_normalizer"] = obs_normalizer.get_state()
    torch.save(checkpoint_data, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAC agent for Cracer Sim")
    parser.add_argument("--config", type=str, default="", help="Path to config YAML file")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--buffer-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load config
    if args.config:
        config_path = args.config if os.path.isabs(args.config) else os.path.join(ROOT, args.config)
        config = TrainConfig.from_yaml(config_path)
    else:
        default_config = os.path.join(ROOT, "rl", "sac", "config.yaml")
        if os.path.exists(default_config):
            config = TrainConfig.from_yaml(default_config)
        else:
            config = TrainConfig()

    # Override with command line args
    if args.total_timesteps is not None:
        config.total_timesteps = args.total_timesteps
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.buffer_size is not None:
        config.buffer_size = args.buffer_size
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.device is not None:
        config.device = args.device

    train(config)


if __name__ == "__main__":
    main()
