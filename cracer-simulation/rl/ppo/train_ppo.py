#!/usr/bin/env python3
"""
PPO (Proximal Policy Optimization) Training Script for Cracer Sim.

PPO is a policy gradient method that uses clipped surrogate objective
to ensure stable policy updates.

Key differences from DQN:
- On-policy: Uses fresh experience, doesn't store replay buffer
- Actor-Critic: Separate networks for policy (actor) and value (critic)
- Direct policy optimization: Learns probability distribution over actions
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cracer_sim import CracerGymEnv  # noqa: E402

try:
    import yaml
except ImportError:
    yaml = None


@dataclass
class TrainConfig:
    """PPO Training configuration."""
    # Environment
    env_fps: int = 60
    max_episode_steps: int = 10_000

    # PPO Hyperparameters
    learning_rate: float = 3e-4
    learning_rate_end: float = 1e-5  # Final LR for annealing
    anneal_lr: bool = True  # Enable learning rate annealing
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE parameter
    clip_epsilon: float = 0.2  # PPO clipping parameter
    value_coef: float = 0.5  # Value loss coefficient
    entropy_coef: float = 0.05  # Entropy bonus coefficient (higher for exploration)
    entropy_coef_end: float = 0.005  # Final entropy coefficient
    anneal_entropy: bool = True  # Enable entropy annealing
    max_grad_norm: float = 0.5  # Gradient clipping

    # Training
    num_envs: int = 1  # Number of parallel environments (future: vectorized)
    rollout_steps: int = 2048  # Steps per rollout before update
    num_epochs: int = 10  # PPO epochs per update
    batch_size: int = 64  # Minibatch size
    total_timesteps: int = 1_000_000  # Total training timesteps

    # Network architecture
    hidden_sizes: Tuple[int, ...] = (512, 512, 256)  # Larger network
    use_layer_norm: bool = True  # Enable layer normalization
    shared_backbone: bool = False  # Share layers between actor and critic

    # Observation normalization
    normalize_obs: bool = True  # Enable observation normalization
    obs_clip: float = 10.0  # Clip normalized observations

    # Frame stacking
    frame_stack: int = 1

    # Reward shaping - improved for stage progression
    reward_speed_scale: float = 0.05  # Reduced to not dominate
    reward_fuel_bonus: float = 30.0
    reward_crash_penalty: float = 50.0  # Reduced - don't over-penalize
    reward_stage_bonus: float = 500.0  # Much higher to incentivize progression
    reward_survival_bonus: float = 0.1  # Small bonus for staying alive
    reward_distance_scale: float = 0.01  # Bonus for distance traveled
    reward_pothole_penalty: float = 5.0  # Penalty for hitting potholes
    reward_near_miss_bonus: float = 2.0  # Bonus for dodging obstacles

    # Generalization
    randomize_seed: bool = True
    seed_range: int = 100

    # Logging
    log_interval: int = 1  # Log every N updates
    save_interval: int = 10  # Save checkpoint every N updates
    checkpoint_dir: str = "rl/ppo/checkpoints"

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


class RolloutBuffer:
    """Stores rollout data for PPO updates."""
    def __init__(self, buffer_size: int, obs_size: int, device: str):
        self.buffer_size = buffer_size
        self.obs_size = obs_size
        self.device = device
        self.reset()

    def reset(self):
        self.observations = np.zeros((self.buffer_size, self.obs_size), dtype=np.float32)
        self.actions = np.zeros(self.buffer_size, dtype=np.int64)
        self.rewards = np.zeros(self.buffer_size, dtype=np.float32)
        self.dones = np.zeros(self.buffer_size, dtype=np.float32)
        self.values = np.zeros(self.buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(self.buffer_size, dtype=np.float32)
        self.advantages = np.zeros(self.buffer_size, dtype=np.float32)
        self.returns = np.zeros(self.buffer_size, dtype=np.float32)
        self.pos = 0
        self.full = False

    def add(self, obs: np.ndarray, action: int, reward: float, done: bool,
            value: float, log_prob: float):
        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob
        self.pos += 1
        if self.pos >= self.buffer_size:
            self.full = True

    def compute_returns_and_advantages(self, last_value: float, gamma: float, gae_lambda: float):
        """Compute GAE advantages and returns."""
        last_gae = 0
        for t in reversed(range(self.pos)):
            if t == self.pos - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            self.advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae

        self.returns[:self.pos] = self.advantages[:self.pos] + self.values[:self.pos]

    def get_batches(self, batch_size: int):
        """Yield minibatches for training."""
        indices = np.random.permutation(self.pos)
        for start in range(0, self.pos, batch_size):
            end = min(start + batch_size, self.pos)
            batch_indices = indices[start:end]

            yield (
                torch.tensor(self.observations[batch_indices], device=self.device),
                torch.tensor(self.actions[batch_indices], device=self.device),
                torch.tensor(self.log_probs[batch_indices], device=self.device),
                torch.tensor(self.advantages[batch_indices], device=self.device),
                torch.tensor(self.returns[batch_indices], device=self.device),
            )


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
        """Get normalizer state for saving."""
        return {
            "mean": self.mean.copy(),
            "var": self.var.copy(),
            "count": self.count,
        }

    def load_state(self, state: dict):
        """Load normalizer state."""
        self.mean = state["mean"]
        self.var = state["var"]
        self.count = state["count"]


def linear_schedule(start: float, end: float, progress: float) -> float:
    """Linear interpolation between start and end based on progress (0 to 1)."""
    return start + (end - start) * progress


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO."""
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
                layers.append(nn.Tanh())  # Tanh often works better for policy gradients
                last_size = h
            layers.append(nn.Linear(last_size, output_size))
            return nn.Sequential(*layers)

        if shared_backbone:
            # Shared feature extractor
            self.backbone = build_mlp(obs_size, hidden_sizes[-1], hidden_sizes[:-1])
            self.actor_head = nn.Linear(hidden_sizes[-1], num_actions)
            self.critic_head = nn.Linear(hidden_sizes[-1], 1)
        else:
            # Separate networks
            self.actor = build_mlp(obs_size, num_actions, hidden_sizes)
            self.critic = build_mlp(obs_size, 1, hidden_sizes)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.zeros_(module.bias)

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

    def get_action_and_value(self, x: torch.Tensor, action: Optional[torch.Tensor] = None):
        """Get action, log prob, entropy, and value."""
        action_logits, value = self(x)
        dist = Categorical(logits=action_logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Get value estimate only."""
        _, value = self(x)
        return value


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
            "update", "timesteps", "episodes", "mean_reward", "mean_score",
            "mean_stage", "mean_length", "policy_loss", "value_loss",
            "entropy", "learning_rate", "explained_var"
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


def shape_reward(
    reward: float,
    info: dict,
    prev_info: dict,
    config: TrainConfig
) -> float:
    """Apply reward shaping based on game events."""
    shaped_reward = reward

    # Survival bonus - small reward for staying alive
    shaped_reward += config.reward_survival_bonus

    # Speed bonus (reduced to not dominate)
    speed = info.get("speed", 0)
    shaped_reward += speed * config.reward_speed_scale * 0.01

    # Distance traveled bonus
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

    # Stage completion bonus - HIGH value to incentivize progression
    stage_current = info.get("stage", 1)
    stage_prev = prev_info.get("stage", 1)
    if stage_current > stage_prev:
        # Exponential bonus for higher stages
        stage_multiplier = stage_current  # Stage 2 = 2x, Stage 3 = 3x, etc.
        shaped_reward += config.reward_stage_bonus * stage_multiplier

    # Pothole penalty
    potholes_current = info.get("potholes_hit", 0)
    potholes_prev = prev_info.get("potholes_hit", 0)
    if potholes_current > potholes_prev:
        shaped_reward -= config.reward_pothole_penalty

    # Crash penalty (moderate - don't over-penalize exploration)
    if info.get("crashed", False):
        shaped_reward -= config.reward_crash_penalty

    return shaped_reward


def train(config: TrainConfig) -> None:
    """Main PPO training loop."""
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

    # Create network
    actor_critic = ActorCritic(
        obs_size=obs_size,
        num_actions=num_actions,
        hidden_sizes=config.hidden_sizes,
        use_layer_norm=config.use_layer_norm,
        shared_backbone=config.shared_backbone,
    ).to(device)

    optimizer = optim.Adam(actor_critic.parameters(), lr=config.learning_rate, eps=1e-5)

    # Frame stacker
    frame_stacker = FrameStack(config.frame_stack, base_obs_size) if config.frame_stack > 1 else None

    # Observation normalizer
    obs_normalizer = ObservationNormalizer(obs_size, config.obs_clip) if config.normalize_obs else None

    # Rollout buffer
    buffer = RolloutBuffer(config.rollout_steps, obs_size, device)

    # Logging
    checkpoint_dir = os.path.join(ROOT, config.checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = TrainingLogger(os.path.join(checkpoint_dir, "training_log.csv"))

    # Training state
    global_step = 0
    num_updates = 0
    episode_count = 0
    episode_rewards: List[float] = []
    episode_scores: List[float] = []
    episode_stages: List[int] = []
    episode_lengths: List[int] = []

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

    print(f"\nStarting PPO training for {config.total_timesteps} timesteps...")
    print(f"Rollout steps: {config.rollout_steps}, Epochs: {config.num_epochs}, Batch size: {config.batch_size}")
    print(f"Obs normalization: {config.normalize_obs}, LR annealing: {config.anneal_lr}, Entropy annealing: {config.anneal_entropy}")

    # Current scheduled values
    current_lr = config.learning_rate
    current_entropy_coef = config.entropy_coef

    while global_step < config.total_timesteps:
        # Update learning rate and entropy coefficient
        progress = global_step / config.total_timesteps
        if config.anneal_lr:
            current_lr = linear_schedule(config.learning_rate, config.learning_rate_end, progress)
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr
        if config.anneal_entropy:
            current_entropy_coef = linear_schedule(config.entropy_coef, config.entropy_coef_end, progress)

        # Collect rollout
        buffer.reset()

        for _ in range(config.rollout_steps):
            global_step += 1

            # Normalize observation if enabled
            obs_normalized = obs_normalizer.normalize(obs) if obs_normalizer else obs

            with torch.no_grad():
                obs_tensor = torch.tensor(obs_normalized, dtype=torch.float32, device=device).unsqueeze(0)
                action, log_prob, _, value = actor_critic.get_action_and_value(obs_tensor)
                action = action.item()
                log_prob = log_prob.item()
                value = value.item()

            next_obs, reward, terminated, truncated, info = env.step(action)
            next_obs = np.asarray(next_obs, dtype=np.float32)

            # Shape reward
            shaped_reward = shape_reward(reward, info, prev_info, config)
            prev_info = info.copy()

            done = terminated or truncated
            current_episode_reward += shaped_reward
            current_episode_length += 1
            max_stage = max(max_stage, info.get("stage", 1))

            # Store normalized observation in buffer
            buffer.add(obs_normalized, action, shaped_reward, done, value, log_prob)

            # Update observation
            if frame_stacker:
                next_obs = frame_stacker.push(next_obs) if not done else next_obs
            obs = next_obs

            # Update observation normalizer statistics
            if obs_normalizer:
                obs_normalizer.update(obs)

            if done:
                # Log episode
                episode_rewards.append(current_episode_reward)
                episode_scores.append(info.get("score", 0))
                episode_stages.append(max_stage)
                episode_lengths.append(current_episode_length)
                episode_count += 1

                # Reset for new episode
                new_seed = random.randint(0, config.seed_range) if config.randomize_seed else 42
                obs, info = env.reset(seed=new_seed)
                obs = np.asarray(obs, dtype=np.float32)
                if frame_stacker:
                    obs = frame_stacker.reset(obs)

                current_episode_reward = 0.0
                current_episode_length = 0
                max_stage = 1
                prev_info = info.copy()

            if global_step >= config.total_timesteps:
                break

        # Compute returns and advantages
        with torch.no_grad():
            obs_normalized = obs_normalizer.normalize(obs) if obs_normalizer else obs
            obs_tensor = torch.tensor(obs_normalized, dtype=torch.float32, device=device).unsqueeze(0)
            last_value = actor_critic.get_value(obs_tensor).item()

        buffer.compute_returns_and_advantages(last_value, config.gamma, config.gae_lambda)

        # Normalize advantages
        advantages = buffer.advantages[:buffer.pos]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        buffer.advantages[:buffer.pos] = advantages

        # PPO update
        policy_losses = []
        value_losses = []
        entropies = []
        clip_fractions = []

        for _ in range(config.num_epochs):
            for batch in buffer.get_batches(config.batch_size):
                b_obs, b_actions, b_log_probs, b_advantages, b_returns = batch

                _, new_log_probs, entropy, new_values = actor_critic.get_action_and_value(b_obs, b_actions)

                # Policy loss with clipping
                log_ratio = new_log_probs - b_log_probs
                ratio = torch.exp(log_ratio)

                # Clip fraction for logging
                with torch.no_grad():
                    clip_fraction = ((ratio - 1.0).abs() > config.clip_epsilon).float().mean().item()
                    clip_fractions.append(clip_fraction)

                policy_loss1 = -b_advantages * ratio
                policy_loss2 = -b_advantages * torch.clamp(ratio, 1 - config.clip_epsilon, 1 + config.clip_epsilon)
                policy_loss = torch.max(policy_loss1, policy_loss2).mean()

                # Value loss
                value_loss = ((new_values - b_returns) ** 2).mean()

                # Entropy bonus
                entropy_loss = entropy.mean()

                # Total loss (using scheduled entropy coefficient)
                loss = policy_loss + config.value_coef * value_loss - current_entropy_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(actor_critic.parameters(), config.max_grad_norm)
                optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy_loss.item())

        num_updates += 1

        # Compute explained variance
        with torch.no_grad():
            values_np = buffer.values[:buffer.pos]
            returns_np = buffer.returns[:buffer.pos]
            var_returns = np.var(returns_np)
            explained_var = 1 - np.var(returns_np - values_np) / (var_returns + 1e-8) if var_returns > 0 else 0

        # Logging
        if num_updates % config.log_interval == 0 and episode_rewards:
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

            print(f"Update {num_updates} | Steps: {global_step}/{config.total_timesteps} | "
                  f"Episodes: {episode_count} | FPS: {fps:.0f}")
            print(f"  Mean reward: {mean_reward:.1f} | Score: {mean_score:.0f} | "
                  f"Stage: {mean_stage:.1f} | Length: {mean_length:.0f}")
            print(f"  Policy loss: {np.mean(policy_losses):.4f} | Value loss: {np.mean(value_losses):.4f} | "
                  f"Entropy: {np.mean(entropies):.4f} | Clip frac: {np.mean(clip_fractions):.3f}")
            print(f"  LR: {current_lr:.2e} | Entropy coef: {current_entropy_coef:.4f}")

            logger.log({
                "update": num_updates,
                "timesteps": global_step,
                "episodes": episode_count,
                "mean_reward": mean_reward,
                "mean_score": mean_score,
                "mean_stage": mean_stage,
                "mean_length": mean_length,
                "policy_loss": np.mean(policy_losses),
                "value_loss": np.mean(value_losses),
                "entropy": np.mean(entropies),
                "learning_rate": current_lr,
                "explained_var": explained_var,
            })

            # Save best model
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                save_checkpoint(actor_critic, optimizer, config, global_step, num_updates,
                               checkpoint_dir, "best.pt", obs_normalizer)

        # Periodic checkpoint
        if num_updates % config.save_interval == 0:
            save_checkpoint(actor_critic, optimizer, config, global_step, num_updates,
                           checkpoint_dir, f"checkpoint_{num_updates}.pt", obs_normalizer)

    # Final save
    save_checkpoint(actor_critic, optimizer, config, global_step, num_updates,
                   checkpoint_dir, "final.pt", obs_normalizer)

    env.close()
    print(f"\nTraining complete! Total timesteps: {global_step}, Updates: {num_updates}, Episodes: {episode_count}")
    print(f"Best mean reward: {best_mean_reward:.1f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, config: TrainConfig,
                   global_step: int, num_updates: int, checkpoint_dir: str, filename: str,
                   obs_normalizer: Optional[ObservationNormalizer] = None):
    """Save training checkpoint."""
    path = os.path.join(checkpoint_dir, filename)
    checkpoint_data = {
        "actor_critic": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": global_step,
        "num_updates": num_updates,
        "model_config": {
            "hidden_sizes": config.hidden_sizes,
            "use_layer_norm": config.use_layer_norm,
            "shared_backbone": config.shared_backbone,
            "frame_stack": config.frame_stack,
            "normalize_obs": config.normalize_obs,
        },
    }
    if obs_normalizer is not None:
        checkpoint_data["obs_normalizer"] = obs_normalizer.get_state()
    torch.save(checkpoint_data, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO agent for Cracer Sim")
    parser.add_argument("--config", type=str, default="", help="Path to config YAML file")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
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
        default_config = os.path.join(ROOT, "rl", "ppo", "config.yaml")
        if os.path.exists(default_config):
            config = TrainConfig.from_yaml(default_config)
        else:
            config = TrainConfig()

    # Override with command line args
    if args.total_timesteps is not None:
        config.total_timesteps = args.total_timesteps
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.rollout_steps is not None:
        config.rollout_steps = args.rollout_steps
    if args.num_epochs is not None:
        config.num_epochs = args.num_epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.device is not None:
        config.device = args.device

    train(config)


if __name__ == "__main__":
    main()
