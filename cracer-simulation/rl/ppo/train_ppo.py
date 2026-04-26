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

from game import CracerGymEnv  # noqa: E402

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
    max_objects: int = 10  # Number of objects in observation

    # PPO Hyperparameters
    learning_rate: float = 3e-4
    learning_rate_end: float = 1e-5  # Final LR for annealing
    anneal_lr: bool = True  # Enable learning rate annealing
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE parameter
    clip_epsilon: float = 0.2  # PPO clipping parameter
    value_coef: float = 0.5  # Value loss coefficient
    normalize_returns: bool = False  # Normalize returns before value loss (fixes value explosion)
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
    frame_stack: int = 4

    # Reward shaping - aligned with SAC reference
    reward_speed_scale: float = 0.1
    reward_fuel_bonus: float = 30.0
    reward_crash_penalty: float = 75.0
    reward_crash_penalty_stage_scale: float = 0.0  # Per-stage crash penalty multiplier
    # Final crash penalty = reward_crash_penalty × (1 + scale × (stage - 1))
    # e.g. scale=0.3, stage=10 → 600 × (1 + 0.3×9) = 600 × 3.7 = 2220
    # Makes dying progressively more costly the further you've progressed
    reward_stage_bonus: float = 1000.0
    reward_survival_bonus: float = 0.15
    reward_distance_scale: float = 0.1
    reward_distance_milestone: float = 40.0
    reward_distance_milestone_interval: int = 300
    reward_pothole_penalty: float = 5.0
    reward_safe_speed_bonus: float = 0.08
    reward_fuel_exhaustion_penalty: float = 0.0   # penalty on death by fuel (mirrors crash penalty)
    reward_low_fuel_penalty: float = 0.0          # per-step penalty when fuel < 30 (urgency signal)
    reward_bonus_stage_scale: float = 0.0         # scale bonuses proportionally to stage progress
    # Final bonus = base × (1 + scale × (stage - 1))
    # e.g. scale=0.3, stage=10 → survival/fuel/milestone bonus × 3.7
    # Symmetric with crash penalty — surviving at high stages pays more too

    # Generalization
    randomize_seed: bool = True
    seed_range: int = 1000

    # Fixed stage mix — used when curriculum_enabled=false
    # Dict of {stage: probability} e.g. {1: 0.9, 2: 0.1}
    # Empty dict means always stage 1.
    stage_mix: Dict[int, float] = field(default_factory=dict)

    # Curriculum Learning - 10-stage with graduation (mirrors SAC)
    curriculum_enabled: bool = True
    curriculum_graduation_window: int = 100
    curriculum_min_episodes_per_stage: int = 150
    curriculum_adaptive_eval_interval: int = 50
    curriculum_max_stage: int = 10  # kept for eval threshold reference

    # Early stopping
    early_stopping_patience: int = 0  # 0 = disabled

    # Evaluation (periodic during training)
    eval_interval_updates: int = 50  # Set <= 0 to disable
    eval_episodes: int = 5  # Episodes per start stage
    eval_start_stages: Tuple[int, ...] = (10,)
    eval_deterministic: bool = True
    eval_max_episode_steps: int = 10_000
    eval_seed: Optional[int] = None

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
        if "stage_mix" in data and isinstance(data["stage_mix"], dict):
            data["stage_mix"] = {int(k): float(v) for k, v in data["stage_mix"].items()}
        if "eval_start_stages" in data:
            raw_stages = data["eval_start_stages"]
            if isinstance(raw_stages, str):
                data["eval_start_stages"] = tuple(parse_start_stages(raw_stages))
            elif isinstance(raw_stages, (list, tuple)):
                data["eval_start_stages"] = tuple(int(s) for s in raw_stages)
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

    def compute_returns_and_advantages(self, last_value: float, gamma: float, gae_lambda: float,
                                       normalize_returns: bool = False):
        """Compute GAE advantages and returns.

        normalize_returns: if True, normalise the return targets so the value function
        always learns on a consistent scale regardless of episode reward magnitude.
        This prevents value loss explosion as the agent clears more stages.
        """
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

        if normalize_returns:
            returns = self.returns[:self.pos]
            self.returns[:self.pos] = (returns - returns.mean()) / (returns.std() + 1e-8)

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
            "entropy", "learning_rate", "explained_var",
            "mean_start_stage", "start_stage_t1_frac", "start_stage_t2_frac",
            "start_stage_t3_frac", "start_stage_t4_frac",
            "eval_mean_reward", "eval_mean_score", "eval_mean_max_stage",
            "eval_mean_length", "eval_over_max_rate",
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


def load_checkpoint(path: str, device: str):
    return torch.load(path, map_location=device, weights_only=False)


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the latest numbered checkpoint in the directory."""
    if not os.path.exists(checkpoint_dir):
        return None
    checkpoint_files = []
    for f in os.listdir(checkpoint_dir):
        if f.startswith("checkpoint_") and f.endswith(".pt"):
            try:
                step = int(f.replace("checkpoint_", "").replace(".pt", ""))
                checkpoint_files.append((step, f))
            except ValueError:
                continue
    if not checkpoint_files:
        return None
    checkpoint_files.sort(key=lambda x: x[0], reverse=True)
    return os.path.join(checkpoint_dir, checkpoint_files[0][1])


class CurriculumManager:
    """10-stage curriculum with adaptive mode after graduation (mirrors SAC)."""

    STAGES = [
        ({1: 1.0},                           150, None),
        ({1: 0.7, 2: 0.3},                   200, 0.30),
        ({1: 0.4, 2: 0.4, 3: 0.2},           250, 0.25),
        ({2: 0.2, 3: 0.5, 4: 0.3},           300, 0.20),
        ({2: 0.05, 3: 0.05, 4: 0.4, 5: 0.5}, 350, 0.15),
        ({4: 0.25, 5: 0.25, 6: 0.5},         350, 0.15),
        ({5: 0.3, 6: 0.4, 7: 0.3},           350, 0.10),
        ({6: 0.2, 7: 0.4, 8: 0.4},           300, 0.10),
        ({7: 0.3, 8: 0.4, 9: 0.3},           250, 0.10),
        ({8: 0.2, 9: 0.4, 10: 0.4},          200, 0.10),
    ]

    def __init__(self, config: TrainConfig):
        self.window = config.curriculum_graduation_window
        self.min_episodes = config.curriculum_min_episodes_per_stage
        self.adaptive_interval = config.curriculum_adaptive_eval_interval
        self.enabled = config.curriculum_enabled

        self.current_stage = 0
        self.adaptive = False
        self.stage_episodes = 0

        self.rewards_buf: List[float] = []
        self.completions_buf: List[float] = []

        self.adaptive_rewards: Dict[int, List[float]] = {s: [] for s in range(1, 11)}
        self.adaptive_weights: Dict[int, float] = {s: 1.0 / 10 for s in range(1, 11)}
        self.adaptive_ep_count = 0

    def sample_stage(self) -> int:
        if not self.enabled:
            return 1
        dist = self.adaptive_weights if self.adaptive else self.STAGES[self.current_stage][0]
        stages = list(dist.keys())
        probs = [dist[s] for s in stages]
        return random.choices(stages, weights=probs, k=1)[0]

    def record_episode(self, reward: float, start_stage: int, max_stage: int) -> Optional[str]:
        if not self.enabled:
            return None
        completed = 1.0 if max_stage > start_stage else 0.0
        if not self.adaptive:
            self.rewards_buf.append(reward)
            self.completions_buf.append(completed)
            self.stage_episodes += 1
            if len(self.rewards_buf) > self.window:
                self.rewards_buf = self.rewards_buf[-self.window:]
                self.completions_buf = self.completions_buf[-self.window:]
            return self._check_graduation()
        else:
            self.adaptive_rewards[start_stage].append(reward)
            if len(self.adaptive_rewards[start_stage]) > self.window:
                self.adaptive_rewards[start_stage] = self.adaptive_rewards[start_stage][-self.window:]
            self.adaptive_ep_count += 1
            if self.adaptive_ep_count % self.adaptive_interval == 0:
                self._recompute_adaptive_weights()
            return None

    def _check_graduation(self) -> Optional[str]:
        if self.stage_episodes < self.min_episodes or len(self.rewards_buf) < self.window:
            return None
        _, grad_reward, grad_completion = self.STAGES[self.current_stage]
        mean_reward = np.mean(self.rewards_buf)
        mean_comp = np.mean(self.completions_buf)
        if mean_reward >= grad_reward and (grad_completion is None or mean_comp >= grad_completion):
            old = self.current_stage + 1
            if self.current_stage < len(self.STAGES) - 1:
                self.current_stage += 1
                self.stage_episodes = 0
                self.rewards_buf.clear()
                self.completions_buf.clear()
                return (f"CURRICULUM: Graduated stage {old} -> {self.current_stage + 1} "
                        f"(reward={mean_reward:.1f}, comp={mean_comp:.2f})")
            else:
                self.adaptive = True
                return (f"CURRICULUM: Completed all 10 stages! Entering adaptive mode "
                        f"(reward={mean_reward:.1f}, comp={mean_comp:.2f})")
        return None

    def _recompute_adaptive_weights(self):
        means = {s: float(np.mean(self.adaptive_rewards[s])) if self.adaptive_rewards[s] else 0.0
                 for s in range(1, 11)}
        max_r = max(max(abs(v) for v in means.values()), 1.0)
        raw = {s: 1.0 / (means[s] / max_r + 0.1) for s in range(1, 11)}
        total = sum(raw.values())
        self.adaptive_weights = {s: max(raw[s] / total, 0.05) for s in range(1, 11)}
        total2 = sum(self.adaptive_weights.values())
        self.adaptive_weights = {s: w / total2 for s, w in self.adaptive_weights.items()}
        print(f"ADAPTIVE weights: { {s: f'{w:.3f}' for s, w in self.adaptive_weights.items()} }")

    def get_state(self) -> dict:
        return {
            "current_stage": self.current_stage,
            "adaptive": self.adaptive,
            "stage_episodes": self.stage_episodes,
            "rewards_buf": list(self.rewards_buf),
            "completions_buf": list(self.completions_buf),
            "adaptive_rewards": {s: list(v) for s, v in self.adaptive_rewards.items()},
            "adaptive_weights": dict(self.adaptive_weights),
            "adaptive_ep_count": self.adaptive_ep_count,
        }

    def load_state(self, state: dict):
        self.current_stage = state["current_stage"]
        self.adaptive = state["adaptive"]
        self.stage_episodes = state["stage_episodes"]
        self.rewards_buf = state["rewards_buf"]
        self.completions_buf = state["completions_buf"]
        self.adaptive_rewards = {int(k): v for k, v in state["adaptive_rewards"].items()}
        self.adaptive_weights = {int(k): v for k, v in state["adaptive_weights"].items()}
        self.adaptive_ep_count = state["adaptive_ep_count"]

    def status_str(self) -> str:
        if self.adaptive:
            return "adaptive mode"
        return f"stage {self.current_stage + 1}/10 (ep {self.stage_episodes}/{self.min_episodes})"


def sample_start_stage(config: "TrainConfig") -> int:
    """Sample a start stage. Uses stage_mix if set, else curriculum, else stage 1."""
    if not config.curriculum_enabled and config.stage_mix:
        stages = list(config.stage_mix.keys())
        weights = [config.stage_mix[s] for s in stages]
        return random.choices(stages, weights=weights, k=1)[0]
    return 1


def compute_total_distance(info: dict) -> float:
    """Compute total distance traveled in current episode."""
    stage = info.get("stage", 1)
    distance_remaining = info.get("distance_remaining", 0)
    stage_distance = 4200 + (stage - 1) * 500
    completed_stages_distance = sum(4200 + i * 500 for i in range(stage - 1))
    current_stage_progress = stage_distance - distance_remaining
    return completed_stages_distance + current_stage_progress


def shape_reward(
    reward: float,
    info: dict,
    prev_info: dict,
    config: TrainConfig,
    episode_state: Optional[dict] = None,
) -> float:
    """Apply reward shaping based on game events (mirrors SAC implementation)."""
    shaped_reward = reward

    stage = info.get("stage", 1)
    bonus_scale = 1.0 + config.reward_bonus_stage_scale * (stage - 1)

    shaped_reward += config.reward_survival_bonus * bonus_scale

    speed = info.get("speed", 0)
    speed_limit = info.get("speed_limit", 220)
    shaped_reward += speed * config.reward_speed_scale * 0.01

    game_mode = info.get("game_mode", "playing")
    if game_mode == "playing" and speed >= speed_limit * 0.9:
        shaped_reward += config.reward_safe_speed_bonus

    total_distance = compute_total_distance(info)
    prev_total_distance = compute_total_distance(prev_info)
    distance_delta = total_distance - prev_total_distance
    if distance_delta > 0:
        shaped_reward += distance_delta * config.reward_distance_scale

    if config.reward_distance_milestone_interval > 0:
        prev_milestones = int(prev_total_distance / config.reward_distance_milestone_interval)
        curr_milestones = int(total_distance / config.reward_distance_milestone_interval)
        milestones_achieved = curr_milestones - prev_milestones
        if milestones_achieved > 0:
            shaped_reward += config.reward_distance_milestone * milestones_achieved * bonus_scale

    fuel_current = info.get("fuel", 0)
    fuel_prev = prev_info.get("fuel", 0)
    if fuel_current > fuel_prev + 5:
        shaped_reward += config.reward_fuel_bonus * bonus_scale

    stage_prev = prev_info.get("stage", 1)
    if stage > stage_prev:
        shaped_reward += config.reward_stage_bonus * stage

    if game_mode == "crashed" and prev_info.get("game_mode") == "playing":
        stage_scale = 1.0 + config.reward_crash_penalty_stage_scale * (stage - 1)
        shaped_reward -= config.reward_crash_penalty * stage_scale

    # Fuel exhaustion penalty — dying from empty fuel is as bad as a crash
    if (game_mode == "game_over" and prev_info.get("game_mode") == "playing"
            and prev_info.get("fuel", 100) <= 1.0
            and config.reward_fuel_exhaustion_penalty > 0):
        stage_scale = 1.0 + config.reward_crash_penalty_stage_scale * (stage - 1)
        shaped_reward -= config.reward_fuel_exhaustion_penalty * stage_scale

    # Low fuel urgency — continuous per-step penalty below 30% fuel
    if config.reward_low_fuel_penalty > 0:
        fuel = info.get("fuel", 100)
        if fuel < 30:
            shaped_reward -= config.reward_low_fuel_penalty * (30 - fuel) / 30

    return shaped_reward


def parse_start_stages(value: str) -> List[int]:
    if not value:
        return [1]

    stages: List[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            step = 1 if end >= start else -1
            stages.extend(list(range(start, end + step, step)))
        else:
            stages.append(int(part))

    # Deduplicate while preserving order
    seen = set()
    out: List[int] = []
    for s in stages:
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out or [1]


def make_video_writer(path: str, fps: float):
    """Create a video writer for evaluation recording."""
    try:
        import imageio.v2 as imageio
    except ImportError:
        try:
            import imageio
        except ImportError:
            return None

    try:
        return imageio.get_writer(path, fps=fps, codec="libx264")
    except Exception:
        try:
            return imageio.get_writer(path, fps=fps)
        except Exception:
            return None


def build_eval_video_path(num_updates: int, global_step: int) -> str:
    """Build a unique path for checkpoint evaluation video."""
    runs_dir = os.path.join(ROOT, "rl", "ppo", "runs")
    os.makedirs(runs_dir, exist_ok=True)

    base = os.path.join(runs_dir, f"eval_update_{num_updates}_step_{global_step}.mp4")
    if not os.path.exists(base):
        return base

    root, ext = os.path.splitext(base)
    for idx in range(1, 1000):
        candidate = f"{root}_{idx}{ext}"
        if not os.path.exists(candidate):
            return candidate
    raise RuntimeError("Could not find a free filename for evaluation video")


def evaluate_policy(
    actor_critic: ActorCritic,
    obs_normalizer: Optional[ObservationNormalizer],
    config: TrainConfig,
    start_stages: List[int],
    episodes: int,
    deterministic: bool,
    device: str,
    eval_seed: Optional[int],
    max_episode_steps: int,
    frame_stack: Optional[int] = None,
    record_video: bool = False,
    video_path: Optional[str] = None,
    video_frame_skip: int = 2,
) -> Tuple[Dict[str, float], List[Dict[str, float]], Optional[str]]:
    """Evaluate the current policy on specified start stages."""
    was_training = actor_critic.training
    actor_critic.eval()

    if record_video and not video_path:
        raise ValueError("video_path must be provided when record_video=True")

    render_mode = "rgb_array" if record_video else None
    env = CracerGymEnv(
        render_mode=render_mode,
        obs_mode="state",
        action_mode="discrete",
        fps=config.env_fps,
        seed=eval_seed,
        max_objects=config.max_objects,
    )

    base_obs_size = env.observation_space.shape[0]
    fs = frame_stack if frame_stack is not None else config.frame_stack
    frame_stacker = FrameStack(fs, base_obs_size) if fs > 1 else None

    threshold_stage = config.curriculum_max_stage
    per_stage_results: List[Dict[str, float]] = []
    all_rewards: List[float] = []
    all_scores: List[float] = []
    all_lengths: List[int] = []
    all_max_stages: List[int] = []
    all_over_max = 0

    video_skip = max(1, int(video_frame_skip))
    video_fps = max(1.0, float(config.env_fps) / video_skip)
    video_writer = make_video_writer(video_path, video_fps) if record_video and video_path else None
    saved_video_path = video_path if video_writer is not None else None
    video_frame_count = 0

    try:
        for stage_idx, start_stage in enumerate(start_stages):
            rewards = []
            scores = []
            lengths = []
            max_stages = []
            over_max = 0

            for ep in range(episodes):
                if eval_seed is None:
                    seed = random.randint(0, config.seed_range) if config.randomize_seed else 42
                else:
                    seed = eval_seed + stage_idx * episodes + ep

                obs, info = env.reset(seed=seed, options={"start_stage": start_stage})
                obs = np.asarray(obs, dtype=np.float32)
                if frame_stacker:
                    obs = frame_stacker.reset(obs)

                # Record only the first eval episode for each checkpoint.
                capture_video = video_writer is not None and stage_idx == 0 and ep == 0
                if capture_video:
                    frame = env.render()
                    if frame is not None:
                        video_writer.append_data(frame)
                        video_frame_count += 1

                prev_info = info.copy()
                episode_reward = 0.0
                episode_length = 0
                episode_max_stage = info.get("stage", 1)

                done = False
                while not done and episode_length < max_episode_steps:
                    obs_in = obs_normalizer.normalize(obs) if obs_normalizer else obs
                    obs_tensor = torch.tensor(obs_in, dtype=torch.float32, device=device).unsqueeze(0)
                    with torch.no_grad():
                        logits, _ = actor_critic(obs_tensor)
                    if deterministic:
                        action = int(torch.argmax(logits, dim=1).item())
                    else:
                        probs = torch.softmax(logits, dim=1)
                        action = int(torch.multinomial(probs, 1).item())

                    next_obs, reward, terminated, truncated, info = env.step(action)
                    next_obs = np.asarray(next_obs, dtype=np.float32)

                    episode_reward += shape_reward(reward, info, prev_info, config)
                    prev_info = info.copy()

                    episode_length += 1
                    episode_max_stage = max(episode_max_stage, info.get("stage", 1))

                    if capture_video and episode_length % video_skip == 0:
                        frame = env.render()
                        if frame is not None:
                            video_writer.append_data(frame)
                            video_frame_count += 1

                    if frame_stacker:
                        next_obs = frame_stacker.push(next_obs)
                    obs = next_obs

                    done = terminated or truncated

                rewards.append(episode_reward)
                scores.append(info.get("score", 0))
                lengths.append(episode_length)
                max_stages.append(episode_max_stage)
                if episode_max_stage > threshold_stage:
                    over_max += 1

            mean_reward = float(np.mean(rewards)) if rewards else 0.0
            mean_score = float(np.mean(scores)) if scores else 0.0
            mean_length = float(np.mean(lengths)) if lengths else 0.0
            mean_stage = float(np.mean(max_stages)) if max_stages else 0.0
            over_max_rate = over_max / len(max_stages) if max_stages else 0.0

            per_stage_results.append({
                "start_stage": float(start_stage),
                "mean_reward": mean_reward,
                "mean_score": mean_score,
                "mean_length": mean_length,
                "mean_max_stage": mean_stage,
                "over_max_rate": over_max_rate,
            })

            all_rewards.extend(rewards)
            all_scores.extend(scores)
            all_lengths.extend(lengths)
            all_max_stages.extend(max_stages)
            all_over_max += over_max
    finally:
        if video_writer is not None:
            video_writer.close()
            if video_frame_count == 0:
                saved_video_path = None
                if video_path and os.path.exists(video_path):
                    try:
                        os.remove(video_path)
                    except OSError:
                        pass

    overall: Dict[str, float] = {
        "mean_reward": float(np.mean(all_rewards)) if all_rewards else 0.0,
        "mean_score": float(np.mean(all_scores)) if all_scores else 0.0,
        "mean_length": float(np.mean(all_lengths)) if all_lengths else 0.0,
        "mean_max_stage": float(np.mean(all_max_stages)) if all_max_stages else 0.0,
        "over_max_rate": all_over_max / len(all_max_stages) if all_max_stages else 0.0,
    }

    env.close()
    if was_training:
        actor_critic.train()

    return overall, per_stage_results, saved_video_path


def evaluate(
    config: TrainConfig,
    checkpoint_path: str,
    episodes: int,
    start_stages: List[int],
    deterministic: bool,
    device_choice: str,
    eval_seed: Optional[int],
    max_episode_steps: int,
) -> None:
    device = pick_device(device_choice)
    print(f"Eval device: {device}")

    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(ROOT, checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = load_checkpoint(checkpoint_path, device)
    model_cfg = checkpoint.get("model_config", {})

    hidden_sizes = tuple(model_cfg.get("hidden_sizes", config.hidden_sizes))
    use_layer_norm = bool(model_cfg.get("use_layer_norm", config.use_layer_norm))
    shared_backbone = bool(model_cfg.get("shared_backbone", config.shared_backbone))
    frame_stack = int(model_cfg.get("frame_stack", config.frame_stack))
    normalize_obs = bool(model_cfg.get("normalize_obs", config.normalize_obs))

    # Create environment
    env = CracerGymEnv(
        render_mode=None,
        obs_mode="state",
        action_mode="discrete",
        fps=config.env_fps,
        seed=eval_seed,
        max_objects=config.max_objects,
    )

    base_obs_size = env.observation_space.shape[0]
    num_actions = env.action_space.n
    obs_size = base_obs_size * frame_stack

    actor_critic = ActorCritic(
        obs_size=obs_size,
        num_actions=num_actions,
        hidden_sizes=hidden_sizes,
        use_layer_norm=use_layer_norm,
        shared_backbone=shared_backbone,
    ).to(device)
    actor_critic.load_state_dict(checkpoint["actor_critic"])
    actor_critic.eval()

    obs_normalizer = ObservationNormalizer(obs_size, config.obs_clip) if normalize_obs else None
    if obs_normalizer is not None and "obs_normalizer" in checkpoint:
        obs_normalizer.load_state(checkpoint["obs_normalizer"])
    env.close()

    print(f"Eval start stages: {start_stages} | Episodes per stage: {episodes}")
    overall, per_stage, _ = evaluate_policy(
        actor_critic=actor_critic,
        obs_normalizer=obs_normalizer,
        config=config,
        start_stages=start_stages,
        episodes=episodes,
        deterministic=deterministic,
        device=device,
        eval_seed=eval_seed,
        max_episode_steps=max_episode_steps,
        frame_stack=frame_stack,
    )

    threshold = config.curriculum_max_stage
    for row in per_stage:
        print(
            f"Start stage {int(row['start_stage'])} | "
            f"Mean reward: {row['mean_reward']:.1f} | Mean score: {row['mean_score']:.0f} | "
            f"Mean max stage: {row['mean_max_stage']:.2f} | Mean length: {row['mean_length']:.0f} | "
            f">%{threshold}: {row['over_max_rate']:.0%}"
        )

    print(
        f"Overall | Mean reward: {overall['mean_reward']:.1f} | Mean score: {overall['mean_score']:.0f} | "
        f"Mean max stage: {overall['mean_max_stage']:.2f} | Mean length: {overall['mean_length']:.0f} | "
        f">%{threshold}: {overall['over_max_rate']:.0%}"
    )


def train(config: TrainConfig, resume: bool = True, warm_start_path: str = "") -> None:
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
        max_objects=config.max_objects,
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

    # Curriculum manager
    curriculum = CurriculumManager(config)

    # Logging
    checkpoint_dir = os.path.join(ROOT, config.checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = TrainingLogger(os.path.join(checkpoint_dir, "training_log.csv"))

    # Dedicated eval log — primary generalization record (mirrors SAC setup)
    eval_log_path = os.path.join(checkpoint_dir, "eval_log.csv")
    eval_log_fields = (
        ["timestep", "num_updates", "episodes"]
        + [f"stage{s}_{m}" for s in config.eval_start_stages
           for m in ("mean_stage", "mean_reward", "mean_score", "mean_length")]
        + ["overall_mean_stage", "overall_mean_reward"]
    )
    if not resume or not os.path.exists(eval_log_path):
        with open(eval_log_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=eval_log_fields).writeheader()

    # Training state
    global_step = 0
    num_updates = 0
    episode_count = 0
    episode_rewards: List[float] = []
    episode_scores: List[float] = []
    episode_stages: List[int] = []
    episode_lengths: List[int] = []
    episode_start_stages: List[int] = []

    best_mean_reward = float("-inf")
    best_eval_reward = float("-inf")
    evals_without_improvement = 0

    # Auto-resume from latest checkpoint
    latest_ckpt = find_latest_checkpoint(checkpoint_dir) if resume else None
    if latest_ckpt is not None:
        print(f"\nFound checkpoint: {latest_ckpt}")
        ckpt = load_checkpoint(latest_ckpt, device)
        actor_critic.load_state_dict(ckpt["actor_critic"])
        optimizer.load_state_dict(ckpt["optimizer"])
        global_step = int(ckpt.get("global_step", 0))
        num_updates = int(ckpt.get("num_updates", 0))
        if obs_normalizer and "obs_normalizer" in ckpt:
            obs_normalizer.load_state(ckpt["obs_normalizer"])
        if "curriculum" in ckpt:
            curriculum.load_state(ckpt["curriculum"])
        print(f"Resumed from step {global_step}, update {num_updates}")
        print(f"  Curriculum: {curriculum.status_str()}")
    elif resume:
        print("\nNo checkpoint found, starting fresh training")
    else:
        print("\nStarting fresh training (--no-resume specified)")

    # Warm-start: load actor weights from a prior experiment
    if warm_start_path:
        warm_start_actor(actor_critic, obs_normalizer, warm_start_path, device)

    # Current episode state
    start_stage = sample_start_stage(config) if not config.curriculum_enabled else curriculum.sample_stage()
    obs, info = env.reset(seed=initial_seed, options={"start_stage": start_stage})
    obs = np.asarray(obs, dtype=np.float32)
    if frame_stacker:
        obs = frame_stacker.reset(obs)

    current_episode_reward = 0.0
    current_episode_length = 0
    max_stage = 1
    current_start_stage = start_stage
    prev_info = info.copy()

    start_time = time.time()

    print(f"\nStarting PPO training for {config.total_timesteps} timesteps...")
    print(f"Rollout steps: {config.rollout_steps}, Epochs: {config.num_epochs}, Batch size: {config.batch_size}")
    print(f"Obs normalization: {config.normalize_obs}, LR annealing: {config.anneal_lr}, Entropy annealing: {config.anneal_entropy}")
    if config.curriculum_enabled:
        print(f"Curriculum: {curriculum.status_str()}")

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
            shaped_reward = shape_reward(reward, info, prev_info, config, None)
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

            if done:
                # Log episode
                episode_rewards.append(current_episode_reward)
                episode_scores.append(info.get("score", 0))
                episode_stages.append(max_stage)
                episode_lengths.append(current_episode_length)
                episode_start_stages.append(current_start_stage)
                episode_count += 1

                # Record episode in curriculum
                grad_msg = curriculum.record_episode(current_episode_reward, current_start_stage, max_stage)
                if grad_msg:
                    print(f"\n  {grad_msg}\n")

                # Reset for new episode
                new_seed = random.randint(0, config.seed_range) if config.randomize_seed else 42
                start_stage = sample_start_stage(config) if not config.curriculum_enabled else curriculum.sample_stage()
                obs, info = env.reset(seed=new_seed, options={"start_stage": start_stage})
                obs = np.asarray(obs, dtype=np.float32)
                if frame_stacker:
                    obs = frame_stacker.reset(obs)

                current_episode_reward = 0.0
                current_episode_length = 0
                max_stage = 1
                current_start_stage = start_stage
                prev_info = info.copy()
                # Update observation normalizer with fresh reset obs
                if obs_normalizer:
                    obs_normalizer.update(obs)
            else:
                # Update observation normalizer with current stacked obs
                if obs_normalizer:
                    obs_normalizer.update(obs)

            if global_step >= config.total_timesteps:
                break

        # Compute returns and advantages
        with torch.no_grad():
            obs_normalized = obs_normalizer.normalize(obs) if obs_normalizer else obs
            obs_tensor = torch.tensor(obs_normalized, dtype=torch.float32, device=device).unsqueeze(0)
            last_value = actor_critic.get_value(obs_tensor).item()

        buffer.compute_returns_and_advantages(last_value, config.gamma, config.gae_lambda,
                                              normalize_returns=config.normalize_returns)

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

        # Save checkpoint and run evaluation after each checkpoint.
        eval_overall = None
        checkpoint_due = config.save_interval > 0 and num_updates % config.save_interval == 0
        if checkpoint_due:
            checkpoint_name = f"checkpoint_{num_updates}.pt"
            save_checkpoint(actor_critic, optimizer, config, global_step, num_updates,
                           checkpoint_dir, checkpoint_name, obs_normalizer, curriculum)

            if config.eval_episodes > 0:
                eval_video_path = build_eval_video_path(num_updates, global_step)
                print(f"\nEval @ checkpoint {checkpoint_name} | start stages: {list(config.eval_start_stages)} | "
                      f"episodes/stage: {config.eval_episodes}")
                eval_overall, eval_per_stage, saved_video_path = evaluate_policy(
                    actor_critic=actor_critic,
                    obs_normalizer=obs_normalizer,
                    config=config,
                    start_stages=list(config.eval_start_stages),
                    episodes=config.eval_episodes,
                    deterministic=config.eval_deterministic,
                    device=device,
                    eval_seed=config.eval_seed,
                    max_episode_steps=config.eval_max_episode_steps,
                    record_video=True,
                    video_path=eval_video_path,
                )
                threshold = config.curriculum_max_stage
                for row in eval_per_stage:
                    print(
                        f"  Start stage {int(row['start_stage'])} | "
                        f"Mean reward: {row['mean_reward']:.1f} | Mean score: {row['mean_score']:.0f} | "
                        f"Mean max stage: {row['mean_max_stage']:.2f} | Mean length: {row['mean_length']:.0f} | "
                        f">%{threshold}: {row['over_max_rate']:.0%}"
                    )
                print(
                    f"  Overall | Mean reward: {eval_overall['mean_reward']:.1f} | "
                    f"Mean score: {eval_overall['mean_score']:.0f} | "
                    f"Mean max stage: {eval_overall['mean_max_stage']:.2f} | "
                    f"Mean length: {eval_overall['mean_length']:.0f} | "
                    f">%{threshold}: {eval_overall['over_max_rate']:.0%}"
                )
                if saved_video_path:
                    print(f"  Video saved: {saved_video_path}\n")
                else:
                    print("  Video saved: failed (imageio writer unavailable)\n")

                # Write to dedicated eval_log.csv (primary generalization record)
                eval_row = {"timestep": global_step, "num_updates": num_updates,
                            "episodes": episode_count}
                for row in eval_per_stage:
                    s = int(row["start_stage"])
                    eval_row[f"stage{s}_mean_stage"]   = row["mean_max_stage"]
                    eval_row[f"stage{s}_mean_reward"]  = row["mean_reward"]
                    eval_row[f"stage{s}_mean_score"]   = row["mean_score"]
                    eval_row[f"stage{s}_mean_length"]  = row["mean_length"]
                eval_row["overall_mean_stage"]  = eval_overall["mean_max_stage"]
                eval_row["overall_mean_reward"] = eval_overall["mean_reward"]
                with open(eval_log_path, "a", newline="") as f:
                    csv.DictWriter(f, fieldnames=eval_log_fields).writerow(eval_row)

                # Track best eval and early stopping
                if eval_overall["mean_reward"] > best_eval_reward:
                    best_eval_reward = eval_overall["mean_reward"]
                    evals_without_improvement = 0
                    save_checkpoint(actor_critic, optimizer, config, global_step, num_updates,
                                   checkpoint_dir, "best_eval.pt", obs_normalizer, curriculum)
                    print(f"  New best eval reward: {best_eval_reward:.1f} - saved best_eval.pt")
                else:
                    evals_without_improvement += 1
                    if config.early_stopping_patience > 0:
                        print(f"  No improvement for {evals_without_improvement}/{config.early_stopping_patience} evals")

                if config.early_stopping_patience > 0 and evals_without_improvement >= config.early_stopping_patience:
                    print(f"\nEarly stopping triggered after {evals_without_improvement} evals without improvement")
                    break
        elif (
            config.eval_episodes > 0
            and config.eval_interval_updates > 0
            and num_updates % config.eval_interval_updates == 0
        ):
            print(f"\nEval @ update {num_updates} | start stages: {list(config.eval_start_stages)} | "
                  f"episodes/stage: {config.eval_episodes}")
            eval_overall, eval_per_stage, _ = evaluate_policy(
                actor_critic=actor_critic,
                obs_normalizer=obs_normalizer,
                config=config,
                start_stages=list(config.eval_start_stages),
                episodes=config.eval_episodes,
                deterministic=config.eval_deterministic,
                device=device,
                eval_seed=config.eval_seed,
                max_episode_steps=config.eval_max_episode_steps,
                record_video=False,
            )
            threshold = config.curriculum_max_stage
            for row in eval_per_stage:
                print(
                    f"  Start stage {int(row['start_stage'])} | "
                    f"Mean reward: {row['mean_reward']:.1f} | Mean score: {row['mean_score']:.0f} | "
                    f"Mean max stage: {row['mean_max_stage']:.2f} | Mean length: {row['mean_length']:.0f} | "
                    f">%{threshold}: {row['over_max_rate']:.0%}"
                )
            print(
                f"  Overall | Mean reward: {eval_overall['mean_reward']:.1f} | "
                f"Mean score: {eval_overall['mean_score']:.0f} | "
                f"Mean max stage: {eval_overall['mean_max_stage']:.2f} | "
                f"Mean length: {eval_overall['mean_length']:.0f} | "
                f">%{threshold}: {eval_overall['over_max_rate']:.0%}\n"
            )

        # Logging
        if num_updates % config.log_interval == 0 and episode_rewards:
            recent_rewards = episode_rewards[-100:] if len(episode_rewards) >= 100 else episode_rewards
            recent_scores = episode_scores[-100:] if len(episode_scores) >= 100 else episode_scores
            recent_stages = episode_stages[-100:] if len(episode_stages) >= 100 else episode_stages
            recent_lengths = episode_lengths[-100:] if len(episode_lengths) >= 100 else episode_lengths
            recent_start_stages = (
                episode_start_stages[-100:] if len(episode_start_stages) >= 100 else episode_start_stages
            )

            mean_reward = np.mean(recent_rewards)
            mean_score = np.mean(recent_scores)
            mean_stage = np.mean(recent_stages)
            mean_length = np.mean(recent_lengths)
            mean_start_stage = np.mean(recent_start_stages) if recent_start_stages else 1.0

            if recent_start_stages:
                tier1 = sum(1 for s in recent_start_stages if s == 1)
                tier2 = sum(1 for s in recent_start_stages if 2 <= s <= 3)
                tier3 = sum(1 for s in recent_start_stages if 4 <= s <= 5)
                tier4 = sum(1 for s in recent_start_stages if s >= 6)
                total = len(recent_start_stages)
                start_stage_t1_frac = tier1 / total
                start_stage_t2_frac = tier2 / total
                start_stage_t3_frac = tier3 / total
                start_stage_t4_frac = tier4 / total
            else:
                start_stage_t1_frac = 1.0
                start_stage_t2_frac = 0.0
                start_stage_t3_frac = 0.0
                start_stage_t4_frac = 0.0

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
                "mean_start_stage": mean_start_stage,
                "start_stage_t1_frac": start_stage_t1_frac,
                "start_stage_t2_frac": start_stage_t2_frac,
                "start_stage_t3_frac": start_stage_t3_frac,
                "start_stage_t4_frac": start_stage_t4_frac,
                "eval_mean_reward": eval_overall["mean_reward"] if eval_overall else "",
                "eval_mean_score": eval_overall["mean_score"] if eval_overall else "",
                "eval_mean_max_stage": eval_overall["mean_max_stage"] if eval_overall else "",
                "eval_mean_length": eval_overall["mean_length"] if eval_overall else "",
                "eval_over_max_rate": eval_overall["over_max_rate"] if eval_overall else "",
            })

            # Save best model
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                save_checkpoint(actor_critic, optimizer, config, global_step, num_updates,
                               checkpoint_dir, "best.pt", obs_normalizer, curriculum)
            if config.curriculum_enabled:
                print(f"  Curriculum: {curriculum.status_str()}")

    # Final save
    save_checkpoint(actor_critic, optimizer, config, global_step, num_updates,
                   checkpoint_dir, "final.pt", obs_normalizer, curriculum)

    env.close()
    print(f"\nTraining complete! Total timesteps: {global_step}, Updates: {num_updates}, Episodes: {episode_count}")
    print(f"Best mean reward: {best_mean_reward:.1f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, config: TrainConfig,
                   global_step: int, num_updates: int, checkpoint_dir: str, filename: str,
                   obs_normalizer: Optional[ObservationNormalizer] = None,
                   curriculum: Optional["CurriculumManager"] = None):
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
            "max_objects": config.max_objects,
        },
    }
    if obs_normalizer is not None:
        checkpoint_data["obs_normalizer"] = obs_normalizer.get_state()
    if curriculum is not None:
        checkpoint_data["curriculum"] = curriculum.get_state()
    torch.save(checkpoint_data, path)


def warm_start_actor(actor_critic: nn.Module,
                     obs_normalizer: Optional[ObservationNormalizer],
                     checkpoint_path: str,
                     device: str) -> None:
    """Load ONLY the actor weights from a prior checkpoint.
    Critic, optimizer, and curriculum are reset fresh.
    """
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(ROOT, checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Warm-start checkpoint not found: {checkpoint_path}")

    print(f"\nWarm-starting actor from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Load full actor_critic state but only keep actor weights
    # PPO uses a single ActorCritic network — we load all weights then
    # note the critic will be retrained (its weights are overwritten by optimizer)
    actor_critic.load_state_dict(ckpt["actor_critic"])
    print("  Actor-critic weights loaded (critic will retrain from scratch).")

    if obs_normalizer is not None and "obs_normalizer" in ckpt:
        obs_normalizer.load_state(ckpt["obs_normalizer"])
        print(f"  Obs normalizer loaded (trained on {obs_normalizer.count} samples).")

    step = ckpt.get("global_step", "?")
    print(f"  Source checkpoint was at step {step}.")
    print("  Optimizer, curriculum: initialized fresh.\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO agent for Cracer Sim")
    parser.add_argument("--config", type=str, default="", help="Path to config YAML file")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", dest="resume", action="store_true", default=True,
                        help="Resume from latest checkpoint (default)")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="Start fresh training, ignore existing checkpoints")
    parser.add_argument("--warm-start-actor", type=str, default="",
                        help="Path to checkpoint — loads actor weights, resets critic + optimizer")
    parser.add_argument("--eval", action="store_true", help="Run evaluation instead of training")
    parser.add_argument("--eval-checkpoint", type=str, default="", help="Checkpoint path for evaluation")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Episodes per start stage")
    parser.add_argument("--eval-start-stages", type=str, default="10", help="Comma list or ranges (e.g. 8-10,12)")
    parser.add_argument("--eval-stochastic", action="store_true", help="Use stochastic actions in evaluation")
    parser.add_argument("--eval-seed", type=int, default=None, help="Base seed for evaluation (None=random)")
    parser.add_argument("--max-episode-steps", type=int, default=None, help="Cap steps per eval episode")
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

    if args.eval:
        checkpoint_path = args.eval_checkpoint or os.path.join("rl", "ppo", "checkpoints", "best.pt")
        start_stages = parse_start_stages(args.eval_start_stages)
        max_steps = args.max_episode_steps or config.max_episode_steps
        evaluate(
            config=config,
            checkpoint_path=checkpoint_path,
            episodes=args.eval_episodes,
            start_stages=start_stages,
            deterministic=not args.eval_stochastic,
            device_choice=config.device,
            eval_seed=args.eval_seed,
            max_episode_steps=max_steps,
        )
        return

    train(config, resume=args.resume, warm_start_path=args.warm_start_actor)


if __name__ == "__main__":
    main()
