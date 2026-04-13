from __future__ import annotations

"""
Improved DQN Training for Cracer Simulation (v3)

Aligned with SAC reference implementation. Key improvements over v2:
1. Observation normalization (replaces broken reward normalization)
2. CurriculumManager - 10-stage curriculum with adaptive mode
3. Auto-resume from latest checkpoint
4. Centralized shape_reward() function
5. Evaluation loop with video recording
6. max_objects + start_stage support in env reset
7. Double DQN, Dueling, Prioritized Experience Replay (PER)
8. Frame stacking for temporal context
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import deque, namedtuple
from dataclasses import dataclass, fields, asdict
from typing import Deque, Optional, Tuple, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from game import CracerGymEnv  # noqa: E402


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


@dataclass
class TrainConfig:
    # Training duration
    total_episodes: int = 5_000
    max_steps_per_episode: int = 10_000

    # Replay buffer
    memory_size: int = 200_000
    batch_size: int = 64
    min_replay_size: int = 5_000

    # DQN hyperparameters
    gamma: float = 0.99
    learning_rate: float = 1e-4
    lr_decay: float = 1.0
    lr_min: float = 1e-6

    # Exploration
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: float = 50_000.0

    # Target network
    tau: float = 0.005
    target_update_freq: int = 1

    # Network architecture
    hidden_sizes: Tuple[int, ...] = (512, 512, 256)
    use_dueling: bool = True
    use_double: bool = True
    use_layer_norm: bool = True

    # Prioritized Experience Replay
    use_per: bool = True
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_beta_episodes: int = 4_000
    per_epsilon: float = 1e-6

    # Observation normalization (replaces reward normalization)
    normalize_obs: bool = True
    obs_clip: float = 10.0

    # Reward shaping (same scale as SAC)
    reward_speed_scale: float = 0.1
    reward_fuel_bonus: float = 30.0
    reward_crash_penalty: float = 75.0
    reward_pothole_penalty: float = 5.0
    reward_survival_bonus: float = 0.15
    reward_distance_scale: float = 0.1
    reward_stage_bonus: float = 1000.0
    reward_distance_milestone: float = 40.0
    reward_distance_milestone_interval: int = 300
    reward_safe_speed_bonus: float = 0.08

    # DQN-specific: low fuel urgency
    low_fuel_penalty_scale: float = 1.0
    low_fuel_threshold: float = 0.3

    # Gradient clipping
    grad_clip_norm: float = 10.0

    # Frame stacking for temporal context
    frame_stack: int = 4

    # Observation noise for robustness
    obs_noise_std: float = 0.0

    # Dropout for regularization
    dropout_rate: float = 0.0

    # Generalization
    randomize_seed: bool = True
    seed_range: int = 1_000

    # Max objects in observation
    max_objects: int = 10

    # Curriculum Learning
    curriculum_enabled: bool = True
    curriculum_graduation_window: int = 100
    curriculum_min_episodes_per_stage: int = 150
    curriculum_adaptive_eval_interval: int = 50

    # Early stopping
    early_stopping_patience: int = 0  # 0 = disabled

    # Evaluation
    eval_episodes: int = 10
    eval_deterministic: bool = True

    # Visualization
    plot_interval: int = 100
    save_plots: bool = True

    # Checkpointing
    save_every: int = 100
    checkpoint_dir: str = "rl/dqn/checkpoints"
    log_interval: int = 10

    # Environment
    seed: int = 42
    device: str = "auto"
    render: bool = False


class ObservationNormalizer:
    """Running mean/std normalization for observations (Welford's algorithm)."""
    def __init__(self, obs_size: int, clip: float = 10.0, epsilon: float = 1e-8):
        self.obs_size = obs_size
        self.clip = clip
        self.epsilon = epsilon
        self.mean = np.zeros(obs_size, dtype=np.float64)
        self.var = np.ones(obs_size, dtype=np.float64)
        self.count = 0

    def update(self, obs: np.ndarray):
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
        normalized = (obs - self.mean.astype(np.float32)) / (np.sqrt(self.var.astype(np.float32)) + self.epsilon)
        return np.clip(normalized, -self.clip, self.clip)

    def normalize_batch(self, obs: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, device=obs.device, dtype=obs.dtype)
        var = torch.as_tensor(self.var, device=obs.device, dtype=obs.dtype)
        normalized = (obs - mean) / (torch.sqrt(var) + self.epsilon)
        return torch.clamp(normalized, -self.clip, self.clip)

    def get_state(self) -> dict:
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": self.count}

    def load_state(self, state: dict):
        self.mean = state["mean"]
        self.var = state["var"]
        self.count = state["count"]


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
        if not self.adaptive:
            dist = self.STAGES[self.current_stage][0]
        else:
            dist = self.adaptive_weights
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
        if self.stage_episodes < self.min_episodes:
            return None
        if len(self.rewards_buf) < self.window:
            return None
        _, grad_reward, grad_completion = self.STAGES[self.current_stage]
        mean_reward = np.mean(self.rewards_buf)
        mean_comp = np.mean(self.completions_buf)
        reward_ok = mean_reward >= grad_reward
        comp_ok = grad_completion is None or mean_comp >= grad_completion
        if reward_ok and comp_ok:
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
        means = {}
        for s in range(1, 11):
            buf = self.adaptive_rewards[s]
            means[s] = np.mean(buf) if buf else 0.0
        max_r = max(abs(v) for v in means.values()) if means else 1.0
        max_r = max(max_r, 1.0)
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


class TrainingLogger:
    """Logs training metrics to CSV."""

    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        self.log_file = os.path.join(save_dir, "training_log.csv")
        self.fieldnames = [
            "episode", "steps", "reward", "score", "episode_length",
            "epsilon", "learning_rate", "loss", "max_stage",
            "mean_reward_100", "mean_score_100", "mean_stage_100",
            "curriculum_stage",
        ]
        self._init_csv()

    def _init_csv(self):
        os.makedirs(self.save_dir, exist_ok=True)
        with open(self.log_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def log(self, **kwargs):
        with open(self.log_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow({k: kwargs.get(k, "") for k in self.fieldnames})

    def plot(self, save_path: Optional[str] = None):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            return

        try:
            import csv as csv_mod
            rows = []
            with open(self.log_file, "r") as f:
                reader = csv_mod.DictReader(f)
                for row in reader:
                    rows.append(row)
            if len(rows) < 2:
                return

            episodes = [int(r["episode"]) for r in rows if r["episode"]]
            rewards = [float(r["reward"]) for r in rows if r["reward"]]
            mean_rewards = [float(r["mean_reward_100"]) for r in rows if r["mean_reward_100"]]
            stages = [float(r["max_stage"]) for r in rows if r["max_stage"]]
            mean_stages = [float(r["mean_stage_100"]) for r in rows if r["mean_stage_100"]]

            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle("DQN Training Progress", fontsize=14, fontweight='bold')

            ax = axes[0, 0]
            ax.plot(episodes[:len(rewards)], rewards, alpha=0.3, color='blue', label='Episode')
            ax.plot(episodes[:len(mean_rewards)], mean_rewards, color='blue', linewidth=2, label='Mean(100)')
            ax.set_title("Reward")
            ax.legend()
            ax.grid(True, alpha=0.3)

            ax = axes[0, 1]
            ax.plot(episodes[:len(stages)], stages, alpha=0.5, color='purple', label='Max Stage')
            ax.plot(episodes[:len(mean_stages)], mean_stages, color='purple', linewidth=2, label='Mean(100)')
            ax.set_title("Stage Reached")
            ax.legend()
            ax.grid(True, alpha=0.3)

            scores = [float(r["score"]) for r in rows if r["score"]]
            mean_scores = [float(r["mean_score_100"]) for r in rows if r["mean_score_100"]]
            ax = axes[1, 0]
            ax.plot(episodes[:len(scores)], scores, alpha=0.3, color='green', label='Score')
            ax.plot(episodes[:len(mean_scores)], mean_scores, color='green', linewidth=2, label='Mean(100)')
            ax.set_title("Game Score")
            ax.legend()
            ax.grid(True, alpha=0.3)

            losses = [float(r["loss"]) for r in rows if r.get("loss") and r["loss"]]
            loss_eps = [int(r["episode"]) for r in rows if r.get("loss") and r["loss"]]
            ax = axes[1, 1]
            if losses:
                ax.plot(loss_eps, losses, alpha=0.5, color='brown', label='Loss')
            ax.set_title("Training Loss")
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        except Exception:
            pass


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

    @property
    def stacked_size(self) -> int:
        return self.obs_size * self.num_frames


class SumTree:
    """Sum tree for efficient priority sampling in PER."""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float) -> None:
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        return self.tree[0]

    def add(self, priority: float, data: Any) -> None:
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx: int, priority: float) -> None:
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float) -> Tuple[int, float, Any]:
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayMemory:
    """Prioritized Experience Replay buffer."""
    def __init__(self, capacity: int, alpha: float = 0.6, epsilon: float = 1e-6):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.epsilon = epsilon
        self.max_priority = 1.0

    def push(self, *args) -> None:
        transition = Transition(*args)
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, transition)

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[List[Transition], np.ndarray, List[int]]:
        batch = []
        indices = []
        priorities = []
        segment = self.tree.total() / batch_size
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, priority, data = self.tree.get(s)
            if data is None or data == 0:
                s = random.uniform(0, self.tree.total())
                idx, priority, data = self.tree.get(s)
            batch.append(data)
            indices.append(idx)
            priorities.append(priority)
        total = self.tree.total()
        probs = np.array(priorities) / total
        weights = (self.tree.n_entries * probs) ** (-beta)
        weights = weights / weights.max()
        return batch, weights.astype(np.float32), indices

    def update_priorities(self, indices: List[int], td_errors: np.ndarray) -> None:
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)

    def __len__(self) -> int:
        return self.tree.n_entries


class ReplayMemory:
    """Standard uniform replay buffer."""
    def __init__(self, capacity: int) -> None:
        self.memory = deque(maxlen=capacity)

    def push(self, *args) -> None:
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[List[Transition], np.ndarray, None]:
        transitions = random.sample(self.memory, batch_size)
        weights = np.ones(batch_size, dtype=np.float32)
        return transitions, weights, None

    def update_priorities(self, indices, td_errors) -> None:
        pass

    def __len__(self) -> int:
        return len(self.memory)


class DuelingQNetwork(nn.Module):
    """Dueling DQN architecture."""
    def __init__(self, obs_size: int, num_actions: int, hidden_sizes: Tuple[int, ...],
                 use_layer_norm: bool = True, dropout_rate: float = 0.0) -> None:
        super().__init__()
        self.num_actions = num_actions
        layers = []
        last_size = obs_size
        for hidden in hidden_sizes[:-1]:
            layers.append(nn.Linear(last_size, hidden))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            last_size = hidden
        self.feature_net = nn.Sequential(*layers)
        value_layers = [nn.Linear(last_size, hidden_sizes[-1])]
        if use_layer_norm:
            value_layers.append(nn.LayerNorm(hidden_sizes[-1]))
        value_layers.extend([nn.ReLU(), nn.Linear(hidden_sizes[-1], 1)])
        self.value_stream = nn.Sequential(*value_layers)
        adv_layers = [nn.Linear(last_size, hidden_sizes[-1])]
        if use_layer_norm:
            adv_layers.append(nn.LayerNorm(hidden_sizes[-1]))
        adv_layers.extend([nn.ReLU(), nn.Linear(hidden_sizes[-1], num_actions)])
        self.advantage_stream = nn.Sequential(*adv_layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in [self.value_stream[-1], self.advantage_stream[-1]]:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_net(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class QNetwork(nn.Module):
    """Standard Q-Network."""
    def __init__(self, obs_size: int, num_actions: int, hidden_sizes: Tuple[int, ...],
                 use_layer_norm: bool = True, dropout_rate: float = 0.0) -> None:
        super().__init__()
        layers = []
        last_size = obs_size
        for hidden in hidden_sizes:
            layers.append(nn.Linear(last_size, hidden))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            last_size = hidden
        layers.append(nn.Linear(last_size, num_actions))
        self.net = nn.Sequential(*layers)
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)
        nn.init.constant_(self.net[-1].bias, 0.0)

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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_total_distance(info: dict) -> float:
    """Compute total distance traveled in current episode."""
    stage = info.get("stage", 1)
    distance_remaining = info.get("distance_remaining", 0)
    stage_distance = 4200 + (stage - 1) * 500
    completed_stages_distance = sum(4200 + i * 500 for i in range(stage - 1))
    current_stage_progress = stage_distance - distance_remaining
    return completed_stages_distance + current_stage_progress


def shape_reward(reward: float, info: dict, prev_info: dict, config: TrainConfig,
                 episode_state: dict, max_fuel: float = 100.0) -> float:
    """Centralized reward shaping (mirrors SAC implementation)."""
    shaped = reward

    shaped += config.reward_survival_bonus

    speed = info.get("speed", 0)
    speed_limit = info.get("speed_limit", 220)
    shaped += speed * config.reward_speed_scale * 0.01

    game_mode = info.get("game_mode", "playing")
    if game_mode == "playing" and speed >= speed_limit * 0.9:
        shaped += config.reward_safe_speed_bonus

    total_distance = compute_total_distance(info)
    prev_total_distance = compute_total_distance(prev_info)
    distance_delta = total_distance - prev_total_distance
    if distance_delta > 0:
        shaped += distance_delta * config.reward_distance_scale

    milestone_interval = config.reward_distance_milestone_interval
    prev_milestones = int(prev_total_distance / milestone_interval)
    curr_milestones = int(total_distance / milestone_interval)
    milestones_achieved = curr_milestones - prev_milestones
    if milestones_achieved > 0:
        shaped += config.reward_distance_milestone * milestones_achieved

    fuel_current = info.get("fuel", 0)
    fuel_prev = prev_info.get("fuel", 0)
    if fuel_current > fuel_prev + 5:
        shaped += config.reward_fuel_bonus

    stage_current = info.get("stage", 1)
    stage_prev = prev_info.get("stage", 1)
    if stage_current > stage_prev:
        shaped += config.reward_stage_bonus * stage_current

    if game_mode == "crashed" and prev_info.get("game_mode") == "playing":
        shaped -= config.reward_crash_penalty

    # DQN-specific: low fuel urgency
    if config.low_fuel_penalty_scale > 0 and config.low_fuel_threshold > 0:
        fuel_ratio = max(0.0, min(1.0, fuel_current / max_fuel))
        if fuel_ratio < config.low_fuel_threshold:
            urgency = (config.low_fuel_threshold - fuel_ratio) / config.low_fuel_threshold
            shaped -= config.low_fuel_penalty_scale * urgency

    return shaped


def make_video_writer(path: str, fps: float):
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
            if path.endswith(".mp4"):
                path = path.replace(".mp4", ".gif")
            return imageio.get_writer(path, fps=fps)
        except Exception:
            return None


def evaluate(
    policy_net: nn.Module,
    config: TrainConfig,
    obs_normalizer: Optional[ObservationNormalizer],
    device: str,
    num_episodes: int = 10,
    deterministic: bool = True,
    global_step: int = 0,
    record_video: bool = True,
    video_dir: Optional[str] = None,
) -> Dict[str, float]:
    """Run evaluation episodes and return metrics."""
    render_mode = "rgb_array" if record_video else None
    eval_env = CracerGymEnv(
        render_mode=render_mode,
        obs_mode="state",
        action_mode="discrete",
        fps=60,
        seed=42,
        max_objects=config.max_objects,
    )

    base_obs_size = eval_env.observation_space.shape[0]
    frame_stacker = FrameStack(config.frame_stack, base_obs_size) if config.frame_stack > 1 else None

    episode_rewards = []
    episode_scores = []
    episode_stages = []
    episode_lengths = []

    if video_dir is None:
        video_dir = os.path.join(ROOT, "rl", "dqn", "runs")
    os.makedirs(video_dir, exist_ok=True)

    best_episode_reward = float("-inf")
    best_episode_frames: List[np.ndarray] = []
    record_fps = 30

    policy_net.eval()

    for ep in range(num_episodes):
        obs, info = eval_env.reset(seed=42 + ep)
        obs = np.asarray(obs, dtype=np.float32)
        if frame_stacker:
            obs = frame_stacker.reset(obs)

        episode_reward = 0.0
        episode_length = 0
        max_stage = 1
        prev_info = info.copy()
        episode_state: Dict[str, Any] = {}
        done = False
        episode_frames: List[np.ndarray] = []

        if record_video and render_mode == "rgb_array":
            frame = eval_env.render()
            if frame is not None:
                episode_frames.append(frame)

        while not done and episode_length < config.max_steps_per_episode:
            obs_in = obs_normalizer.normalize(obs) if obs_normalizer else obs
            with torch.no_grad():
                obs_tensor = torch.tensor(obs_in, dtype=torch.float32, device=device).unsqueeze(0)
                q_values = policy_net(obs_tensor)
                action = int(torch.argmax(q_values, dim=1).item())

            next_obs, reward, terminated, truncated, info = eval_env.step(action)
            next_obs = np.asarray(next_obs, dtype=np.float32)

            shaped_reward = shape_reward(reward, info, prev_info, config, episode_state)
            prev_info = info.copy()

            episode_reward += shaped_reward
            episode_length += 1
            max_stage = max(max_stage, info.get("stage", 1))
            done = terminated or truncated

            if record_video and render_mode == "rgb_array" and episode_length % 2 == 0:
                frame = eval_env.render()
                if frame is not None:
                    episode_frames.append(frame)

            if frame_stacker:
                obs = frame_stacker.push(next_obs)
            else:
                obs = next_obs

        episode_rewards.append(episode_reward)
        episode_scores.append(info.get("score", 0))
        episode_stages.append(max_stage)
        episode_lengths.append(episode_length)

        if episode_reward > best_episode_reward:
            best_episode_reward = episode_reward
            best_episode_frames = episode_frames

    eval_env.close()
    policy_net.train()

    video_path = None
    if record_video and best_episode_frames:
        video_filename = f"eval_step_{global_step}.mp4"
        video_path = os.path.join(video_dir, video_filename)
        writer = make_video_writer(video_path, record_fps)
        if writer is not None:
            for frame in best_episode_frames:
                writer.append_data(frame)
            writer.close()
        else:
            video_path = None

    return {
        "eval_mean_reward": float(np.mean(episode_rewards)),
        "eval_std_reward": float(np.std(episode_rewards)),
        "eval_mean_score": float(np.mean(episode_scores)),
        "eval_mean_stage": float(np.mean(episode_stages)),
        "eval_mean_length": float(np.mean(episode_lengths)),
        "eval_min_reward": float(np.min(episode_rewards)),
        "eval_max_reward": float(np.max(episode_rewards)),
        "eval_video_path": video_path,
    }


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the latest numbered checkpoint in the directory."""
    if not os.path.exists(checkpoint_dir):
        return None
    checkpoint_files = []
    for f in os.listdir(checkpoint_dir):
        if f.startswith("checkpoint_") and f.endswith(".pt"):
            try:
                ep = int(f.replace("checkpoint_", "").replace(".pt", ""))
                checkpoint_files.append((ep, f))
            except ValueError:
                continue
    if not checkpoint_files:
        return None
    checkpoint_files.sort(key=lambda x: x[0], reverse=True)
    return os.path.join(checkpoint_dir, checkpoint_files[0][1])


def save_checkpoint(
    path: str,
    policy_net: nn.Module,
    target_net: nn.Module,
    optimizer: torch.optim.Optimizer,
    episode: int,
    steps_done: int,
    epsilon: float,
    best_mean: float,
    obs_normalizer: Optional[ObservationNormalizer] = None,
    model_config: Optional[dict] = None,
    curriculum: Optional[CurriculumManager] = None,
) -> None:
    payload = {
        "q_net": policy_net.state_dict(),
        "target_net": target_net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "episode": episode,
        "steps_done": steps_done,
        "epsilon": epsilon,
        "best_mean": best_mean,
    }
    if obs_normalizer is not None:
        payload["obs_normalizer"] = obs_normalizer.get_state()
    if model_config is not None:
        payload["model_config"] = model_config
    if curriculum is not None:
        payload["curriculum"] = curriculum.get_state()
    torch.save(payload, path)


def load_checkpoint(path: str, device: str):
    return torch.load(path, map_location=device, weights_only=False)


def add_observation_noise(obs: np.ndarray, noise_std: float) -> np.ndarray:
    if noise_std <= 0:
        return obs
    return obs + np.random.normal(0, noise_std, obs.shape).astype(np.float32)


def pump_pygame_events() -> None:
    try:
        import pygame
    except Exception:
        return
    pygame.event.pump()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DQN agent for cracer-simulation")
    default_config = os.path.join(os.path.dirname(__file__), "config.yaml")
    parser.add_argument("--config", type=str, default=default_config)
    parser.add_argument("--resume", dest="resume", action="store_true", default=True,
                        help="Resume from latest checkpoint (default)")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="Start fresh training, ignore existing checkpoints")
    return parser.parse_args()


def load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required. Install with `pip install pyyaml`.") from exc
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a YAML mapping (key/value pairs).")
    return data


def build_config(config_data: Dict[str, Any]) -> TrainConfig:
    cfg = TrainConfig()
    valid_fields = {field.name for field in fields(TrainConfig)}
    extras = sorted(set(config_data.keys()) - valid_fields)
    if extras:
        print(f"Warning: unknown config keys ignored: {', '.join(extras)}")
    for field in fields(TrainConfig):
        if field.name in config_data:
            setattr(cfg, field.name, config_data[field.name])
    if isinstance(cfg.hidden_sizes, list):
        cfg.hidden_sizes = tuple(int(x) for x in cfg.hidden_sizes)
    return cfg


def main() -> None:
    args = parse_args()
    config = build_config(load_yaml(args.config))
    if not os.path.isabs(config.checkpoint_dir):
        config.checkpoint_dir = os.path.join(ROOT, config.checkpoint_dir)

    device = pick_device(config.device)
    print(f"Device: {device}")
    print(f"Double DQN: {config.use_double}, Dueling: {config.use_dueling}, PER: {config.use_per}")
    print(f"Frame stack: {config.frame_stack}, Obs normalization: {config.normalize_obs}")
    set_seed(int(config.seed))

    env = CracerGymEnv(
        render_mode="human" if config.render else None,
        obs_mode="state",
        action_mode="discrete",
        fps=60,
        seed=int(config.seed),
        max_objects=config.max_objects,
    )
    obs, info = env.reset(seed=int(config.seed))
    max_fuel = float(getattr(env.env, "max_fuel", 100.0))

    base_obs_size = env.observation_space.shape[0]
    num_actions = env.action_space.n
    obs_size = base_obs_size * config.frame_stack

    frame_stacker = FrameStack(config.frame_stack, base_obs_size) if config.frame_stack > 1 else None

    model_config = {
        "obs_size": int(obs_size),
        "num_actions": int(num_actions),
        "hidden_sizes": list(config.hidden_sizes),
        "use_dueling": config.use_dueling,
        "use_layer_norm": config.use_layer_norm,
        "frame_stack": config.frame_stack,
        "dropout_rate": config.dropout_rate,
        "normalize_obs": config.normalize_obs,
        "max_objects": config.max_objects,
    }

    NetworkClass = DuelingQNetwork if config.use_dueling else QNetwork
    policy_net = NetworkClass(
        obs_size, num_actions, config.hidden_sizes, config.use_layer_norm, config.dropout_rate
    ).to(device)
    target_net = NetworkClass(
        obs_size, num_actions, config.hidden_sizes, config.use_layer_norm, config.dropout_rate
    ).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    n_params = sum(p.numel() for p in policy_net.parameters())
    print(f"Network parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(policy_net.parameters(), lr=config.learning_rate, amsgrad=True)

    if config.use_per:
        memory = PrioritizedReplayMemory(config.memory_size, config.per_alpha, config.per_epsilon)
    else:
        memory = ReplayMemory(config.memory_size)

    obs_normalizer = ObservationNormalizer(obs_size, config.obs_clip) if config.normalize_obs else None
    curriculum = CurriculumManager(config)

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    logger = TrainingLogger(config.checkpoint_dir)

    # Save config
    config_path = os.path.join(config.checkpoint_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(asdict(config), f, indent=2, default=str)

    start_episode = 0
    steps_done = 0
    best_mean = float("-inf")
    best_eval_reward = float("-inf")
    evals_without_improvement = 0

    # Auto-resume from latest checkpoint
    latest_ckpt = find_latest_checkpoint(config.checkpoint_dir) if args.resume else None
    if latest_ckpt is not None:
        print(f"\nFound checkpoint: {latest_ckpt}")
        ckpt = load_checkpoint(latest_ckpt, device)
        policy_net.load_state_dict(ckpt["q_net"])
        target_net.load_state_dict(ckpt["target_net"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_episode = int(ckpt.get("episode", 0))
        steps_done = int(ckpt.get("steps_done", 0))
        best_mean = float(ckpt.get("best_mean", float("-inf")))
        if obs_normalizer and "obs_normalizer" in ckpt:
            obs_normalizer.load_state(ckpt["obs_normalizer"])
        if "curriculum" in ckpt:
            curriculum.load_state(ckpt["curriculum"])
        print(f"Resumed from episode {start_episode}, steps {steps_done}")
        print(f"  Curriculum: {curriculum.status_str()}")
    elif args.resume:
        print("\nNo checkpoint found, starting fresh training")
    else:
        print("\nStarting fresh training (--no-resume specified)")

    reward_window: Deque[float] = deque(maxlen=100)
    score_window: Deque[float] = deque(maxlen=100)
    stage_window: Deque[int] = deque(maxlen=100)
    loss_window: Deque[float] = deque(maxlen=100)
    last_log_time = time.time()
    last_log_step = steps_done

    def epsilon_by_steps(steps: int) -> float:
        return config.eps_end + (config.eps_start - config.eps_end) * math.exp(-1.0 * steps / config.eps_decay)

    def beta_by_episode(episode: int) -> float:
        progress = min(1.0, episode / config.per_beta_episodes)
        return config.per_beta_start + progress * (config.per_beta_end - config.per_beta_start)

    def select_action(state: torch.Tensor, steps: int) -> Tuple[torch.Tensor, float]:
        eps = epsilon_by_steps(steps)
        if random.random() > eps:
            with torch.no_grad():
                policy_net.eval()
                action = policy_net(state).max(1).indices.view(1, 1)
                policy_net.train()
        else:
            action = torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
        return action, eps

    def optimize_model(beta: float = 0.4) -> Optional[float]:
        if len(memory) < config.min_replay_size:
            return None
        transitions, weights, indices = memory.sample(config.batch_size, beta)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(s is not None for s in batch.next_state), device=device, dtype=torch.bool)
        non_final_next_states = None
        non_final_list = [s for s in batch.next_state if s is not None]
        if non_final_list:
            non_final_next_states = torch.cat(non_final_list)

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        weights_tensor = torch.tensor(weights, device=device)

        # Normalize batches if obs_normalizer is active
        if obs_normalizer:
            state_batch = obs_normalizer.normalize_batch(state_batch)
            if non_final_next_states is not None:
                non_final_next_states = obs_normalizer.normalize_batch(non_final_next_states)

        state_action_values = policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(config.batch_size, device=device)
        if non_final_next_states is not None:
            with torch.no_grad():
                if config.use_double:
                    next_actions = policy_net(non_final_next_states).max(1).indices.unsqueeze(1)
                    next_state_values[non_final_mask] = (
                        target_net(non_final_next_states).gather(1, next_actions).squeeze(1))
                else:
                    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

        expected_state_action_values = reward_batch + (config.gamma * next_state_values)
        td_errors = (state_action_values.squeeze(1) - expected_state_action_values).detach().cpu().numpy()

        element_wise_loss = F.smooth_l1_loss(
            state_action_values.squeeze(1), expected_state_action_values, reduction='none')
        loss = (element_wise_loss * weights_tensor).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), config.grad_clip_norm)
        optimizer.step()

        if indices is not None:
            memory.update_priorities(indices, td_errors)

        return float(loss.item())

    def soft_update_target() -> None:
        for tgt, pol in zip(target_net.parameters(), policy_net.parameters()):
            tgt.data.copy_(config.tau * pol.data + (1.0 - config.tau) * tgt.data)

    def decay_lr() -> float:
        for param_group in optimizer.param_groups:
            new_lr = max(config.lr_min, param_group['lr'] * config.lr_decay)
            param_group['lr'] = new_lr
            return new_lr
        return config.learning_rate

    current_lr = config.learning_rate
    eps_threshold = epsilon_by_steps(steps_done)

    print(f"\nStarting DQN training from episode {start_episode + 1}")
    print(f"Epsilon: {eps_threshold:.3f} -> {config.eps_end:.3f}")
    if config.curriculum_enabled:
        print(f"Curriculum: {curriculum.status_str()}")
    print("-" * 80)

    start_time = time.time()

    for episode_idx in range(start_episode, config.total_episodes):
        episode_seed = random.randint(0, config.seed_range) if config.randomize_seed else config.seed
        start_stage = curriculum.sample_stage()

        obs, info = env.reset(seed=episode_seed, options={"start_stage": start_stage})
        obs = np.asarray(obs, dtype=np.float32)

        if frame_stacker:
            stacked_obs = frame_stacker.reset(obs)
        else:
            stacked_obs = obs.copy()

        if config.obs_noise_std > 0:
            stacked_obs = add_observation_noise(stacked_obs, config.obs_noise_std)

        # Update obs normalizer with initial obs
        if obs_normalizer:
            obs_normalizer.update(stacked_obs)

        prev_info = info.copy()
        episode_state: Dict[str, Any] = {}

        max_stage = start_stage
        episode_losses: List[float] = []

        state_raw = stacked_obs.copy()  # raw for buffer
        state = torch.tensor(stacked_obs, dtype=torch.float32, device=device).unsqueeze(0)
        episode_reward = 0.0
        episode_score = 0.0

        beta = beta_by_episode(episode_idx)

        done = False
        t = 0
        while not done and t < config.max_steps_per_episode:
            action, eps_threshold = select_action(state, steps_done)
            action_int = int(action.item())
            steps_done += 1

            next_obs, reward, terminated, truncated, info = env.step(action_int)
            next_obs = np.asarray(next_obs, dtype=np.float32)

            max_stage = max(max_stage, int(info.get("stage", 1)))
            episode_score = float(info.get("score", 0.0))

            shaped_reward = shape_reward(reward, info, prev_info, config, episode_state, max_fuel)
            prev_info = info.copy()
            episode_reward += shaped_reward

            if frame_stacker:
                next_stacked = frame_stacker.push(next_obs)
            else:
                next_stacked = next_obs.copy()

            if config.obs_noise_std > 0:
                next_stacked = add_observation_noise(next_stacked, config.obs_noise_std)

            # Update obs normalizer
            if obs_normalizer:
                obs_normalizer.update(next_stacked)

            done = terminated or truncated
            reward_tensor = torch.tensor([shaped_reward], device=device)

            next_state_raw = next_stacked
            if not done:
                next_state = torch.tensor(next_stacked, dtype=torch.float32, device=device).unsqueeze(0)
            else:
                next_state = None

            memory.push(state, action, next_state, reward_tensor)
            state = (next_state if next_state is not None
                     else torch.tensor(next_stacked, dtype=torch.float32, device=device).unsqueeze(0))

            loss = optimize_model(beta)
            if loss is not None:
                episode_losses.append(loss)

            if steps_done % config.target_update_freq == 0:
                soft_update_target()

            if config.render:
                env.render()
                pump_pygame_events()

            t += 1

        # Episode done
        reward_window.append(episode_reward)
        score_window.append(episode_score)
        stage_window.append(max_stage)
        avg_loss = sum(episode_losses) / max(1, len(episode_losses)) if episode_losses else 0
        loss_window.append(avg_loss)

        mean_reward = float(np.mean(reward_window))
        mean_score = float(np.mean(score_window))
        mean_stage = float(np.mean(stage_window))

        # Decay learning rate per episode
        current_lr = decay_lr()

        # Record in curriculum
        grad_msg = curriculum.record_episode(episode_reward, start_stage, max_stage)
        if grad_msg:
            print(f"\n  {grad_msg}\n")

        logger.log(
            episode=episode_idx + 1,
            steps=steps_done,
            reward=episode_reward,
            score=episode_score,
            episode_length=t,
            epsilon=eps_threshold,
            learning_rate=current_lr,
            loss=avg_loss,
            max_stage=max_stage,
            mean_reward_100=mean_reward,
            mean_score_100=mean_score,
            mean_stage_100=mean_stage,
            curriculum_stage=curriculum.current_stage + 1 if not curriculum.adaptive else "adaptive",
        )

        if (episode_idx + 1) % config.log_interval == 0:
            now = time.time()
            sps = (steps_done - last_log_step) / max(1e-6, now - last_log_time)
            last_log_time = now
            last_log_step = steps_done
            elapsed = now - start_time
            print(
                f"ep={episode_idx + 1:5d} | steps={steps_done:7d} | "
                f"mean_r={mean_reward:7.1f} | score={mean_score:8.0f} | "
                f"stage={mean_stage:.1f} | eps={eps_threshold:.3f} | "
                f"sps={sps:5.0f} | curriculum={curriculum.status_str()}"
            )

        if config.save_plots and (episode_idx + 1) % config.plot_interval == 0:
            plot_path = os.path.join(config.checkpoint_dir, "training_progress.png")
            logger.plot(save_path=plot_path)

        # Save best training checkpoint
        if len(reward_window) >= 10 and mean_reward > best_mean:
            best_mean = mean_reward
            best_path = os.path.join(config.checkpoint_dir, "best.pt")
            save_checkpoint(best_path, policy_net, target_net, optimizer,
                           episode_idx + 1, steps_done, eps_threshold, best_mean,
                           obs_normalizer, model_config, curriculum)
            print(f"  -> New best model! mean_reward={best_mean:.1f}")

        # Periodic checkpoint + evaluation
        if config.save_every > 0 and (episode_idx + 1) % config.save_every == 0:
            checkpoint_path = os.path.join(config.checkpoint_dir, f"checkpoint_{episode_idx + 1}.pt")
            save_checkpoint(checkpoint_path, policy_net, target_net, optimizer,
                           episode_idx + 1, steps_done, eps_threshold, best_mean,
                           obs_normalizer, model_config, curriculum)

            if config.eval_episodes > 0:
                print(f"\n  Running evaluation ({config.eval_episodes} episodes)...")
                eval_metrics = evaluate(
                    policy_net, config, obs_normalizer, device,
                    num_episodes=config.eval_episodes,
                    deterministic=config.eval_deterministic,
                    global_step=steps_done,
                    record_video=True,
                )
                print(f"  Eval reward: {eval_metrics['eval_mean_reward']:.1f} ± {eval_metrics['eval_std_reward']:.1f}")
                print(f"  Eval score: {eval_metrics['eval_mean_score']:.0f} | "
                      f"Stage: {eval_metrics['eval_mean_stage']:.1f} | "
                      f"Length: {eval_metrics['eval_mean_length']:.0f}")
                if eval_metrics.get('eval_video_path'):
                    print(f"  Video saved: {eval_metrics['eval_video_path']}")
                print()

                if eval_metrics['eval_mean_reward'] > best_eval_reward:
                    best_eval_reward = eval_metrics['eval_mean_reward']
                    evals_without_improvement = 0
                    save_checkpoint(
                        os.path.join(config.checkpoint_dir, "best_eval.pt"),
                        policy_net, target_net, optimizer,
                        episode_idx + 1, steps_done, eps_threshold, best_mean,
                        obs_normalizer, model_config, curriculum)
                    print(f"  New best eval reward: {best_eval_reward:.1f} - saved best_eval.pt")
                else:
                    evals_without_improvement += 1
                    if config.early_stopping_patience > 0:
                        print(f"  No improvement for {evals_without_improvement}/{config.early_stopping_patience} evals")

                if config.early_stopping_patience > 0 and evals_without_improvement >= config.early_stopping_patience:
                    print(f"\nEarly stopping triggered after {evals_without_improvement} evals without improvement")
                    break

    # Final save
    final_path = os.path.join(config.checkpoint_dir, "final.pt")
    save_checkpoint(final_path, policy_net, target_net, optimizer,
                   config.total_episodes, steps_done, eps_threshold, best_mean,
                   obs_normalizer, model_config, curriculum)

    if config.save_plots:
        plot_path = os.path.join(config.checkpoint_dir, "training_progress.png")
        logger.plot(save_path=plot_path)

    env.close()
    print(f"\nTraining complete! Episodes: {config.total_episodes}, Steps: {steps_done}")
    print(f"Best mean reward: {best_mean:.1f}")
    print(f"Checkpoints saved to: {config.checkpoint_dir}")


if __name__ == "__main__":
    main()
