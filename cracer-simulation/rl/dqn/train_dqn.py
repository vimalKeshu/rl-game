from __future__ import annotations

"""
Improved DQN Training for Cracer Simulation (v2)

Key improvements:
1. Double DQN - Reduces overestimation bias
2. Dueling Architecture - Separates state value and action advantage
3. Prioritized Experience Replay (PER) - Samples important transitions more frequently
4. Random Seed Variation - Different seed each episode for generalization
5. Frame Stacking - Temporal context for reactive behavior
6. Observation Noise - Robustness to input variations
7. Training Visualization - Live plots and CSV logging
8. Action Entropy Tracking - Monitor policy diversity
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
from itertools import count

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cracer_sim import CracerGymEnv  # noqa: E402


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))
OBJECT_TYPES = ("enemy", "truck", "race", "fuel", "pothole", "bump")
FUEL_INDEX = 3
BASE_STATE_SIZE = 11
PER_OBJECT_SIZE = 2 + len(OBJECT_TYPES)


@dataclass
class TrainConfig:
    # Training duration
    total_episodes: int = 5_000
    max_steps_per_episode: int = 10_000

    # Replay buffer
    memory_size: int = 100_000
    batch_size: int = 64
    min_replay_size: int = 1_000

    # DQN hyperparameters
    gamma: float = 0.99
    learning_rate: float = 1e-4
    lr_decay: float = 0.9999
    lr_min: float = 1e-6

    # Exploration
    eps_start: float = 1.0
    eps_end: float = 0.02
    eps_decay: float = 50_000.0

    # Target network
    tau: float = 0.001
    target_update_freq: int = 1

    # Network architecture
    hidden_sizes: Tuple[int, ...] = (512, 256, 128)
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

    # Reward shaping
    reward_speed_scale: float = 0.5
    reward_fuel_bonus: float = 100.0
    reward_crash_penalty: float = 200.0
    reward_pothole_penalty: float = 20.0
    reward_bump_penalty: float = 10.0
    reward_survival_bonus: float = 0.5
    reward_distance_scale: float = 1.0
    reward_stage_clear_bonus: float = 500.0

    # Low fuel urgency
    low_fuel_penalty_scale: float = 2.0
    low_fuel_threshold: float = 0.3

    # Fuel direction guidance
    fuel_direction_scale: float = 5.0

    # Episode termination
    terminate_on_crash: bool = False

    # Reward normalization
    normalize_rewards: bool = True
    reward_clip: float = 10.0

    # Gradient clipping
    grad_clip_norm: float = 10.0

    # === NEW: Generalization improvements ===
    # Random seed variation (crucial for generalization!)
    randomize_seed: bool = True  # Use different seed each episode
    seed_range: int = 1_000_000  # Range for random seeds

    # Frame stacking for temporal context
    frame_stack: int = 4  # Number of frames to stack (1 = no stacking)

    # Observation noise for robustness
    obs_noise_std: float = 0.02  # Gaussian noise std (0 = no noise)

    # Dropout for regularization
    dropout_rate: float = 0.1

    # === NEW: Visualization ===
    plot_interval: int = 100  # Update plots every N episodes
    save_plots: bool = True  # Save plot images

    # Checkpointing
    save_every: int = 100
    save_dir: str = "rl/checkpoints"
    log_interval: int = 10

    # Environment
    seed: int = 42
    device: str = "auto"
    resume: str = ""
    render: bool = False


class TrainingLogger:
    """Logs training metrics to CSV and generates plots."""

    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        self.log_file = os.path.join(save_dir, "training_log.csv")
        self.metrics_history: Dict[str, List[float]] = {
            "episode": [],
            "steps": [],
            "reward": [],
            "score": [],
            "episode_length": [],
            "epsilon": [],
            "learning_rate": [],
            "loss": [],
            "max_stage": [],
            "fuel_pickups": [],
            "crashes": [],
            "mean_reward_100": [],
            "mean_score_100": [],
            "mean_stage_100": [],
            "action_entropy": [],
        }
        self._init_csv()

    def _init_csv(self):
        os.makedirs(self.save_dir, exist_ok=True)
        with open(self.log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.metrics_history.keys())

    def log(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)

        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            row = [kwargs.get(k, "") for k in self.metrics_history.keys()]
            writer.writerow(row)

    def plot(self, save_path: Optional[str] = None):
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed, skipping plots")
            return

        episodes = self.metrics_history["episode"]
        if len(episodes) < 2:
            return

        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        fig.suptitle("DQN Training Progress", fontsize=14, fontweight='bold')

        # Plot 1: Episode Reward (with moving average)
        ax = axes[0, 0]
        rewards = self.metrics_history["reward"]
        ax.plot(episodes, rewards, alpha=0.3, color='blue', label='Episode Reward')
        if len(self.metrics_history["mean_reward_100"]) > 0:
            ax.plot(episodes, self.metrics_history["mean_reward_100"],
                   color='blue', linewidth=2, label='Mean (100 ep)')
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.set_title("Episode Reward")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Game Score
        ax = axes[0, 1]
        scores = self.metrics_history["score"]
        ax.plot(episodes, scores, alpha=0.3, color='green', label='Episode Score')
        if len(self.metrics_history["mean_score_100"]) > 0:
            ax.plot(episodes, self.metrics_history["mean_score_100"],
                   color='green', linewidth=2, label='Mean (100 ep)')
        ax.set_xlabel("Episode")
        ax.set_ylabel("Score")
        ax.set_title("Game Score")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Max Stage Reached
        ax = axes[1, 0]
        stages = self.metrics_history["max_stage"]
        ax.plot(episodes, stages, alpha=0.5, color='purple', label='Max Stage')
        if len(self.metrics_history["mean_stage_100"]) > 0:
            ax.plot(episodes, self.metrics_history["mean_stage_100"],
                   color='purple', linewidth=2, label='Mean (100 ep)')
        ax.set_xlabel("Episode")
        ax.set_ylabel("Stage")
        ax.set_title("Max Stage Reached")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Episode Length
        ax = axes[1, 1]
        lengths = self.metrics_history["episode_length"]
        ax.plot(episodes, lengths, alpha=0.3, color='orange')
        # Moving average
        window = min(100, len(lengths))
        if window > 1:
            ma = np.convolve(lengths, np.ones(window)/window, mode='valid')
            ax.plot(episodes[window-1:], ma, color='orange', linewidth=2, label=f'Mean ({window} ep)')
        ax.set_xlabel("Episode")
        ax.set_ylabel("Steps")
        ax.set_title("Episode Length")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 5: Epsilon and Learning Rate
        ax = axes[2, 0]
        ax.plot(episodes, self.metrics_history["epsilon"], color='red', label='Epsilon')
        ax.set_xlabel("Episode")
        ax.set_ylabel("Epsilon", color='red')
        ax.tick_params(axis='y', labelcolor='red')
        ax2 = ax.twinx()
        lr_values = self.metrics_history["learning_rate"]
        ax2.plot(episodes, lr_values, color='cyan', label='Learning Rate')
        ax2.set_ylabel("Learning Rate", color='cyan')
        ax2.tick_params(axis='y', labelcolor='cyan')
        ax.set_title("Exploration & Learning Rate")
        ax.grid(True, alpha=0.3)

        # Plot 6: Loss and Action Entropy
        ax = axes[2, 1]
        losses = [l for l in self.metrics_history["loss"] if l and l > 0]
        loss_eps = [e for e, l in zip(episodes, self.metrics_history["loss"]) if l and l > 0]
        if losses:
            ax.plot(loss_eps, losses, alpha=0.5, color='brown', label='Loss')
            window = min(100, len(losses))
            if window > 1:
                ma = np.convolve(losses, np.ones(window)/window, mode='valid')
                ax.plot(loss_eps[window-1:], ma, color='brown', linewidth=2)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Loss", color='brown')
        ax.tick_params(axis='y', labelcolor='brown')
        ax.set_title("Training Loss")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()


class RunningMeanStd:
    """Tracks running mean and std for reward normalization."""
    def __init__(self, epsilon: float = 1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x: float) -> None:
        batch_mean = x
        batch_var = 0.0
        batch_count = 1
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean: float, batch_var: float, batch_count: int) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        self.mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        self.var = M2 / tot_count
        self.count = tot_count

    def normalize(self, x: float) -> float:
        return (x - self.mean) / (math.sqrt(self.var) + 1e-8)


class FrameStack:
    """Stacks multiple frames for temporal context."""
    def __init__(self, num_frames: int, obs_size: int):
        self.num_frames = num_frames
        self.obs_size = obs_size
        self.frames: Deque[np.ndarray] = deque(maxlen=num_frames)

    def reset(self, obs: np.ndarray) -> np.ndarray:
        """Reset with initial observation."""
        for _ in range(self.num_frames):
            self.frames.append(obs.copy())
        return self.get()

    def push(self, obs: np.ndarray) -> np.ndarray:
        """Add new frame and return stacked observation."""
        self.frames.append(obs.copy())
        return self.get()

    def get(self) -> np.ndarray:
        """Return stacked frames as single array."""
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

    def sample(self, batch_size: int) -> Tuple[List[Transition], np.ndarray, None]:
        transitions = random.sample(self.memory, batch_size)
        weights = np.ones(batch_size, dtype=np.float32)
        return transitions, weights, None

    def update_priorities(self, indices, td_errors) -> None:
        pass

    def __len__(self) -> int:
        return len(self.memory)


class DuelingQNetwork(nn.Module):
    """Dueling DQN architecture with dropout for regularization."""
    def __init__(
        self,
        obs_size: int,
        num_actions: int,
        hidden_sizes: Tuple[int, ...],
        use_layer_norm: bool = True,
        dropout_rate: float = 0.0
    ) -> None:
        super().__init__()
        self.num_actions = num_actions

        # Shared feature extraction layers
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
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values


class QNetwork(nn.Module):
    """Standard Q-Network with dropout."""
    def __init__(
        self,
        obs_size: int,
        num_actions: int,
        hidden_sizes: Tuple[int, ...],
        use_layer_norm: bool = True,
        dropout_rate: float = 0.0
    ) -> None:
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


def save_checkpoint(
    path: str,
    policy_net: nn.Module,
    target_net: nn.Module,
    optimizer: torch.optim.Optimizer,
    episode: int,
    steps_done: int,
    epsilon: float,
    best_mean: float,
    reward_stats: Optional[RunningMeanStd] = None,
    model_config: Optional[dict] = None,
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
    if reward_stats:
        payload["reward_stats"] = {"mean": reward_stats.mean, "var": reward_stats.var, "count": reward_stats.count}
    if model_config:
        payload["model_config"] = model_config
    torch.save(payload, path)


def load_checkpoint(path: str, device: str):
    return torch.load(path, map_location=device, weights_only=False)


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


def nearest_fuel_distance(obs: np.ndarray, max_objects: int, frame_stack: int = 1) -> Optional[float]:
    """Find nearest fuel distance from (possibly stacked) observation."""
    # If frame stacked, use the most recent frame
    single_obs_size = BASE_STATE_SIZE + max_objects * PER_OBJECT_SIZE
    if len(obs) > single_obs_size:
        # Extract most recent frame (last one in stack)
        obs = obs[-single_obs_size:]

    if obs is None or len(obs) < BASE_STATE_SIZE + PER_OBJECT_SIZE:
        return None
    min_dist = None
    for idx in range(max_objects):
        offset = BASE_STATE_SIZE + idx * PER_OBJECT_SIZE
        if offset + PER_OBJECT_SIZE > len(obs):
            break
        dx = float(obs[offset])
        dy = float(obs[offset + 1])
        fuel_flag = float(obs[offset + 2 + FUEL_INDEX])
        if fuel_flag <= 0.5:
            continue
        if dy < -0.1:
            continue
        dist = math.sqrt(dx * dx + dy * dy)
        if min_dist is None or dist < min_dist:
            min_dist = dist
    return min_dist


def add_observation_noise(obs: np.ndarray, noise_std: float) -> np.ndarray:
    """Add Gaussian noise to observation for robustness."""
    if noise_std <= 0:
        return obs
    noise = np.random.normal(0, noise_std, obs.shape).astype(np.float32)
    return obs + noise


def compute_action_entropy(action_counts: Dict[int, int], num_actions: int) -> float:
    """Compute entropy of action distribution (higher = more diverse)."""
    total = sum(action_counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in action_counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log(p + 1e-10)
    # Normalize by max entropy (uniform distribution)
    max_entropy = math.log(num_actions)
    return entropy / max_entropy if max_entropy > 0 else 0.0


def main() -> None:
    args = parse_args()
    config = build_config(load_yaml(args.config))
    if not os.path.isabs(config.save_dir):
        config.save_dir = os.path.join(ROOT, config.save_dir)
    if config.resume and not os.path.isabs(config.resume):
        config.resume = os.path.join(ROOT, config.resume)

    device = pick_device(config.device)
    print(f"Device: {device}")
    print(f"Using Double DQN: {config.use_double}")
    print(f"Using Dueling Architecture: {config.use_dueling}")
    print(f"Using PER: {config.use_per}")
    print(f"Using Layer Normalization: {config.use_layer_norm}")
    print(f"Frame Stacking: {config.frame_stack}")
    print(f"Observation Noise: {config.obs_noise_std}")
    print(f"Random Seed per Episode: {config.randomize_seed}")
    set_seed(int(config.seed))

    # Create environment
    env = CracerGymEnv(
        render_mode="human" if config.render else None,
        obs_mode="state",
        action_mode="discrete",
        seed=int(config.seed),
    )
    obs, info = env.reset(seed=int(config.seed))
    max_fuel = float(getattr(env.env, "max_fuel", 100.0))
    max_objects = int(getattr(env.env, "max_objects", 6))

    base_obs_size = env.observation_space.shape[0]
    num_actions = env.action_space.n

    # Frame stacking
    frame_stacker = FrameStack(config.frame_stack, base_obs_size) if config.frame_stack > 1 else None
    obs_size = base_obs_size * config.frame_stack

    model_config = {
        "obs_size": int(obs_size),
        "num_actions": int(num_actions),
        "hidden_sizes": list(config.hidden_sizes),
        "use_dueling": config.use_dueling,
        "use_layer_norm": config.use_layer_norm,
        "frame_stack": config.frame_stack,
        "dropout_rate": config.dropout_rate,
    }

    # Create networks (both need same architecture for state_dict compatibility)
    NetworkClass = DuelingQNetwork if config.use_dueling else QNetwork
    policy_net = NetworkClass(
        obs_size, num_actions, config.hidden_sizes, config.use_layer_norm, config.dropout_rate
    ).to(device)
    target_net = NetworkClass(
        obs_size, num_actions, config.hidden_sizes, config.use_layer_norm, config.dropout_rate
    ).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()  # eval() mode disables dropout

    n_params = sum(p.numel() for p in policy_net.parameters())
    print(f"Network parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(policy_net.parameters(), lr=config.learning_rate, amsgrad=True)

    # Create replay buffer
    if config.use_per:
        memory = PrioritizedReplayMemory(config.memory_size, config.per_alpha, config.per_epsilon)
    else:
        memory = ReplayMemory(config.memory_size)

    # Reward normalization
    reward_stats = RunningMeanStd() if config.normalize_rewards else None

    # Training logger
    logger = TrainingLogger(config.save_dir)

    # Save config
    config_path = os.path.join(config.save_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(asdict(config), f, indent=2, default=str)

    start_episode = 0
    steps_done = 0
    best_mean = -float("inf")

    if config.resume:
        checkpoint = load_checkpoint(config.resume, device)
        policy_net.load_state_dict(checkpoint["q_net"])
        target_net.load_state_dict(checkpoint["target_net"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_episode = int(checkpoint.get("episode", 0))
        steps_done = int(checkpoint.get("steps_done", 0))
        best_mean = float(checkpoint.get("best_mean", -float("inf")))
        if reward_stats and "reward_stats" in checkpoint:
            rs = checkpoint["reward_stats"]
            reward_stats.mean = rs["mean"]
            reward_stats.var = rs["var"]
            reward_stats.count = rs["count"]
        print(f"Resumed from episode {start_episode}, steps {steps_done}")

    os.makedirs(config.save_dir, exist_ok=True)

    reward_window: Deque[float] = deque(maxlen=100)
    score_window: Deque[float] = deque(maxlen=100)
    steps_window: Deque[int] = deque(maxlen=100)
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
        eps_threshold = epsilon_by_steps(steps)
        if random.random() > eps_threshold:
            with torch.no_grad():
                policy_net.eval()
                action = policy_net(state).max(1).indices.view(1, 1)
                policy_net.train()
        else:
            action = torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
        return action, eps_threshold

    def optimize_model(beta: float = 0.4) -> Optional[float]:
        if len(memory) < config.min_replay_size:
            return None

        transitions, weights, indices = memory.sample(config.batch_size, beta)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(next_state is not None for next_state in batch.next_state),
            device=device,
            dtype=torch.bool,
        )
        non_final_next_states = None
        non_final_list = [s for s in batch.next_state if s is not None]
        if non_final_list:
            non_final_next_states = torch.cat(non_final_list)

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        weights_tensor = torch.tensor(weights, device=device)

        state_action_values = policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(config.batch_size, device=device)
        if non_final_next_states is not None:
            with torch.no_grad():
                if config.use_double:
                    next_actions = policy_net(non_final_next_states).max(1).indices.unsqueeze(1)
                    next_state_values[non_final_mask] = target_net(non_final_next_states).gather(1, next_actions).squeeze(1)
                else:
                    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

        expected_state_action_values = reward_batch + (config.gamma * next_state_values)

        td_errors = (state_action_values.squeeze(1) - expected_state_action_values).detach().cpu().numpy()

        element_wise_loss = F.smooth_l1_loss(
            state_action_values.squeeze(1),
            expected_state_action_values,
            reduction='none'
        )
        loss = (element_wise_loss * weights_tensor).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), config.grad_clip_norm)
        optimizer.step()

        if indices is not None:
            memory.update_priorities(indices, td_errors)

        return float(loss.item())

    def soft_update_target() -> None:
        for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
            target_param.data.copy_(
                config.tau * policy_param.data + (1.0 - config.tau) * target_param.data
            )

    def decay_lr() -> float:
        for param_group in optimizer.param_groups:
            new_lr = max(config.lr_min, param_group['lr'] * config.lr_decay)
            param_group['lr'] = new_lr
            return new_lr
        return config.learning_rate

    current_lr = config.learning_rate
    eps_threshold = epsilon_by_steps(steps_done)

    print(f"\nStarting training from episode {start_episode + 1}")
    print(f"Epsilon: {eps_threshold:.3f} -> {config.eps_end:.3f}")
    print(f"Learning rate: {current_lr:.2e}")
    print("-" * 80)

    for episode_idx in range(start_episode, config.total_episodes):
        # Random seed for generalization
        if config.randomize_seed:
            episode_seed = random.randint(0, config.seed_range)
        else:
            episode_seed = config.seed

        obs, info = env.reset(seed=episode_seed)

        # Frame stacking
        if frame_stacker:
            stacked_obs = frame_stacker.reset(obs)
        else:
            stacked_obs = obs

        # Add observation noise
        if config.obs_noise_std > 0:
            stacked_obs = add_observation_noise(stacked_obs, config.obs_noise_std)

        prev_fuel = float(info.get("fuel", 0.0))
        prev_lives = int(info.get("lives", 0))
        prev_message = info.get("message") or ""
        prev_distance = float(info.get("distance_remaining", 0.0))
        prev_stage = int(info.get("stage", 1))
        prev_fuel_dist = nearest_fuel_distance(obs, max_objects)

        # Episode statistics
        fuel_events = 0
        crash_events = 0
        pothole_events = 0
        bump_events = 0
        max_stage = 1
        action_counts: Dict[int, int] = {i: 0 for i in range(num_actions)}
        episode_losses: List[float] = []

        state = torch.tensor(stacked_obs, dtype=torch.float32, device=device).unsqueeze(0)
        episode_reward = 0.0
        episode_score = 0.0

        beta = beta_by_episode(episode_idx)

        for t in count():
            action, eps_threshold = select_action(state, steps_done)
            action_int = int(action.item())
            action_counts[action_int] = action_counts.get(action_int, 0) + 1
            steps_done += 1

            next_obs, reward, terminated, truncated, info = env.step(action_int)

            message = info.get("message") or ""
            speed = float(info.get("speed", 0.0))
            fuel_now = float(info.get("fuel", prev_fuel))
            lives_now = int(info.get("lives", prev_lives))
            distance_now = float(info.get("distance_remaining", prev_distance))
            stage_now = int(info.get("stage", prev_stage))
            score = float(info.get("score", 0.0))
            episode_score = score
            max_stage = max(max_stage, stage_now)

            dt = getattr(env.env, "dt", 1.0 / 60.0)
            speed_reward = speed * dt * 0.6

            shaped_reward = (reward - speed_reward) + speed_reward * config.reward_speed_scale
            shaped_reward += config.reward_survival_bonus

            if prev_distance > 0:
                distance_progress = prev_distance - distance_now
                if distance_progress > 0:
                    shaped_reward += config.reward_distance_scale * distance_progress * 0.01

            fuel_pickup = (fuel_now - prev_fuel) > 1.0 and not message.startswith("STAGE")
            crash_event = lives_now < prev_lives
            pothole_event = message == "POTHOLE!" and prev_message != "POTHOLE!"
            bump_event = message == "BUMP!" and prev_message != "BUMP!"
            stage_clear = stage_now > prev_stage

            if fuel_pickup:
                shaped_reward += config.reward_fuel_bonus
                fuel_events += 1

            if crash_event:
                shaped_reward -= config.reward_crash_penalty
                crash_events += 1
                if config.terminate_on_crash:
                    terminated = True

            if pothole_event:
                shaped_reward -= config.reward_pothole_penalty
                pothole_events += 1

            if bump_event:
                shaped_reward -= config.reward_bump_penalty
                bump_events += 1

            if stage_clear:
                shaped_reward += config.reward_stage_clear_bonus * stage_now

            if config.low_fuel_penalty_scale > 0.0 and config.low_fuel_threshold > 0.0:
                fuel_ratio = max(0.0, min(1.0, fuel_now / max_fuel))
                if fuel_ratio < config.low_fuel_threshold:
                    urgency = (config.low_fuel_threshold - fuel_ratio) / config.low_fuel_threshold
                    shaped_reward -= config.low_fuel_penalty_scale * urgency

            next_fuel_dist = nearest_fuel_distance(next_obs, max_objects)
            if prev_fuel_dist is not None and next_fuel_dist is not None:
                fuel_dir_delta = prev_fuel_dist - next_fuel_dist
                if config.fuel_direction_scale != 0.0:
                    shaped_reward += config.fuel_direction_scale * fuel_dir_delta
            prev_fuel_dist = next_fuel_dist

            if reward_stats is not None:
                reward_stats.update(shaped_reward)
                shaped_reward = reward_stats.normalize(shaped_reward)
                shaped_reward = max(-config.reward_clip, min(config.reward_clip, shaped_reward))

            reward_value = float(shaped_reward)
            episode_reward += reward_value

            # Frame stacking for next state
            if frame_stacker:
                next_stacked_obs = frame_stacker.push(next_obs)
            else:
                next_stacked_obs = next_obs

            if config.obs_noise_std > 0:
                next_stacked_obs = add_observation_noise(next_stacked_obs, config.obs_noise_std)

            reward_tensor = torch.tensor([reward_value], device=device)
            done = terminated or truncated

            if not done:
                next_state = torch.tensor(next_stacked_obs, dtype=torch.float32, device=device).unsqueeze(0)
            else:
                next_state = None

            memory.push(state, action, next_state, reward_tensor)
            state = next_state if next_state is not None else torch.tensor(next_stacked_obs, dtype=torch.float32, device=device).unsqueeze(0)

            prev_fuel = fuel_now
            prev_lives = lives_now
            prev_message = message
            prev_distance = distance_now
            prev_stage = stage_now

            loss = optimize_model(beta)
            if loss is not None:
                episode_losses.append(loss)

            if steps_done % config.target_update_freq == 0:
                soft_update_target()

            # NOTE: LR decay moved to end of episode (was per-step before)

            if config.render:
                env.render()
                pump_pygame_events()

            if done or t >= config.max_steps_per_episode:
                break

        # Episode statistics
        reward_window.append(episode_reward)
        score_window.append(episode_score)
        steps_window.append(t + 1)
        stage_window.append(max_stage)
        avg_loss = sum(episode_losses) / max(1, len(episode_losses)) if episode_losses else 0
        loss_window.append(avg_loss)

        mean_reward = sum(reward_window) / max(1, len(reward_window))
        mean_score = sum(score_window) / max(1, len(score_window))
        mean_steps = sum(steps_window) / max(1, len(steps_window))
        mean_stage = sum(stage_window) / max(1, len(stage_window))
        mean_loss = sum(loss_window) / max(1, len(loss_window))
        action_entropy = compute_action_entropy(action_counts, num_actions)

        # Decay learning rate per EPISODE (not per step!)
        current_lr = decay_lr()

        # Log metrics
        logger.log(
            episode=episode_idx + 1,
            steps=steps_done,
            reward=episode_reward,
            score=episode_score,
            episode_length=t + 1,
            epsilon=eps_threshold,
            learning_rate=current_lr,
            loss=avg_loss,
            max_stage=max_stage,
            fuel_pickups=fuel_events,
            crashes=crash_events,
            mean_reward_100=mean_reward,
            mean_score_100=mean_score,
            mean_stage_100=mean_stage,
            action_entropy=action_entropy,
        )

        if (episode_idx + 1) % config.log_interval == 0:
            now = time.time()
            step_delta = steps_done - last_log_step
            steps_per_sec = step_delta / max(1e-6, now - last_log_time)
            last_log_time = now
            last_log_step = steps_done

            print(
                f"ep={episode_idx + 1:5d} | "
                f"steps={steps_done:7d} | "
                f"mean_r={mean_reward:7.1f} | "
                f"mean_score={mean_score:8.0f} | "
                f"stage={mean_stage:.1f} | "
                f"eps={eps_threshold:.3f} | "
                f"entropy={action_entropy:.2f} | "
                f"sps={steps_per_sec:5.0f}"
            )

        # Update plots
        if config.save_plots and (episode_idx + 1) % config.plot_interval == 0:
            plot_path = os.path.join(config.save_dir, "training_progress.png")
            logger.plot(save_path=plot_path)

        # Save best checkpoint
        if len(reward_window) >= 10 and mean_reward > best_mean:
            best_mean = mean_reward
            best_path = os.path.join(config.save_dir, "best.pt")
            save_checkpoint(
                best_path, policy_net, target_net, optimizer,
                episode_idx + 1, steps_done, eps_threshold, best_mean,
                reward_stats, model_config
            )
            print(f"  -> New best model saved! mean_reward={best_mean:.1f}")

        # Periodic checkpoint
        if config.save_every > 0 and (episode_idx + 1) % config.save_every == 0:
            checkpoint_path = os.path.join(config.save_dir, f"episode_{episode_idx + 1}.pt")
            save_checkpoint(
                checkpoint_path, policy_net, target_net, optimizer,
                episode_idx + 1, steps_done, eps_threshold, best_mean,
                reward_stats, model_config
            )

    # Save final checkpoint and plots
    final_path = os.path.join(config.save_dir, "final.pt")
    save_checkpoint(
        final_path, policy_net, target_net, optimizer,
        config.total_episodes, steps_done, eps_threshold, best_mean,
        reward_stats, model_config
    )

    if config.save_plots:
        plot_path = os.path.join(config.save_dir, "training_progress.png")
        logger.plot(save_path=plot_path)
        print(f"Training plots saved to {plot_path}")

    print(f"\nTraining complete! Final checkpoint saved to {final_path}")
    print(f"Best mean reward: {best_mean:.1f}")
    env.close()


if __name__ == "__main__":
    main()
