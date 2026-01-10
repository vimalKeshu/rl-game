from __future__ import annotations

"""
Improved DQN Training for Cracer Simulation

Key improvements over basic DQN:
1. Double DQN - Reduces overestimation bias by decoupling action selection from evaluation
2. Dueling Architecture - Separates state value and action advantage for better learning
3. Prioritized Experience Replay (PER) - Samples important transitions more frequently
4. Balanced Reward Shaping - Fixed reward scale imbalances
5. Learning Rate Scheduling - Anneals LR for stable convergence
6. Survival & Progress Rewards - Incentivizes staying alive and making progress
7. Reward Normalization - Running statistics to normalize reward scale
"""

import argparse
import math
import os
import random
import sys
import time
from collections import deque, namedtuple
from dataclasses import dataclass, fields
from typing import Deque, Optional, Tuple, Dict, Any, List
from itertools import count

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
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
    min_replay_size: int = 1_000  # Start learning after this many transitions

    # DQN hyperparameters
    gamma: float = 0.99
    learning_rate: float = 1e-4
    lr_decay: float = 0.9999  # Per-step LR decay
    lr_min: float = 1e-6

    # Exploration
    eps_start: float = 1.0
    eps_end: float = 0.02
    eps_decay: float = 50_000.0  # Much faster decay

    # Target network
    tau: float = 0.001  # Softer updates for stability
    target_update_freq: int = 1  # Update every step (soft update)

    # Network architecture
    hidden_sizes: Tuple[int, ...] = (512, 256, 128)
    use_dueling: bool = True  # Dueling DQN
    use_double: bool = True   # Double DQN
    use_layer_norm: bool = True  # Layer normalization for stability

    # Prioritized Experience Replay
    use_per: bool = True
    per_alpha: float = 0.6  # Priority exponent (0 = uniform, 1 = full prioritization)
    per_beta_start: float = 0.4  # Importance sampling correction
    per_beta_end: float = 1.0
    per_beta_episodes: int = 4_000  # Episodes to anneal beta
    per_epsilon: float = 1e-6  # Small constant for numerical stability

    # Reward shaping (balanced values)
    reward_speed_scale: float = 0.5  # Keep more of the speed reward
    reward_fuel_bonus: float = 100.0  # Moderate fuel bonus
    reward_crash_penalty: float = 200.0  # Balanced with fuel bonus
    reward_pothole_penalty: float = 20.0
    reward_bump_penalty: float = 10.0
    reward_survival_bonus: float = 0.5  # Per-step survival reward
    reward_distance_scale: float = 1.0  # Reward for distance progress
    reward_stage_clear_bonus: float = 500.0  # Bonus for clearing a stage

    # Low fuel urgency
    low_fuel_penalty_scale: float = 2.0
    low_fuel_threshold: float = 0.3

    # Fuel direction guidance
    fuel_direction_scale: float = 5.0

    # Episode termination
    terminate_on_crash: bool = False  # Don't terminate - let agent learn recovery

    # Reward normalization
    normalize_rewards: bool = True
    reward_clip: float = 10.0  # Clip normalized rewards

    # Gradient clipping
    grad_clip_norm: float = 10.0

    # Checkpointing
    save_every: int = 100
    save_dir: str = "rl/checkpoints"
    log_interval: int = 10

    # Environment
    seed: int = 42
    device: str = "auto"
    resume: str = ""
    render: bool = False


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
                # Handle edge case - sample again
                s = random.uniform(0, self.tree.total())
                idx, priority, data = self.tree.get(s)
            batch.append(data)
            indices.append(idx)
            priorities.append(priority)

        # Importance sampling weights
        total = self.tree.total()
        probs = np.array(priorities) / total
        weights = (self.tree.n_entries * probs) ** (-beta)
        weights = weights / weights.max()  # Normalize

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
        pass  # No-op for uniform sampling

    def __len__(self) -> int:
        return len(self.memory)


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

        # Initialize final layers with smaller weights
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
        # Combine using mean advantage baseline
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values


class QNetwork(nn.Module):
    """Standard Q-Network."""
    def __init__(
        self,
        obs_size: int,
        num_actions: int,
        hidden_sizes: Tuple[int, ...],
        use_layer_norm: bool = True
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

        # Initialize final layer with smaller weights
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


def nearest_fuel_distance(obs: np.ndarray, max_objects: int) -> Optional[float]:
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
    set_seed(int(config.seed))

    env = CracerGymEnv(
        render_mode="human" if config.render else None,
        obs_mode="state",
        action_mode="discrete",
        seed=int(config.seed),
    )
    obs, info = env.reset(seed=int(config.seed))
    max_fuel = float(getattr(env.env, "max_fuel", 100.0))
    max_objects = int(getattr(env.env, "max_objects", 6))

    obs_size = env.observation_space.shape[0]
    num_actions = env.action_space.n
    model_config = {
        "obs_size": int(obs_size),
        "num_actions": int(num_actions),
        "hidden_sizes": list(config.hidden_sizes),
        "use_dueling": config.use_dueling,
        "use_layer_norm": config.use_layer_norm,
    }

    # Create networks
    NetworkClass = DuelingQNetwork if config.use_dueling else QNetwork
    policy_net = NetworkClass(
        obs_size, num_actions, config.hidden_sizes, config.use_layer_norm
    ).to(device)
    target_net = NetworkClass(
        obs_size, num_actions, config.hidden_sizes, config.use_layer_norm
    ).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # Count parameters
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
    last_log_time = time.time()
    last_log_step = steps_done

    def epsilon_by_steps(steps: int) -> float:
        return config.eps_end + (config.eps_start - config.eps_end) * math.exp(-1.0 * steps / config.eps_decay)

    def beta_by_episode(episode: int) -> float:
        """Anneal PER beta from start to end over episodes."""
        progress = min(1.0, episode / config.per_beta_episodes)
        return config.per_beta_start + progress * (config.per_beta_end - config.per_beta_start)

    def select_action(state: torch.Tensor, steps: int) -> Tuple[torch.Tensor, float]:
        eps_threshold = epsilon_by_steps(steps)
        if random.random() > eps_threshold:
            with torch.no_grad():
                action = policy_net(state).max(1).indices.view(1, 1)
        else:
            action = torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
        return action, eps_threshold

    def optimize_model(beta: float = 0.4) -> Tuple[Optional[float], Optional[np.ndarray]]:
        if len(memory) < config.min_replay_size:
            return None, None

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

        # Current Q-values
        state_action_values = policy_net(state_batch).gather(1, action_batch)

        # Compute target Q-values
        next_state_values = torch.zeros(config.batch_size, device=device)
        if non_final_next_states is not None:
            with torch.no_grad():
                if config.use_double:
                    # Double DQN: use policy net to select action, target net to evaluate
                    next_actions = policy_net(non_final_next_states).max(1).indices.unsqueeze(1)
                    next_state_values[non_final_mask] = target_net(non_final_next_states).gather(1, next_actions).squeeze(1)
                else:
                    # Standard DQN
                    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

        expected_state_action_values = reward_batch + (config.gamma * next_state_values)

        # Compute TD errors for PER
        td_errors = (state_action_values.squeeze(1) - expected_state_action_values).detach().cpu().numpy()

        # Weighted Huber loss
        element_wise_loss = F.smooth_l1_loss(
            state_action_values.squeeze(1),
            expected_state_action_values,
            reduction='none'
        )
        loss = (element_wise_loss * weights_tensor).mean()

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping by norm (better than by value)
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), config.grad_clip_norm)

        optimizer.step()

        # Update priorities in PER
        if indices is not None:
            memory.update_priorities(indices, td_errors)

        return float(loss.item()), td_errors

    def soft_update_target() -> None:
        """Soft update target network parameters."""
        for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
            target_param.data.copy_(
                config.tau * policy_param.data + (1.0 - config.tau) * target_param.data
            )

    def decay_lr() -> float:
        """Decay learning rate."""
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
        obs, info = env.reset()
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
        stage_clears = 0

        state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        episode_reward = 0.0
        episode_score = 0.0

        beta = beta_by_episode(episode_idx)

        for t in count():
            action, eps_threshold = select_action(state, steps_done)
            steps_done += 1

            next_obs, reward, terminated, truncated, info = env.step(int(action.item()))

            # Extract info
            message = info.get("message") or ""
            speed = float(info.get("speed", 0.0))
            fuel_now = float(info.get("fuel", prev_fuel))
            lives_now = int(info.get("lives", prev_lives))
            distance_now = float(info.get("distance_remaining", prev_distance))
            stage_now = int(info.get("stage", prev_stage))
            score = float(info.get("score", 0.0))
            episode_score = score

            # Calculate base speed reward
            dt = getattr(env.env, "dt", 1.0 / 60.0)
            speed_reward = speed * dt * 0.6

            # Start with scaled speed reward
            shaped_reward = (reward - speed_reward) + speed_reward * config.reward_speed_scale

            # Survival bonus (per step alive)
            shaped_reward += config.reward_survival_bonus

            # Distance progress reward
            if prev_distance > 0:
                distance_progress = prev_distance - distance_now
                if distance_progress > 0:
                    shaped_reward += config.reward_distance_scale * distance_progress * 0.01

            # Event detection
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
                stage_clears += 1

            # Low fuel urgency penalty
            if config.low_fuel_penalty_scale > 0.0 and config.low_fuel_threshold > 0.0:
                fuel_ratio = max(0.0, min(1.0, fuel_now / max_fuel))
                if fuel_ratio < config.low_fuel_threshold:
                    urgency = (config.low_fuel_threshold - fuel_ratio) / config.low_fuel_threshold
                    shaped_reward -= config.low_fuel_penalty_scale * urgency

            # Fuel direction guidance
            next_fuel_dist = nearest_fuel_distance(next_obs, max_objects)
            if prev_fuel_dist is not None and next_fuel_dist is not None:
                fuel_dir_delta = prev_fuel_dist - next_fuel_dist
                if config.fuel_direction_scale != 0.0:
                    shaped_reward += config.fuel_direction_scale * fuel_dir_delta
            prev_fuel_dist = next_fuel_dist

            # Normalize reward
            if reward_stats is not None:
                reward_stats.update(shaped_reward)
                shaped_reward = reward_stats.normalize(shaped_reward)
                shaped_reward = max(-config.reward_clip, min(config.reward_clip, shaped_reward))

            reward_value = float(shaped_reward)
            episode_reward += reward_value

            reward_tensor = torch.tensor([reward_value], device=device)
            done = terminated or truncated

            if not done:
                next_state = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
            else:
                next_state = None

            memory.push(state, action, next_state, reward_tensor)
            state = next_state if next_state is not None else torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)

            # Update previous values
            prev_fuel = fuel_now
            prev_lives = lives_now
            prev_message = message
            prev_distance = distance_now
            prev_stage = stage_now

            # Optimize model
            optimize_model(beta)

            # Soft update target network
            if steps_done % config.target_update_freq == 0:
                soft_update_target()

            # Decay learning rate
            current_lr = decay_lr()

            if config.render:
                env.render()
                pump_pygame_events()

            if done or t >= config.max_steps_per_episode:
                break

        reward_window.append(episode_reward)
        score_window.append(episode_score)
        steps_window.append(t + 1)
        mean_reward = sum(reward_window) / max(1, len(reward_window))
        mean_score = sum(score_window) / max(1, len(score_window))
        mean_steps = sum(steps_window) / max(1, len(steps_window))

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
                f"mean_len={mean_steps:5.0f} | "
                f"eps={eps_threshold:.3f} | "
                f"lr={current_lr:.2e} | "
                f"sps={steps_per_sec:5.0f} | "
                f"fuel={fuel_events} crash={crash_events} stage={stage_clears}"
            )

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

    # Save final checkpoint
    final_path = os.path.join(config.save_dir, "final.pt")
    save_checkpoint(
        final_path, policy_net, target_net, optimizer,
        config.total_episodes, steps_done, eps_threshold, best_mean,
        reward_stats, model_config
    )
    print(f"\nTraining complete! Final checkpoint saved to {final_path}")
    print(f"Best mean reward: {best_mean:.1f}")
    env.close()


if __name__ == "__main__":
    main()
