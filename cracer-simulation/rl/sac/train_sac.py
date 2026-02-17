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
from typing import Any, Dict, List, Optional, Tuple

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
    max_objects: int = 10  # Number of objects in observation

    # SAC Hyperparameters
    learning_rate: float = 3e-4
    gamma: float = 0.99  # Discount factor (reduced for faster credit assignment)
    tau: float = 0.005  # Soft update coefficient
    alpha: float = 0.2  # Initial entropy coefficient
    auto_alpha: bool = True  # Automatically tune alpha
    target_entropy_ratio: float = 0.5  # Target entropy (reduced to encourage exploitation)
    min_alpha: float = 0.1  # Minimum alpha to prevent entropy collapse
    grad_clip: float = 1.0  # Gradient clipping for Q-networks (0 to disable)

    # Training
    buffer_size: int = 500_000  # Replay buffer size
    batch_size: int = 256
    learning_starts: int = 20_000  # Steps before training starts
    train_freq: int = 1  # Train every N steps
    gradient_steps: int = 2  # Gradient steps per train call
    total_timesteps: int = 3_000_000

    # Network architecture
    hidden_sizes: Tuple[int, ...] = (512, 512, 512, 256)
    use_layer_norm: bool = True

    # Observation normalization
    normalize_obs: bool = True
    obs_clip: float = 10.0

    # Frame stacking (increased for temporal context)
    frame_stack: int = 4

    # Reward shaping (rebalanced for better credit assignment)
    reward_speed_scale: float = 0.1
    reward_fuel_bonus: float = 30.0
    reward_crash_penalty: float = 50.0
    reward_stage_bonus: float = 200.0  # Reduced from 500
    reward_survival_bonus: float = 0.3  # Increased from 0.1
    reward_distance_scale: float = 0.05  # Increased 5x
    reward_pothole_penalty: float = 5.0
    reward_distance_milestone: float = 50.0  # Bonus every milestone
    reward_distance_milestone_interval: int = 500  # Distance between milestones
    reward_safe_speed_bonus: float = 0.05  # Bonus for high speed without crashes

    # Generalization
    randomize_seed: bool = True
    seed_range: int = 1000  # Increased from 100

    # Curriculum Learning
    curriculum_enabled: bool = True
    curriculum_graduation_window: int = 100  # Rolling window for graduation stats
    curriculum_min_episodes_per_stage: int = 200  # Min episodes before graduation
    curriculum_adaptive_eval_interval: int = 50  # Re-compute adaptive weights every N episodes

    # Logging
    log_interval: int = 1000  # Log every N steps
    save_interval: int = 10000  # Save checkpoint every N steps
    checkpoint_dir: str = "rl/sac/checkpoints"

    # Evaluation
    eval_episodes: int = 10  # Number of episodes to run for evaluation
    eval_deterministic: bool = True  # Use deterministic policy during eval

    # Early stopping
    early_stopping_patience: int = 0  # Stop if no improvement for N evals (0 to disable)

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

    def normalize_torch(self, obs: torch.Tensor) -> torch.Tensor:
        """Normalize a torch batch using running statistics."""
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


class CurriculumManager:
    """10-stage curriculum with adaptive mode after graduation.

    Each stage defines an env-stage distribution and graduation thresholds.
    After all 10 stages, switches to adaptive mode where weaker stages
    get more training via inverse-performance weighting.
    """

    STAGES = [
        # (distribution {env_stage: prob}, grad_reward, grad_completion)
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

        self.current_stage = 0  # 0-indexed into STAGES
        self.adaptive = False
        self.stage_episodes = 0  # episodes in current curriculum stage

        # Per-curriculum-stage rolling stats
        self.rewards_buf: List[float] = []
        self.completions_buf: List[float] = []  # 1.0 if agent advanced env stage

        # Adaptive mode: per-env-stage rolling rewards
        self.adaptive_rewards: Dict[int, List[float]] = {s: [] for s in range(1, 11)}
        self.adaptive_weights: Dict[int, float] = {s: 1.0 / 10 for s in range(1, 11)}
        self.adaptive_ep_count = 0

    def sample_stage(self) -> int:
        """Return an env start_stage based on current curriculum or adaptive weights."""
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
        """Record episode result. Returns a log message on graduation or None."""
        if not self.enabled:
            return None

        completed = 1.0 if max_stage > start_stage else 0.0

        if not self.adaptive:
            self.rewards_buf.append(reward)
            self.completions_buf.append(completed)
            self.stage_episodes += 1
            # Trim to window
            if len(self.rewards_buf) > self.window:
                self.rewards_buf = self.rewards_buf[-self.window:]
                self.completions_buf = self.completions_buf[-self.window:]
            return self._check_graduation()
        else:
            # Adaptive mode: track per-env-stage
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
                msg = (f"CURRICULUM: Graduated stage {old} -> {self.current_stage + 1} "
                       f"(reward={mean_reward:.1f}, comp={mean_comp:.2f})")
            else:
                self.adaptive = True
                msg = (f"CURRICULUM: Completed all 10 stages! Entering adaptive mode "
                       f"(reward={mean_reward:.1f}, comp={mean_comp:.2f})")
            return msg
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
        # Renormalize after floor
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


def compute_total_distance(info: dict) -> float:
    """Compute total distance traveled in current episode."""
    stage = info.get("stage", 1)
    distance_remaining = info.get("distance_remaining", 0)
    # Each stage has base distance + 500 per stage after first
    stage_distance = 4200 + (stage - 1) * 500
    # Total distance = sum of completed stages + current stage progress
    completed_stages_distance = sum(4200 + i * 500 for i in range(stage - 1))
    current_stage_progress = stage_distance - distance_remaining
    return completed_stages_distance + current_stage_progress


def shape_reward(reward: float, info: dict, prev_info: dict, config: TrainConfig,
                 episode_state: dict) -> float:
    """Apply reward shaping based on game events.

    Args:
        reward: Base reward from environment
        info: Current step info
        prev_info: Previous step info
        config: Training configuration
        episode_state: Mutable dict tracking episode-level state (e.g., milestones)
    """
    shaped_reward = reward

    # Survival bonus
    shaped_reward += config.reward_survival_bonus

    # Speed bonus (scaled by speed limit ratio for safety awareness)
    speed = info.get("speed", 0)
    speed_limit = info.get("speed_limit", 220)
    shaped_reward += speed * config.reward_speed_scale * 0.01

    # Safe speed bonus: reward for maintaining high speed without being crashed
    game_mode = info.get("game_mode", "playing")
    if game_mode == "playing" and speed >= speed_limit * 0.9:
        shaped_reward += config.reward_safe_speed_bonus

    # Distance progress bonus
    total_distance = compute_total_distance(info)
    prev_total_distance = compute_total_distance(prev_info)
    distance_delta = total_distance - prev_total_distance
    if distance_delta > 0:
        shaped_reward += distance_delta * config.reward_distance_scale

    # Distance milestone bonus (every N distance units)
    milestone_interval = config.reward_distance_milestone_interval
    prev_milestones = int(prev_total_distance / milestone_interval)
    curr_milestones = int(total_distance / milestone_interval)
    milestones_achieved = curr_milestones - prev_milestones
    if milestones_achieved > 0:
        shaped_reward += config.reward_distance_milestone * milestones_achieved
        episode_state["milestones_achieved"] = episode_state.get("milestones_achieved", 0) + milestones_achieved

    # Fuel pickup bonus
    fuel_current = info.get("fuel", 0)
    fuel_prev = prev_info.get("fuel", 0)
    # Fuel increases when picked up (minus drain), so check for significant increase
    if fuel_current > fuel_prev + 5:  # Threshold to detect pickup vs just less drain
        shaped_reward += config.reward_fuel_bonus

    # Stage completion bonus
    stage_current = info.get("stage", 1)
    stage_prev = prev_info.get("stage", 1)
    if stage_current > stage_prev:
        stage_multiplier = stage_current
        shaped_reward += config.reward_stage_bonus * stage_multiplier

    # Pothole penalty (detected by speed drop + fuel drop without crash)
    # The game_mode will show crash messages, we detect via fuel/speed changes

    # Crash penalty
    if game_mode == "crashed" and prev_info.get("game_mode") == "playing":
        shaped_reward -= config.reward_crash_penalty

    return shaped_reward


def make_video_writer(path: str, fps: float):
    """Create a video writer for recording evaluation episodes."""
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
            # Fallback to gif if mp4 fails
            if path.endswith(".mp4"):
                path = path.replace(".mp4", ".gif")
            return imageio.get_writer(path, fps=fps)
        except Exception:
            return None


def evaluate(
    policy: nn.Module,
    config: TrainConfig,
    obs_normalizer: Optional[ObservationNormalizer],
    device: str,
    num_episodes: int = 10,
    deterministic: bool = True,
    global_step: int = 0,
    record_video: bool = True,
    video_dir: Optional[str] = None,
) -> Dict[str, float]:
    """Run evaluation episodes and return metrics.

    Args:
        policy: Policy network to evaluate
        config: Training configuration
        obs_normalizer: Observation normalizer (can be None)
        device: Device to run on
        num_episodes: Number of episodes to run
        deterministic: If True, use argmax action; if False, sample
        global_step: Current training step (for video filename)
        record_video: Whether to record video of best episode
        video_dir: Directory to save videos (defaults to rl/sac/runs)

    Returns:
        Dictionary with mean_reward, mean_score, mean_stage, mean_length
    """
    # Determine if we can record video
    render_mode = "rgb_array" if record_video else None

    # Create a separate eval environment
    eval_env = CracerGymEnv(
        render_mode=render_mode,
        obs_mode="state",
        action_mode="discrete",
        fps=config.env_fps,
        seed=42,  # Fixed seed for reproducibility
        max_objects=config.max_objects,
    )

    base_obs_size = eval_env.observation_space.shape[0]
    frame_stacker = FrameStack(config.frame_stack, base_obs_size) if config.frame_stack > 1 else None

    episode_rewards = []
    episode_scores = []
    episode_stages = []
    episode_lengths = []

    # Video recording setup
    if video_dir is None:
        video_dir = os.path.join(ROOT, "rl", "sac", "runs")
    os.makedirs(video_dir, exist_ok=True)

    best_episode_reward = float("-inf")
    best_episode_frames: List[np.ndarray] = []
    record_fps = config.env_fps // 2  # Record at half fps to reduce file size

    policy.eval()

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

        # Capture initial frame
        if record_video and render_mode == "rgb_array":
            frame = eval_env.render()
            if frame is not None:
                episode_frames.append(frame)

        while not done and episode_length < config.max_episode_steps:
            # Normalize observation
            if obs_normalizer:
                obs_normalized = obs_normalizer.normalize(obs)
            else:
                obs_normalized = obs

            # Select action
            with torch.no_grad():
                obs_tensor = torch.tensor(obs_normalized, dtype=torch.float32, device=device).unsqueeze(0)
                if deterministic:
                    # Use argmax for deterministic action
                    logits = policy(obs_tensor)
                    action = logits.argmax(dim=-1).item()
                else:
                    action, _, _ = policy.sample_action(obs_tensor)

            # Step environment
            next_obs, reward, terminated, truncated, info = eval_env.step(action)
            next_obs = np.asarray(next_obs, dtype=np.float32)

            # Shape reward (same as training for fair comparison)
            shaped_reward = shape_reward(reward, info, prev_info, config, episode_state)
            prev_info = info.copy()

            episode_reward += shaped_reward
            episode_length += 1
            max_stage = max(max_stage, info.get("stage", 1))

            done = terminated or truncated

            # Capture frame every 2 steps to reduce video size
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

        # Keep frames from best episode
        if episode_reward > best_episode_reward:
            best_episode_reward = episode_reward
            best_episode_frames = episode_frames

    eval_env.close()
    policy.train()

    # Save video of best episode
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
            video_path = None  # Failed to create writer

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


def train(config: TrainConfig, resume: bool = True) -> None:
    """Main SAC training loop.

    Args:
        config: Training configuration
        resume: If True, resume from latest checkpoint if available
    """
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
        initial_alpha = max(config.alpha, 1e-8)
        log_alpha = torch.tensor(np.log(initial_alpha), requires_grad=True, device=device, dtype=torch.float32)
        alpha_optimizer = optim.Adam([log_alpha], lr=config.learning_rate)
        alpha = float(log_alpha.exp().item())
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

    # Curriculum manager
    curriculum = CurriculumManager(config)

    # Logging
    checkpoint_dir = os.path.join(ROOT, config.checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training state
    global_step = 0
    episode_count = 0

    # Try to load from latest checkpoint
    latest_checkpoint = find_latest_checkpoint(checkpoint_dir) if resume else None
    if latest_checkpoint is not None:
        print(f"\nFound checkpoint: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=False)

        # Load network weights
        policy.load_state_dict(checkpoint["policy"])
        q1.load_state_dict(checkpoint["q1"])
        q2.load_state_dict(checkpoint["q2"])

        # Load target networks if available
        if "q1_target" in checkpoint:
            q1_target.load_state_dict(checkpoint["q1_target"])
        else:
            q1_target.load_state_dict(q1.state_dict())
        if "q2_target" in checkpoint:
            q2_target.load_state_dict(checkpoint["q2_target"])
        else:
            q2_target.load_state_dict(q2.state_dict())

        # Load optimizers if available
        if "policy_optimizer" in checkpoint:
            policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
        if "q1_optimizer" in checkpoint:
            q1_optimizer.load_state_dict(checkpoint["q1_optimizer"])
        if "q2_optimizer" in checkpoint:
            q2_optimizer.load_state_dict(checkpoint["q2_optimizer"])
        if "alpha_optimizer" in checkpoint and alpha_optimizer is not None:
            alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])

        # Load alpha and log_alpha
        if "log_alpha" in checkpoint and log_alpha is not None:
            log_alpha.data.copy_(torch.tensor(checkpoint["log_alpha"], device=device))
            alpha = float(log_alpha.exp().item())
        elif "alpha" in checkpoint:
            alpha = checkpoint["alpha"]

        # Load observation normalizer state
        if "obs_normalizer" in checkpoint and obs_normalizer is not None:
            obs_normalizer.load_state(checkpoint["obs_normalizer"])

        # Load training state
        global_step = checkpoint.get("global_step", 0)
        episode_count = checkpoint.get("episode_count", 0)

        # Load curriculum state
        if "curriculum" in checkpoint:
            curriculum.load_state(checkpoint["curriculum"])

        print(f"Resumed from step {global_step}, episode {episode_count}")
        print(f"  Curriculum: {curriculum.status_str()}")
    elif resume:
        print("\nNo checkpoint found, starting fresh training")
    else:
        print("\nStarting fresh training (--no-resume specified)")

    logger = TrainingLogger(os.path.join(checkpoint_dir, "training_log.csv"))
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
    start_stage = curriculum.sample_stage()
    obs, info = env.reset(seed=initial_seed, options={"start_stage": start_stage})
    obs = np.asarray(obs, dtype=np.float32)
    if frame_stacker:
        obs = frame_stacker.reset(obs)

    current_episode_reward = 0.0
    current_episode_length = 0
    max_stage = start_stage
    prev_info = info.copy()
    episode_state: Dict[str, Any] = {}  # Track episode-level state for reward shaping
    episode_start_stages: List[int] = []  # Track curriculum start stages

    best_mean_reward = float("-inf")
    best_eval_reward = float("-inf")
    evals_without_improvement = 0
    start_time = time.time()

    print(f"\nStarting SAC training for {config.total_timesteps} timesteps...")
    print(f"Buffer size: {config.buffer_size}, Batch size: {config.batch_size}")
    print(f"Auto alpha: {config.auto_alpha}, Initial alpha: {alpha:.4f}")
    if config.curriculum_enabled:
        print(f"Curriculum learning enabled: {curriculum.status_str()}")

    while global_step < config.total_timesteps:
        global_step += 1

        # Update normalizer
        if obs_normalizer:
            obs_normalizer.update(obs)

        # Normalize observation for policy action selection
        obs_for_policy = obs_normalizer.normalize(obs) if obs_normalizer else obs

        # Select action
        if global_step < config.learning_starts:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs_for_policy, dtype=torch.float32, device=device).unsqueeze(0)
                action, _, _ = policy.sample_action(obs_tensor)

        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_obs = np.asarray(next_obs, dtype=np.float32)

        # Shape reward
        shaped_reward = shape_reward(reward, info, prev_info, config, episode_state)
        prev_info = info.copy()

        current_episode_reward += shaped_reward
        current_episode_length += 1
        max_stage = max(max_stage, info.get("stage", 1))
        time_limit_reached = current_episode_length >= config.max_episode_steps
        truncated = truncated or time_limit_reached
        done = terminated or truncated

        # Update observation
        if frame_stacker:
            next_obs_stacked = frame_stacker.push(next_obs)
        else:
            next_obs_stacked = next_obs

        # Store transition
        buffer.add(obs, action, shaped_reward, next_obs_stacked, done)

        obs = next_obs_stacked

        if done:
            # Log episode
            episode_rewards.append(current_episode_reward)
            episode_scores.append(info.get("score", 0))
            episode_stages.append(max_stage)
            episode_lengths.append(current_episode_length)
            episode_start_stages.append(start_stage)
            episode_count += 1

            # Record in curriculum and check graduation
            grad_msg = curriculum.record_episode(current_episode_reward, start_stage, max_stage)
            if grad_msg:
                print(f"\n  {grad_msg}\n")

            # Reset with curriculum sampling
            new_seed = random.randint(0, config.seed_range) if config.randomize_seed else 42
            start_stage = curriculum.sample_stage()
            obs, info = env.reset(seed=new_seed, options={"start_stage": start_stage})
            obs = np.asarray(obs, dtype=np.float32)
            if frame_stacker:
                obs = frame_stacker.reset(obs)

            current_episode_reward = 0.0
            current_episode_length = 0
            max_stage = start_stage
            prev_info = info.copy()
            episode_state = {}  # Reset episode-level state

        # Training
        if global_step >= config.learning_starts and global_step % config.train_freq == 0:
            for _ in range(config.gradient_steps):
                # Sample batch
                b_obs, b_actions, b_rewards, b_next_obs, b_dones = buffer.sample(config.batch_size)
                if obs_normalizer:
                    b_obs = obs_normalizer.normalize_torch(b_obs)
                    b_next_obs = obs_normalizer.normalize_torch(b_next_obs)

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
                if config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(q1.parameters(), config.grad_clip)
                q1_optimizer.step()

                q2_optimizer.zero_grad()
                q2_loss.backward()
                if config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(q2.parameters(), config.grad_clip)
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

                    # Apply alpha floor to prevent entropy collapse
                    if alpha < config.min_alpha:
                        with torch.no_grad():
                            log_alpha.data.copy_(torch.tensor(np.log(config.min_alpha), device=device))
                        alpha = config.min_alpha

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
            recent_start_stages = episode_start_stages[-100:] if len(episode_start_stages) >= 100 else episode_start_stages

            mean_reward = np.mean(recent_rewards)
            mean_score = np.mean(recent_scores)
            mean_stage = np.mean(recent_stages)
            mean_length = np.mean(recent_lengths)
            mean_start_stage = np.mean(recent_start_stages) if recent_start_stages else 1.0

            elapsed = time.time() - start_time
            fps = global_step / elapsed

            print(f"Step {global_step}/{config.total_timesteps} | Episodes: {episode_count} | FPS: {fps:.0f}")
            print(f"  Mean reward: {mean_reward:.1f} | Score: {mean_score:.0f} | "
                  f"Stage: {mean_stage:.1f} | Length: {mean_length:.0f}")
            if config.curriculum_enabled:
                print(f"  Curriculum: {curriculum.status_str()} | mean start stage: {mean_start_stage:.1f}")

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
                               checkpoint_dir, "best.pt", obs_normalizer, alpha,
                               q1_target, q2_target, policy_optimizer, q1_optimizer,
                               q2_optimizer, alpha_optimizer, log_alpha, curriculum)

        # Periodic checkpoint with evaluation
        if global_step % config.save_interval == 0:
            save_checkpoint(policy, q1, q2, config, global_step, episode_count,
                           checkpoint_dir, f"checkpoint_{global_step}.pt", obs_normalizer, alpha,
                           q1_target, q2_target, policy_optimizer, q1_optimizer,
                           q2_optimizer, alpha_optimizer, log_alpha, curriculum)

            # Run evaluation
            if config.eval_episodes > 0:
                print(f"\n  Running evaluation ({config.eval_episodes} episodes)...")
                eval_metrics = evaluate(
                    policy, config, obs_normalizer, device,
                    num_episodes=config.eval_episodes,
                    deterministic=config.eval_deterministic,
                    global_step=global_step,
                    record_video=True,
                )
                print(f"  Eval reward: {eval_metrics['eval_mean_reward']:.1f} ± {eval_metrics['eval_std_reward']:.1f}")
                print(f"  Eval score: {eval_metrics['eval_mean_score']:.0f} | "
                      f"Stage: {eval_metrics['eval_mean_stage']:.1f} | "
                      f"Length: {eval_metrics['eval_mean_length']:.0f}")
                print(f"  Eval range: [{eval_metrics['eval_min_reward']:.0f}, {eval_metrics['eval_max_reward']:.0f}]")
                if eval_metrics.get('eval_video_path'):
                    print(f"  Video saved: {eval_metrics['eval_video_path']}")
                print()

                # Track best eval and early stopping
                if eval_metrics['eval_mean_reward'] > best_eval_reward:
                    best_eval_reward = eval_metrics['eval_mean_reward']
                    evals_without_improvement = 0
                    # Save best eval checkpoint
                    save_checkpoint(policy, q1, q2, config, global_step, episode_count,
                                   checkpoint_dir, "best_eval.pt", obs_normalizer, alpha,
                                   q1_target, q2_target, policy_optimizer, q1_optimizer,
                                   q2_optimizer, alpha_optimizer, log_alpha, curriculum)
                    print(f"  New best eval reward: {best_eval_reward:.1f} - saved best_eval.pt")
                else:
                    evals_without_improvement += 1
                    if config.early_stopping_patience > 0:
                        print(f"  No improvement for {evals_without_improvement}/{config.early_stopping_patience} evals")

                # Early stopping check
                if config.early_stopping_patience > 0 and evals_without_improvement >= config.early_stopping_patience:
                    print(f"\nEarly stopping triggered after {evals_without_improvement} evals without improvement")
                    break

    # Final save
    save_checkpoint(policy, q1, q2, config, global_step, episode_count,
                   checkpoint_dir, "final.pt", obs_normalizer, alpha,
                   q1_target, q2_target, policy_optimizer, q1_optimizer,
                   q2_optimizer, alpha_optimizer, log_alpha, curriculum)

    env.close()
    print(f"\nTraining complete! Total timesteps: {global_step}, Episodes: {episode_count}")
    print(f"Best mean reward: {best_mean_reward:.1f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the latest checkpoint in the checkpoint directory.

    Returns the path to the latest checkpoint file, or None if no checkpoints exist.
    Checkpoints are named like 'checkpoint_100000.pt' and we find the one with highest step.
    """
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

    # Sort by step number and return the latest
    checkpoint_files.sort(key=lambda x: x[0], reverse=True)
    return os.path.join(checkpoint_dir, checkpoint_files[0][1])


def save_checkpoint(policy: nn.Module, q1: nn.Module, q2: nn.Module, config: TrainConfig,
                   global_step: int, episode_count: int, checkpoint_dir: str, filename: str,
                   obs_normalizer: Optional[ObservationNormalizer] = None, alpha: float = 0.2,
                   q1_target: Optional[nn.Module] = None, q2_target: Optional[nn.Module] = None,
                   policy_optimizer: Optional[optim.Optimizer] = None,
                   q1_optimizer: Optional[optim.Optimizer] = None,
                   q2_optimizer: Optional[optim.Optimizer] = None,
                   alpha_optimizer: Optional[optim.Optimizer] = None,
                   log_alpha: Optional[torch.Tensor] = None,
                   curriculum: Optional[CurriculumManager] = None):
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
            "max_objects": config.max_objects,
        },
    }
    if obs_normalizer is not None:
        checkpoint_data["obs_normalizer"] = obs_normalizer.get_state()
    if q1_target is not None:
        checkpoint_data["q1_target"] = q1_target.state_dict()
    if q2_target is not None:
        checkpoint_data["q2_target"] = q2_target.state_dict()
    if policy_optimizer is not None:
        checkpoint_data["policy_optimizer"] = policy_optimizer.state_dict()
    if q1_optimizer is not None:
        checkpoint_data["q1_optimizer"] = q1_optimizer.state_dict()
    if q2_optimizer is not None:
        checkpoint_data["q2_optimizer"] = q2_optimizer.state_dict()
    if alpha_optimizer is not None:
        checkpoint_data["alpha_optimizer"] = alpha_optimizer.state_dict()
    if log_alpha is not None:
        checkpoint_data["log_alpha"] = log_alpha.detach().cpu().numpy()
    if curriculum is not None:
        checkpoint_data["curriculum"] = curriculum.get_state()
    torch.save(checkpoint_data, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAC agent for Cracer Sim")
    parser.add_argument("--config", type=str, default="", help="Path to config YAML file")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--buffer-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", dest="resume", action="store_true", default=True,
                       help="Resume from latest checkpoint (default)")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                       help="Start fresh training, ignore existing checkpoints")
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

    train(config, resume=args.resume)


if __name__ == "__main__":
    main()
