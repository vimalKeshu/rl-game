from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from collections import deque, namedtuple
from dataclasses import dataclass, fields
from typing import Deque, Optional, Tuple, Dict, Any
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
    total_episodes: int = 1_000
    memory_size: int = 10_000
    batch_size: int = 128
    gamma: float = 0.99
    learning_rate: float = 1e-4
    eps_start: float = 0.9
    eps_end: float = 0.05
    eps_decay: float = 1_000.0
    tau: float = 0.005
    hidden_sizes: Tuple[int, ...] = (256, 256)
    reward_speed_scale: float = 0.1
    reward_fuel_bonus: float = 500.0
    reward_crash_penalty: float = 300.0
    reward_pothole_penalty: float = 100.0
    reward_bump_penalty: float = 50.0
    low_fuel_penalty_scale: float = 0.0
    low_fuel_threshold: float = 0.3
    fuel_direction_scale: float = 0.0
    terminate_on_crash: bool = True
    save_every: int = 100
    save_dir: str = "rl/checkpoints"
    log_interval: int = 10
    seed: int = 0
    device: str = "auto"
    resume: str = ""
    render: bool = False


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


class ReplayMemory:
    def __init__(self, capacity: int) -> None:
        self.memory = deque(maxlen=capacity)

    def push(self, *args) -> None:
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


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
    policy_net: QNetwork,
    target_net: QNetwork,
    optimizer: torch.optim.Optimizer,
    episode: int,
    steps_done: int,
    epsilon: float,
    model_config: Optional[dict] = None,
) -> None:
    payload = {
        "q_net": policy_net.state_dict(),
        "target_net": target_net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "episode": episode,
        "steps_done": steps_done,
        "epsilon": epsilon,
    }
    if model_config:
        payload["model_config"] = model_config
    torch.save(payload, path)


def load_checkpoint(path: str, device: str):
    return torch.load(path, map_location=device)


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
        raise FileNotFoundError(f"Config file not found: {path}")

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
    print(f"device: {device}")
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
    }

    policy_net = QNetwork(obs_size, num_actions, config.hidden_sizes).to(device)
    target_net = QNetwork(obs_size, num_actions, config.hidden_sizes).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.AdamW(policy_net.parameters(), lr=config.learning_rate, amsgrad=True)
    memory = ReplayMemory(config.memory_size)

    start_episode = 0
    steps_done = 0
    if config.resume:
        checkpoint = load_checkpoint(config.resume, device)
        policy_net.load_state_dict(checkpoint["q_net"])
        target_net.load_state_dict(checkpoint["target_net"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_episode = int(checkpoint.get("episode", 0))
        steps_done = int(checkpoint.get("steps_done", 0))

    os.makedirs(config.save_dir, exist_ok=True)

    reward_window: Deque[float] = deque(maxlen=100)
    best_mean = -float("inf")
    last_log_time = time.time()
    last_log_step = steps_done

    def epsilon_by_steps(steps: int) -> float:
        return config.eps_end + (config.eps_start - config.eps_end) * math.exp(-1.0 * steps / config.eps_decay)

    def select_action(state: torch.Tensor):
        nonlocal steps_done
        eps_threshold = epsilon_by_steps(steps_done)
        steps_done += 1
        if random.random() > eps_threshold:
            with torch.no_grad():
                action = policy_net(state).max(1).indices.view(1, 1)
        else:
            action = torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
        return action, eps_threshold

    def optimize_model() -> Optional[float]:
        if len(memory) < config.batch_size:
            return None
        transitions = memory.sample(config.batch_size)
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

        state_action_values = policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(config.batch_size, device=device)
        if non_final_next_states is not None:
            with torch.no_grad():
                next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

        expected_state_action_values = reward_batch + (config.gamma * next_state_values)

        loss = F.smooth_l1_loss(state_action_values.squeeze(1), expected_state_action_values)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()
        return float(loss.item())

    eps_threshold = epsilon_by_steps(steps_done)
    for episode_idx in range(start_episode, config.total_episodes):
        obs, info = env.reset()
        prev_fuel = float(info.get("fuel", 0.0))
        prev_lives = int(info.get("lives", 0))
        prev_message = info.get("message") or ""
        prev_fuel_dist = nearest_fuel_distance(obs, max_objects)
        fuel_dir_delta_sum = 0.0
        fuel_dir_delta_count = 0
        fuel_events = 0
        crash_events = 0
        pothole_events = 0
        bump_events = 0
        state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        episode_reward = 0.0

        for t in count():
            action, eps_threshold = select_action(state)
            next_obs, reward, terminated, truncated, info = env.step(int(action.item()))
            
            # Reward shaping to emphasize pickups and hazards over dense speed reward.
            message = info.get("message") or ""
            speed = float(info.get("speed", 0.0))
            speed_reward = speed * getattr(env.env, "dt", 1.0 / 60.0) * 0.6
            shaped_reward = reward - speed_reward + speed_reward * config.reward_speed_scale
            fuel_now = float(info.get("fuel", prev_fuel))
            lives_now = int(info.get("lives", prev_lives))
            fuel_pickup = (fuel_now - prev_fuel) > 1.0 and not message.startswith("STAGE")
            crash_event = lives_now < prev_lives
            pothole_event = message == "POTHOLE!" and prev_message != "POTHOLE!"
            bump_event = message == "BUMP!" and prev_message != "BUMP!"

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

            if config.low_fuel_penalty_scale > 0.0 and config.low_fuel_threshold > 0.0:
                fuel_ratio = max(0.0, min(1.0, fuel_now / max_fuel))
                if fuel_ratio < config.low_fuel_threshold:
                    urgency = (config.low_fuel_threshold - fuel_ratio) / config.low_fuel_threshold
                    shaped_reward -= config.low_fuel_penalty_scale * urgency

            next_fuel_dist = nearest_fuel_distance(next_obs, max_objects)
            if prev_fuel_dist is not None and next_fuel_dist is not None:
                fuel_dir_delta = prev_fuel_dist - next_fuel_dist
                fuel_dir_delta_sum += fuel_dir_delta
                fuel_dir_delta_count += 1
                if config.fuel_direction_scale != 0.0:
                    shaped_reward += config.fuel_direction_scale * fuel_dir_delta
            prev_fuel_dist = next_fuel_dist

            reward = float(shaped_reward)
            episode_reward += reward

            reward_tensor = torch.tensor([reward], device=device)
            done = terminated or truncated
            if not done:
                next_state = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
            else:
                next_state = None

            memory.push(state, action, next_state, reward_tensor)
            state = next_state if next_state is not None else torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
            prev_fuel = fuel_now
            prev_lives = lives_now
            prev_message = message

            optimize_model()

            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = (
                    policy_net_state_dict[key] * config.tau + target_net_state_dict[key] * (1.0 - config.tau)
                )
            target_net.load_state_dict(target_net_state_dict)

            if config.render:
                env.render()
                pump_pygame_events()

            if done:
                break

        reward_window.append(episode_reward)
        mean_reward = sum(reward_window) / max(1, len(reward_window))
        fuel_dir_avg = fuel_dir_delta_sum / fuel_dir_delta_count if fuel_dir_delta_count else None

        if (episode_idx + 1) % config.log_interval == 0:
            now = time.time()
            step_delta = steps_done - last_log_step
            steps_per_sec = step_delta / max(1e-6, now - last_log_time)
            last_log_time = now
            last_log_step = steps_done
            fuel_dir_text = f"{fuel_dir_avg:.4f}" if fuel_dir_avg is not None else "n/a"
            print(
                f"episodes={episode_idx + 1} steps={steps_done} "
                f"mean_reward_100={mean_reward:.1f} "
                f"epsilon={eps_threshold:.2f} "
                f"steps_per_sec~{steps_per_sec:.1f} "
                f"fuel_dir_avg={fuel_dir_text} "
                f"events[fuel={fuel_events} crash={crash_events} pothole={pothole_events} bump={bump_events}]"
            )

        if mean_reward > best_mean:
            best_mean = mean_reward
            best_path = os.path.join(config.save_dir, "best.pt")
            save_checkpoint(best_path, policy_net, target_net, optimizer, episode_idx + 1, steps_done, eps_threshold, model_config)

        if config.save_every > 0 and (episode_idx + 1) % config.save_every == 0:
            checkpoint_path = os.path.join(config.save_dir, f"episode_{episode_idx + 1}.pt")
            save_checkpoint(
                checkpoint_path,
                policy_net,
                target_net,
                optimizer,
                episode_idx + 1,
                steps_done,
                eps_threshold,
                model_config,
            )

    final_path = os.path.join(config.save_dir, "final.pt")
    save_checkpoint(final_path, policy_net, target_net, optimizer, config.total_episodes, steps_done, eps_threshold, model_config)
    env.close()


if __name__ == "__main__":
    main()
