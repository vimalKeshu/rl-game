#!/usr/bin/env python3
"""
Plot training progress from a training log CSV file.

Usage:
    python rl/plot_training.py                           # Use default log file
    python rl/plot_training.py --log rl/checkpoints/training_log.csv
    python rl/plot_training.py --log training_log.csv --output my_plot.png
    python rl/plot_training.py --show                    # Display interactively
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot DQN training progress")
    default_log = os.path.join(ROOT, "rl", "dqn", "checkpoints", "training_log.csv")
    parser.add_argument("--log", type=str, default=default_log, help="Path to training_log.csv")
    parser.add_argument("--output", type=str, default="", help="Output image path (default: same dir as log)")
    parser.add_argument("--show", action="store_true", help="Show plot interactively")
    parser.add_argument("--window", type=int, default=100, help="Moving average window size")
    return parser.parse_args()


def load_log(path: str) -> Dict[str, List[float]]:
    """Load training log CSV file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Log file not found: {path}")

    data: Dict[str, List[float]] = {}

    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                if key not in data:
                    data[key] = []
                try:
                    data[key].append(float(value) if value else 0.0)
                except ValueError:
                    data[key].append(0.0)

    return data


def plot_training(data: Dict[str, List[float]], output_path: Optional[str] = None,
                  show: bool = False, window: int = 100) -> None:
    """Generate training progress plots."""
    try:
        import matplotlib
        if not show:
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib is required. Install with: pip install matplotlib")
        return

    episodes = data.get("episode", [])
    if len(episodes) < 2:
        print("Not enough data to plot")
        return

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle("DQN Training Progress", fontsize=14, fontweight='bold')

    def moving_average(values: List[float], w: int) -> np.ndarray:
        if len(values) < w:
            return np.array(values)
        return np.convolve(values, np.ones(w)/w, mode='valid')

    # Plot 1: Episode Reward
    ax = axes[0, 0]
    rewards = data.get("reward", [])
    if rewards:
        ax.plot(episodes, rewards, alpha=0.3, color='blue', label='Episode Reward')
        if len(rewards) >= window:
            ma = moving_average(rewards, window)
            ax.plot(episodes[window-1:], ma, color='blue', linewidth=2, label=f'Mean ({window} ep)')
    mean_rewards = data.get("mean_reward_100", [])
    if mean_rewards and any(r != 0 for r in mean_rewards):
        ax.plot(episodes, mean_rewards, color='darkblue', linewidth=2, linestyle='--', label='Running Mean')
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Episode Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Game Score
    ax = axes[0, 1]
    scores = data.get("score", [])
    if scores:
        ax.plot(episodes, scores, alpha=0.3, color='green', label='Episode Score')
        if len(scores) >= window:
            ma = moving_average(scores, window)
            ax.plot(episodes[window-1:], ma, color='green', linewidth=2, label=f'Mean ({window} ep)')
    mean_scores = data.get("mean_score_100", [])
    if mean_scores and any(s != 0 for s in mean_scores):
        ax.plot(episodes, mean_scores, color='darkgreen', linewidth=2, linestyle='--', label='Running Mean')
    ax.set_xlabel("Episode")
    ax.set_ylabel("Score")
    ax.set_title("Game Score")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Max Stage Reached
    ax = axes[1, 0]
    stages = data.get("max_stage", [])
    if stages:
        ax.plot(episodes, stages, alpha=0.5, color='purple', marker='.', markersize=2, linestyle='none', label='Max Stage')
        if len(stages) >= window:
            ma = moving_average(stages, window)
            ax.plot(episodes[window-1:], ma, color='purple', linewidth=2, label=f'Mean ({window} ep)')
    mean_stages = data.get("mean_stage_100", [])
    if mean_stages and any(s != 0 for s in mean_stages):
        ax.plot(episodes, mean_stages, color='darkviolet', linewidth=2, linestyle='--', label='Running Mean')
    ax.set_xlabel("Episode")
    ax.set_ylabel("Stage")
    ax.set_title("Max Stage Reached")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Episode Length
    ax = axes[1, 1]
    lengths = data.get("episode_length", [])
    if lengths:
        ax.plot(episodes, lengths, alpha=0.3, color='orange', label='Episode Length')
        if len(lengths) >= window:
            ma = moving_average(lengths, window)
            ax.plot(episodes[window-1:], ma, color='orange', linewidth=2, label=f'Mean ({window} ep)')
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.set_title("Episode Length (Survival Time)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Epsilon and Learning Rate
    ax = axes[2, 0]
    epsilon = data.get("epsilon", [])
    if epsilon:
        ax.plot(episodes, epsilon, color='red', linewidth=1.5, label='Epsilon')
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon", color='red')
    ax.tick_params(axis='y', labelcolor='red')

    ax2 = ax.twinx()
    lr = data.get("learning_rate", [])
    if lr:
        ax2.plot(episodes, lr, color='cyan', linewidth=1.5, label='Learning Rate')
    ax2.set_ylabel("Learning Rate", color='cyan')
    ax2.tick_params(axis='y', labelcolor='cyan')
    ax.set_title("Exploration (Epsilon) & Learning Rate")
    ax.grid(True, alpha=0.3)

    # Plot 6: Loss and Action Entropy
    ax = axes[2, 1]
    losses = data.get("loss", [])
    valid_losses = [(e, l) for e, l in zip(episodes, losses) if l and l > 0]
    if valid_losses:
        loss_eps, loss_vals = zip(*valid_losses)
        ax.plot(loss_eps, loss_vals, alpha=0.5, color='brown', label='Loss')
        if len(loss_vals) >= window:
            ma = moving_average(list(loss_vals), window)
            ax.plot(loss_eps[window-1:], ma, color='brown', linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Loss", color='brown')
    ax.tick_params(axis='y', labelcolor='brown')
    ax.set_title("Training Loss")
    ax.grid(True, alpha=0.3)

    # Add action entropy on secondary axis
    ax2 = ax.twinx()
    entropy = data.get("action_entropy", [])
    if entropy and any(e != 0 for e in entropy):
        ax2.plot(episodes, entropy, color='teal', alpha=0.7, linewidth=1, label='Action Entropy')
        ax2.set_ylabel("Action Entropy", color='teal')
        ax2.tick_params(axis='y', labelcolor='teal')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def print_summary(data: Dict[str, List[float]]) -> None:
    """Print training summary statistics."""
    episodes = data.get("episode", [])
    if not episodes:
        return

    n = len(episodes)
    last_100 = max(0, n - 100)

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Total episodes: {n}")
    print(f"Total steps: {int(data.get('steps', [0])[-1]) if data.get('steps') else 0}")

    rewards = data.get("reward", [])
    if rewards:
        print(f"\nReward:")
        print(f"  Overall mean: {sum(rewards)/len(rewards):.1f}")
        print(f"  Last 100 mean: {sum(rewards[last_100:])/max(1, len(rewards[last_100:])):.1f}")
        print(f"  Max: {max(rewards):.1f}")

    scores = data.get("score", [])
    if scores:
        print(f"\nScore:")
        print(f"  Overall mean: {sum(scores)/len(scores):.0f}")
        print(f"  Last 100 mean: {sum(scores[last_100:])/max(1, len(scores[last_100:])):.0f}")
        print(f"  Max: {max(scores):.0f}")

    stages = data.get("max_stage", [])
    if stages:
        print(f"\nStage:")
        print(f"  Overall mean: {sum(stages)/len(stages):.1f}")
        print(f"  Last 100 mean: {sum(stages[last_100:])/max(1, len(stages[last_100:])):.1f}")
        print(f"  Max reached: {int(max(stages))}")

    lengths = data.get("episode_length", [])
    if lengths:
        print(f"\nEpisode Length:")
        print(f"  Overall mean: {sum(lengths)/len(lengths):.0f}")
        print(f"  Last 100 mean: {sum(lengths[last_100:])/max(1, len(lengths[last_100:])):.0f}")
        print(f"  Max: {int(max(lengths))}")

    fuel = data.get("fuel_pickups", [])
    crashes = data.get("crashes", [])
    if fuel and crashes:
        print(f"\nEvents (last 100 episodes):")
        print(f"  Fuel pickups: {sum(fuel[last_100:]):.0f} total, {sum(fuel[last_100:])/max(1, len(fuel[last_100:])):.1f} per ep")
        print(f"  Crashes: {sum(crashes[last_100:]):.0f} total, {sum(crashes[last_100:])/max(1, len(crashes[last_100:])):.1f} per ep")

    print("=" * 60)


def main() -> None:
    args = parse_args()

    log_path = args.log
    if not os.path.isabs(log_path):
        log_path = os.path.join(ROOT, log_path)

    print(f"Loading log: {log_path}")
    data = load_log(log_path)

    print_summary(data)

    output_path = args.output
    if not output_path:
        output_path = os.path.join(os.path.dirname(log_path), "training_progress.png")
    elif not os.path.isabs(output_path):
        output_path = os.path.join(ROOT, output_path)

    plot_training(data, output_path=output_path if not args.show else None,
                  show=args.show, window=args.window)


if __name__ == "__main__":
    main()
