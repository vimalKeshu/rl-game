# Reinforcement Learning Agents

This folder contains three RL trainers for the `cracer_sim` Gymnasium wrapper:

| Algorithm | Type | Key Features |
|-----------|------|--------------|
| **DQN** | Off-policy, value-based | Dueling + Double DQN, Prioritized Experience Replay |
| **PPO** | On-policy, policy gradient | LR annealing, observation normalization, curriculum learning |
| **SAC** | Off-policy, actor-critic | Automatic entropy tuning, curriculum learning, adaptive eval |

All three use `obs_mode="state"` and `action_mode="discrete"`.

## Install

You need PyTorch in addition to the root requirements.

```
pip install -r game/requirements.txt
pip install -r rl/requirements.txt
```

Then install PyTorch following the official instructions for your platform:

https://pytorch.org/get-started/locally/

---

## DQN

### Train

Edit `rl/dqn/config.yaml`, then run:

```
python rl/dqn/train_dqn.py --config rl/dqn/config.yaml
```

Checkpoints are saved to `rl/dqn/checkpoints/`.

### Resume

Set `resume` in `rl/dqn/config.yaml`:

```yaml
resume: "rl/dqn/checkpoints/episode_500.pt"
```

### Debug Render

Set `render: true` and lower `total_episodes` in `rl/dqn/config.yaml`.

### Inference + Video

```
python rl/dqn/infer_dqn.py --checkpoint rl/dqn/checkpoints/best.pt --episodes 1
```

Options: `--output`, `--frame-skip`, `--render-live`, `--random-seeds`, `--seed`, `--no-render`.

### Notes

- Uses Dueling + Double DQN with optional Prioritized Experience Replay.
- Tune exploration with `eps_start`, `eps_end`, and `eps_decay`.
- Reward shaping is configurable via `reward_*` keys in the config.
- Set `randomize_seed: true` and `seed_range` for generalization across map seeds.

---

## PPO

### Train

Edit `rl/ppo/config.yaml`, then run:

```
python rl/ppo/train_ppo.py --config rl/ppo/config.yaml
```

Checkpoints are saved to `rl/ppo/checkpoints/`.

Override config values from the command line:

```
python rl/ppo/train_ppo.py --config rl/ppo/config.yaml --total-timesteps 500000 --learning-rate 0.001
```

### Evaluation

Run evaluation on a trained checkpoint:

```
python rl/ppo/train_ppo.py --eval --eval-checkpoint rl/ppo/checkpoints/best.pt --eval-episodes 5 --eval-start-stages 8-10,12
```

### Inference + Video

```
python rl/ppo/infer_ppo.py --checkpoint rl/ppo/checkpoints/best.pt --episodes 1
```

Options: `--output`, `--frame-skip`, `--render-live`, `--random-seeds`, `--seed`, `--stochastic`, `--no-render`.

### Notes

- Supports learning rate annealing (`anneal_lr`) and entropy coefficient annealing (`anneal_entropy`).
- Observation normalization via `normalize_obs`.
- Frame stacking controlled by `frame_stack` (default 4).
- Curriculum learning gradually exposes the agent to harder stages (`curriculum_enabled`).
- Periodic evaluation during training (`eval_interval_updates`).

---

## SAC

### Train

Edit `rl/sac/config.yaml`, then run:

```
python rl/sac/train_sac.py --config rl/sac/config.yaml
```

Checkpoints are saved to `rl/sac/checkpoints/`.

Training automatically resumes from the latest checkpoint if one exists. Use `--no-resume` to start fresh.

Override config values from the command line:

```
python rl/sac/train_sac.py --config rl/sac/config.yaml --total-timesteps 1000000 --batch-size 128
```

### Inference + Video

```
python rl/sac/infer_sac.py --checkpoint rl/sac/checkpoints/best.pt --episodes 1
```

Options: `--output`, `--frame-skip`, `--render-live`, `--random-seeds`, `--seed`, `--stochastic`, `--no-render`.

### Notes

- Automatic entropy tuning (`auto_alpha: true`) with configurable `target_entropy_ratio` and `min_alpha`.
- Adaptive curriculum learning with graduation windows and per-stage episode minimums.
- Frame stacking controlled by `frame_stack` (default 4).
- Observation normalization via `normalize_obs`.
- Early stopping support (`early_stopping_patience`).
- Evaluation runs after each checkpoint save (`eval_episodes`).

---

## Directory Structure

```
rl/
├── README.md
├── requirements.txt
├── test.py
├── dqn/
│   ├── config.yaml
│   ├── train_dqn.py
│   ├── infer_dqn.py
│   ├── plot_training.py
│   └── checkpoints/
├── ppo/
│   ├── config.yaml
│   ├── train_ppo.py
│   ├── infer_ppo.py
│   └── checkpoints/
└── sac/
    ├── config.yaml
    ├── train_sac.py
    ├── infer_sac.py
    └── checkpoints/
```
