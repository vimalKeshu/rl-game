# Reinforcement Learning Agents

This folder contains RL trainers for the `cracer_sim` Gymnasium wrapper — a Road Fighter-style racing game where the agent must clear as many stages as possible from a cold start.

| Algorithm | Type | Best Result |
|-----------|------|-------------|
| **PPO** | On-policy, policy gradient | s1_best=8.6 (Exp27), actively being developed |
| **SAC** | Off-policy, actor-critic | s1_best=2.20 (ceiling hit, moved to PPO) |
| **DQN** | Off-policy, value-based | Baseline only |

**Primary metric**: `stage1_mean_stage` — average stage reached when starting from stage 1 on unseen seeds, deterministic policy.

---

## Install

```bash
pip install -r game/requirements.txt
pip install -r rl/requirements.txt
```

Then install PyTorch for your platform: https://pytorch.org/get-started/locally/

---

## PPO (Active)

PPO is the primary algorithm. It has gone through 27+ experiments progressively improving the agent from avg 2.1 → 8.6. See `rl/ppo/experiments.md` for the full experiment log.

### Quick start — run inference on a trained checkpoint

Point at any experiment's `final.pt` or `best_eval.pt`:

```bash
# Watch the agent play live (Exp27 — best current model)
python rl/ppo/infer_ppo.py \
  --checkpoint rl/ppo/exp27_checkpoints/final.pt \
  --render-live

# Record a video
python rl/ppo/infer_ppo.py \
  --checkpoint rl/ppo/exp27_checkpoints/final.pt \
  --output my_run.mp4 \
  --episodes 3

# Run 10 episodes and print stats
python rl/ppo/infer_ppo.py \
  --checkpoint rl/ppo/exp27_checkpoints/final.pt \
  --episodes 10 \
  --no-render

# Run forever (infinite mode) — tracks cumulative stats
python rl/ppo/infer_ppo.py \
  --checkpoint rl/ppo/exp27_checkpoints/final.pt \
  --infinite \
  --render-live
```

**Checkpoint choice:**
- `best_eval.pt` — highest eval score during training (best play quality, use for demos)
- `final.pt` — last training step (may be slightly weaker than best_eval)

The script reads `max_objects`, `hidden_sizes`, `frame_stack` etc. directly from the checkpoint — no manual config needed.

### Available experiments

Each experiment folder contains `final.pt`, `eval_log.csv`, `training_log.csv`, and a config YAML:

| Folder | Key feature | s1_best |
|--------|-------------|---------|
| `exp18_checkpoints/` | Fresh training from scratch | 4.2 |
| `exp19_checkpoints/` | Stage mix expanded to 1-6 | 7.4 |
| `exp20_checkpoints/` | Stage mix expanded to 1-10 | 10.1 |
| `exp21_checkpoints/` | Fuel exhaustion penalty | 10.4 |
| `exp22_checkpoints/` | Stage-scaled bonuses | 11.3 |
| `exp23_checkpoints/` | Stage mix expanded to 1-15 | 11.7 |
| `exp24_checkpoints/` | Stage mix expanded to 1-20 | 11.8 |
| `exp26_checkpoints/` | max_objects=30 + full-speed incentive (fresh) | 7.0 |
| `exp27_checkpoints/` | Stage mix 1-6 on max_objects=30 chain | **8.6** |

### Train from scratch

```bash
# Edit the config then run
python rl/ppo/train_ppo.py --config rl/ppo/config.yaml --no-resume
```

### Warm-start from an existing checkpoint

```bash
# Load actor weights from a prior experiment, reset critic/optimizer
python rl/ppo/train_ppo.py \
  --config rl/ppo/config.yaml \
  --warm-start-actor rl/ppo/exp27_checkpoints/final.pt \
  --no-resume
```

### Resume an interrupted run

```bash
# Auto-resumes from latest checkpoint_N.pt in checkpoint_dir
python rl/ppo/train_ppo.py --config rl/ppo/config.yaml --resume
```

### Override config values from CLI

```bash
python rl/ppo/train_ppo.py \
  --config rl/ppo/config.yaml \
  --total-timesteps 5000000 \
  --learning-rate 0.0001
```

### Key config parameters explained

```yaml
total_timesteps: 20000000     # Total environment steps
max_episode_steps: 100000     # Never cut off a live agent (set high)
max_objects: 30               # Objects visible in observation (30 = full awareness)
frame_stack: 4                # Stacked frames for temporal context

# Stage mix — probability of starting each episode at that stage
# Heavy stage-1 for fresh training; expand as agent improves
stage_mix:
  1: 0.70   # cold-start anchor
  2: 0.20
  3: 0.10

# Reward signals
reward_crash_penalty: 600.0              # flat crash cost
reward_crash_penalty_stage_scale: 0.4   # scales crash cost by stage (stage 10 = 2760)
reward_fuel_exhaustion_penalty: 600.0   # dying from fuel = same as a crash
reward_low_fuel_penalty: 2.0            # per-step urgency when fuel < 30
reward_bonus_stage_scale: 0.3           # scales survival/fuel/milestone bonuses by stage
reward_speed_scale: 0.5                 # incentivize holding max speed

# Eval settings
eval_start_stages: [1, 5, 10]  # evaluate cold-start and mid-game
eval_episodes: 10              # per start stage
eval_max_episode_steps: 100000 # never cut off eval episodes
```

### Starting a new experiment (recommended workflow)

1. Copy the latest config as your base:
   ```bash
   cp rl/ppo/exp27_checkpoints/config_exp27.yaml rl/ppo/config.yaml
   ```
2. Edit `config.yaml` with your changes
3. Run with warm-start from the latest best checkpoint:
   ```bash
   python rl/ppo/train_ppo.py \
     --config rl/ppo/config.yaml \
     --warm-start-actor rl/ppo/checkpoints/best_eval.pt \
     --no-resume
   ```
4. Monitor progress in `rl/ppo/checkpoints/eval_log.csv`
5. After completion, archive:
   ```bash
   mkdir rl/ppo/expN_checkpoints
   mv rl/ppo/checkpoints/final.pt \
      rl/ppo/checkpoints/eval_log.csv \
      rl/ppo/checkpoints/training_log.csv \
      rl/ppo/expN_checkpoints/
   cp rl/ppo/config.yaml rl/ppo/expN_checkpoints/config_expN.yaml
   ```

### Experiment log

See `rl/ppo/experiments.md` for the full history of all 27 experiments — hypothesis, config, results, and key learnings for each.

---

## SAC (Reference)

SAC was the original algorithm. It hit a ceiling of ~2.20 avg stage after 5 experiments due to off-policy replay buffer incompatibility with sequential skill-building. Moved to PPO.

### Train

```bash
python rl/sac/train_sac.py --config rl/sac/config.yaml
```

Auto-resumes from latest checkpoint. Use `--no-resume` to start fresh.

### Inference

```bash
python rl/sac/infer_sac.py --checkpoint rl/sac/checkpoints/best.pt --episodes 3
```

---

## DQN (Baseline)

### Train

```bash
python rl/dqn/train_dqn.py --config rl/dqn/config.yaml
```

### Inference

```bash
python rl/dqn/infer_dqn.py --checkpoint rl/dqn/checkpoints/best.pt --episodes 1
```

---

## Directory Structure

```
rl/
├── README.md
├── requirements.txt
├── dqn/
│   ├── config.yaml
│   ├── train_dqn.py
│   └── infer_dqn.py
├── ppo/
│   ├── config.yaml               ← current experiment config
│   ├── train_ppo.py
│   ├── infer_ppo.py
│   ├── experiments.md            ← full experiment log (27 experiments)
│   ├── merge_policies.py
│   ├── checkpoints/              ← active training output
│   │   └── best_eval.pt          ← use this for next warm-start
│   ├── exp18_checkpoints/        ← archived experiments
│   ├── exp19_checkpoints/
│   ├── ...
│   └── exp27_checkpoints/
└── sac/
    ├── config.yaml
    ├── train_sac.py
    └── infer_sac.py
```
