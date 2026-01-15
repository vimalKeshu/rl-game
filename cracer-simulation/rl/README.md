# DQN Training

This folder contains a simple DQN trainer for the `cracer_sim` Gymnasium wrapper.

## Install

You need PyTorch in addition to the root requirements.

```
pip install -r game/requirements.txt
pip install -r rl/requirements.txt
```

Then install PyTorch following the official instructions for your platform:

https://pytorch.org/get-started/locally/

## Train

Edit `rl/config.yaml`, then run:

```
python rl/train_dqn.py
```

Use `--config path/to/config.yaml` to load a different config file.

Checkpoints are saved to `rl/checkpoints/`.

## Resume

Set `resume` in `rl/config.yaml`:

```
resume: "rl/checkpoints/episode_500.pt"
```

## Debug Render

Set `render: true` and lower `total_episodes` in `rl/config.yaml`.

## Inference + Video

Record a run to video using a checkpoint:

```
python rl/infer_dqn.py --checkpoint rl/checkpoints/best.pt --episodes 1
```

The output is saved under `rl/runs/` by default. Use `--output` to pick a filename
and `--frame-skip` to reduce file size. For mp4 output, `imageio-ffmpeg` is required.

## Notes

- The DQN trainer uses `obs_mode="state"` and `action_mode="discrete"`.
- This trainer follows the PyTorch DQN tutorial (AdamW + Huber loss + soft target updates).
- Tune exploration with `eps_start`, `eps_end`, and `eps_decay` in `rl/config.yaml`.
- Reward shaping is configurable via `reward_speed_scale`, `reward_fuel_bonus`, `reward_crash_penalty`,
  `reward_pothole_penalty`, `reward_bump_penalty`, `low_fuel_penalty_scale`, `low_fuel_threshold`,
  `fuel_direction_scale`, and `terminate_on_crash`.
