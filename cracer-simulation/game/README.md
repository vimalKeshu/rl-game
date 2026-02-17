# Cracer Simulation (Python)

A Python implementation inspired by the arcade game [Road Fighter](https://en.wikipedia.org/wiki/Road_Fighter). This version extends the classic gameplay with added obstacles (potholes and speed bumps), modernized spawn logic, and a deterministic simulation loop suitable for RL.

## Requirements

- Python 3.9+
- `pygame` (rendering)
- `numpy`
- `gymnasium`

Install from the repo root:

```
pip install -r game/requirements.txt
```

## Play (Human)

From the repo root:

```
python game/run_human.py
```

### Controls

- Left/Right or A/D: steer
- Up or W: accelerate
- Down or S: brake
- Space or R: restart after game over

## Game Rules

- The road has 3 lanes with traffic vehicles spawning ahead.
- Fuel drains over time. If fuel reaches 0, the game ends.
- Collisions with traffic cause a crash: you lose a life, lose fuel, and slow down.
- Potholes and speed bumps also reduce fuel and speed (potholes are harsher).
- Fuel pickups increase fuel and add a score bonus.
- Speed limit zones and slopes adjust the natural cruising speed.
- Each stage has a distance target. Clearing a stage grants bonus score, refuels, and increases difficulty.
- Game ends when lives reach 0 or fuel is empty.

### Scoring

- Score increases with speed over time.
- Fuel pickups and stage clears add bonus points.

## RL API

### Core Environment

`CracerEnv` is the low-level simulation. `CracerGymEnv` wraps it as a standard Gymnasium environment.

```python
from cracer_sim import CracerGymEnv

env = CracerGymEnv(render_mode=None, obs_mode="state", action_mode="discrete")
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(0)
```

Or use the core env directly:

```python
from cracer_sim import CracerEnv

env = CracerEnv(render_mode=None, obs_mode="state", action_mode="discrete")
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(0)
```

### Constructor Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `render_mode` | `None` | `None`, `"human"`, or `"rgb_array"` |
| `obs_mode` | `"state"` | `"state"` (flat vector) or `"pixels"` / `"rgb_array"` (image) |
| `action_mode` | `"discrete"` | `"discrete"`, `"continuous"`, or `"buttons"` |
| `width` | `800` | Window / frame width |
| `height` | `600` | Window / frame height |
| `fps` | `60` | Simulation tick rate |
| `seed` | `None` | Random seed for reproducibility |
| `max_objects` | `6` | Max nearby objects included in state observations |

### Action Modes

**Discrete** (9 actions):

| Action | Meaning |
|--------|---------|
| 0 | No-op |
| 1 | Left |
| 2 | Right |
| 3 | Accelerate |
| 4 | Brake |
| 5 | Left + Accelerate |
| 6 | Right + Accelerate |
| 7 | Left + Brake |
| 8 | Right + Brake |

**Continuous**: `Box([-1, -1], [1, 1])` — `[steer, throttle]`.

**Buttons**: `MultiBinary(4)` — `[left, right, accelerate, brake]`.

### Observation Modes

- **`state`**: Flat float32 vector (normalized to [-1, 1]). Includes player state (speed, fuel, position, etc.) plus per-object features (relative position and one-hot type) for the nearest `max_objects` entities. Vector size: `11 + max_objects * (2 + 6)`.
- **`pixels` / `rgb_array`**: RGB uint8 image of shape `(height, width, 3)`.

### Reset Options

```python
obs, info = env.reset(seed=42, options={"start_stage": 5})
```

- `seed`: Set the random seed.
- `start_stage`: Start at a later stage (1-based). Higher stages simulate mid-game conditions with appropriate speed, difficulty, and reduced fuel.

### Info Dict

Each `step()` and `reset()` returns an info dict with:

| Key | Type | Description |
|-----|------|-------------|
| `score` | float | Current score |
| `speed` | float | Current player speed |
| `speed_limit` | float | Current speed limit |
| `fuel` | float | Remaining fuel (0–1) |
| `lives` | int | Remaining lives |
| `stage` | int | Current stage number |
| `distance_remaining` | float | Distance left in current stage |
| `game_mode` | str | `"playing"`, `"stage_clear"`, or `"game_over"` |
| `message` | str or None | Current on-screen message |

## RL Training

See `rl/README.md` for DQN, PPO, and SAC training and inference.
