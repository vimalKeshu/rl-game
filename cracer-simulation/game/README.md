# Cracer Simulation (Python)

A Python implementation inspired by the arcade game Road Fighter: https://en.wikipedia.org/wiki/Road_Fighter. This version extends the classic gameplay with added obstacles (potholes and speed bumps), modernized spawn logic, and a deterministic simulation loop suitable for RL.

## Requirements

- Python 3.9+
- `pygame` (required for rendering)
- `numpy` (optional, needed for `rgb_array` / pixel observations)
- `gymnasium` (optional, only if you use the Gym wrapper)

Install from the repo root:

```
pip install -r game/requirements.txt
```

Optional RL extras:

```
pip install -r rl/requirements.txt
```

## Play (Human)

From the repo root:

```
python -m cracer_sim.run_human
```

Or from `game/`:

```
python run_human.py
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

## RL API (Quick Peek)

```python
from cracer_sim import CracerEnv

env = CracerEnv(render_mode=None, obs_mode="state", action_mode="discrete")
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(0)
```

See `rl/README.md` for DQN training and inference.
