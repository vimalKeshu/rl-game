# Cancer Kingdom

A playable turn-based strategy prototype based on the game concept in
[`cancer_kingdom_game_concept.md`](cancer_kingdom_game_concept.md).

You command cancer treatments across a connected map of the human body. Cancer
populations grow, mutate into resistant clones, recruit blood supply, suppress
immune activity, damage organs, and metastasize along blood and lymphatic
routes. Treatments can stop them, but every intervention costs action points
and may harm the patient or select for resistance.

## Run

```bash
python3 -m pip install -r cancer-kingdom/requirements.txt
python3 cancer-kingdom/game.py
```

## Controls

- Click an organ to inspect and target it.
- Click a treatment or press `1`–`6` to use it.
- Press `Space` or click **End Turn** to advance the simulation.
- Press `H` for help, `R` to restart, and `Esc` to exit.

## Win condition

Eliminate all detectable cancer, or keep total burden below 8 through turn 35
while maintaining the patient's health. You lose if a vital organ fails, body
health collapses, or disease becomes overwhelming.

The rules live in `simulation.py` without a Pygame dependency, making them
suitable for a later Gymnasium/RL wrapper.

Run the simulation tests with:

```bash
python3 -m unittest discover -v -s cancer-kingdom -p 'test_simulation.py'
```
