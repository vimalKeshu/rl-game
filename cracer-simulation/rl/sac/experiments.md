# SAC AutoResearch Experiment Log
# Objective: Agent clears many stages and generalizes to unseen levels
# Primary metric: eval_stage1_mean_stage (agent starts from stage 1, unseen seeds)
# Status: CONCLUDED — moving to PPO (see conclusion section at bottom)

---

## Experiment 1a: "Longer Horizon + Progress Signal" (DISCARDED)
- **Date**: 2026-03-23
- **Changes**: gamma 0.99→0.995, gradient_steps 2→3, reward shaping, MPS device
- **Result**: DISCARDED — gamma=0.995 caused Q-loss oscillation (400-870)
- **Learnings**: gamma=0.99 is stable. CPU is 2.2x faster than MPS.

---

## Experiment 1b: "Stronger Forward Progress Signal" (BASELINE)
- **Date**: 2026-03-23
- **Result**: Became baseline. eval_stage ~2.6 @ 1.7M training steps.
- **Key settings established**: gamma=0.99, cpu, gradient_steps=2, buffer=1M

---

## Experiment 2: "Smaller Buffer + More Exploitation" (DISCARDED)
- **Date**: 2026-04-10 | **Steps**: 1.2M
- **Root cause of failure**:
  - Training mean_stage reached 7.24 (MISLEADING — curriculum started episodes at stage 5-8)
  - Eval stage1 flat at 1.4–2.1 the entire run → catastrophic forgetting
  - gradient_steps=1 caused Q-loss spikes 5K–13K
- **Key learning**: Training mean_stage is a lie when curriculum starts episodes at high stages.
  Eval stage1 from scratch is the only valid metric.

---

## Experiment 3: "Stage-1 Floor + Stable Critic" (DISCARDED)
- **Date**: 2026-04-11 | **Steps**: 592K
- **Changes**: stage-1 floor 20%, gradient_steps→2, min_ep→200, new eval [1,3,5]
- **Result**: Best stage1=2.4, avg=1.64, then stalled. Q-loss spiked to 2480 at stage 4+.
- **Root cause**: crash_penalty=75 too cheap (0.08/step cost vs 0.50/step gain — agent ignored dying)

---

## Experiment 4: "Fix Reward Economics" (DISCARDED)
- **Date**: 2026-04-11 | **Steps**: 1.23M
- **Key fix**: crash_penalty 75→300 (crash now costs 0.33/step = 46% of per-step gain)
- **Mid-run fix**: target_entropy_ratio 0.35→0.25 (alpha was 0.95 at 198K, causing oscillation)
- **Result**: Best stage1=2.10, after-fix avg=1.59, then plateaued at 1.53.
  Q-loss climbed to mean 1688 as curriculum hit stages 5-6.
- **Root cause of plateau**: Curriculum still started episodes at stages 4-6 → training
  distribution drifted from stage 1 despite 20% floor. Same failure pattern, just slower.

---

## Experiment 5: "No Curriculum + Warm-Start from Exp 4" (DISCARDED)
- **Date**: 2026-04-12 | **Steps**: 964K
- **Core change**: curriculum_enabled=false — every episode starts at stage 1.
  Warm-started policy from Exp 4's best_eval.pt (navigation knowledge preserved).
- **Result**:
  - BEST upward trend of all SAC experiments — window averages rose every 10 checkpoints
  - Window avgs: 1.29 → 1.40 → 1.53 → 1.70 → 1.76 → 1.83 → 1.92 → 1.89 → 1.80 → 1.85
  - Best stage1: 2.20  |  Avg(last 10): 1.83
  - Q-losses: cleanest ever (mean 251, max 399 — no spikes)
  - Training distribution finally aligned with eval distribution
  - Stage3 best: 3.3  |  Stage5 best: 5.2 — still flat (agent rarely reached those stages)
- **Why it plateaued at ~1.85**:
  - Always-stage-1 means dense stage-1 experience but starved stage-2 exposure
  - Agent clears stage 1 in ~half episodes but hits stage 2 with barely any training there
  - Needs stage-2 transitions to learn stage-2, but rarely survives into them
  - This is a fundamental on-policy data scarcity problem for the SAC off-policy setup

---

## SAC CONCLUSION — Why We Are Moving to PPO

After 5 experiments and ~5M total environment steps, SAC has hit a consistent ceiling
of ~1.85–2.20 on eval_stage1_mean_stage. We are moving to PPO. Here is the honest
diagnosis of why SAC is not the best fit for this environment:

### 1. SAC was designed for continuous control — this is a discrete game
SAC originated for MuJoCo (continuous torques, dense rewards, smooth state spaces).
SAC-Discrete is an adaptation that works but carries the wrong inductive biases:
entropy maximisation actively fights the precise, repeatable behaviour needed to clear
game stages reliably. Every experiment required manual alpha/entropy tuning.

### 2. The replay buffer is incompatible with sequential skill-building
SAC stores 500K past transitions in a replay buffer. When the curriculum placed episodes
at stages 4-6, those transitions dominated the buffer and distorted the Q-value landscape.
When we disabled the curriculum (Exp 5), the buffer was clean — but then the agent
couldn't accumulate enough stage-2 transitions to learn stage-2 because it had to
survive stage 1 first. Off-policy learning + sequential stages = fundamental tension.

### 3. Alpha (entropy coefficient) fought learning precision throughout
Experiments 2, 3, 4, 5 all showed the same alpha oscillation problem. Alpha competed
with the policy's ability to commit to good actions. We manually fixed it twice.
PPO's entropy is a small soft regulariser coefficient — it doesn't compete with the
objective, it's a minor term in the total loss.

### 4. Sparse stage-clear rewards are handled better by GAE
The stage_bonus fires once per stage clear (~every 900 steps). In SAC this gets
averaged across a 500K buffer and bootstrapped into Q-values where the signal is weak.
PPO with GAE explicitly propagates the large sparse reward backward through the entire
trajectory — exactly what this environment needs.

### 5. Same ceiling across every configuration
Five experiments with different rewards, different curriculum designs, different entropy
targets, different buffer sizes — all hit the same ~2.0 wall on eval_stage1. When the
ceiling doesn't move regardless of configuration, it is algorithmic, not hyperparameter.

### What we are carrying to PPO (all validated):
- reward_crash_penalty: 300  (crash must hurt — 46% of per-step gain)
- reward_distance_scale: 0.2  (strong continuous progress gradient)
- reward_survival_bonus: 0.2  (staying alive pays)
- reward_stage_bonus: 500    (sparse but not overwhelming)
- curriculum_enabled: false   (always stage 1 — align training with eval distribution)
- eval setup: stages [1, 3, 5], 10 episodes each, deterministic
- primary metric: eval_stage1_mean_stage
- obs_normalizer, frame_stack=4, max_objects=10, seed_range=1000
- device: cpu

### Why PPO is a better fit:
- On-policy: training distribution ALWAYS matches current policy behaviour
- GAE propagates sparse stage-clear rewards backward through full trajectory
- Natural data distribution: agent sees stage 2 exactly as often as it clears stage 1
- Entropy is a soft coefficient (0.01-0.02), not a competing objective
- Used in every major game RL success (OpenAI Five, AlphaStar, etc.)
- Stage-sequential learning is PPO's natural setting

See: rl/ppo/experiments.md for PPO experiment log.
