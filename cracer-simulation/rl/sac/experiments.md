# SAC AutoResearch Experiment Log
# Metric: mean_stage (higher = more levels cleared)
# Baseline: mean_stage ~2.6, mean_reward ~6000, mean_score ~3800 (at ~1.7M steps)

## Experiment 1a: "Longer Horizon + Progress Signal" (DISCARDED)
- **Date**: 2026-03-23
- **Changes**: gamma 0.99→0.995, gradient_steps 2→3, reward shaping, MPS device
- **Result**: DISCARDED — gamma=0.995 caused Q-loss oscillation (400-870), gradient_steps=3 crushed FPS to ~28 on MPS
- **Learnings**: gamma=0.99 is the stable sweet spot for this env. CPU is 2.2x faster than MPS for this model size.

## Experiment 1b: "Stronger Forward Progress Signal" (RUNNING)
- **Date**: 2026-03-23
- **Device**: CPU (~36 FPS steady state)
- **Hypothesis**: Agent plateaus at stage ~2.6 because forward-progress gradient is too weak
- **Changes from original config**:
  - reward_distance_scale: 0.05 → 0.1 (2x progress reward)
  - reward_distance_milestone_interval: 500 → 300 (denser milestones)
  - reward_distance_milestone: 50 → 40 (balance for higher frequency)
  - reward_crash_penalty: 50 → 75 (harder crash punishment)
  - reward_stage_bonus: 800 → 1000 (stronger completion signal)
  - reward_survival_bonus: 0.3 → 0.15 (less survival farming)
  - reward_safe_speed_bonus: 0.05 → 0.08 (reward aggressive driving)
  - target_entropy_ratio: 0.5 → 0.4 (more exploitation)
  - min_alpha: 0.1 → 0.05 (allow sharper policy)
  - buffer_size: 500K → 1M (retain diverse experience)
  - curriculum_min_episodes_per_stage: 200 → 150 (faster graduation)
  - device: cpu (2.2x faster than MPS)
- **Progress**:
  - 130K: Curriculum stage 1 graduated (reward=4211, comp=0.23)
  - 170K: Eval reward 3630, Stage 1.2, now in curriculum stage 2
  - Q-losses stable at 300-500 (much better than 1a)
  - Alpha converged ~0.9, entropy ~0.88 (near target 0.879)
- **Status**: RUNNING (PID 23837)
- **FPS**: ~36 steady state. ETA ~22h for 3M steps.

## Next Experiment Ideas (based on observations)
1. Reduce buffer_size 1M→500K to save ~1.4GB RAM (may improve FPS)
2. Try train_freq=2 (train every 2 steps → ~2x FPS at cost of sample efficiency)
3. Smaller network [256, 256, 128] for faster training
4. Learning rate warmup/schedule (cosine decay)
5. N-step returns (n=3) for better credit assignment instead of higher gamma
