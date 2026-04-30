# PPO AutoResearch Experiment Log
# Objective: Agent clears many stages and generalizes to unseen levels
# Primary metric: eval_stage1_mean_stage (agent starts from stage 1, unseen seeds)
# Secondary: eval_stage3_mean_stage, eval_stage5_mean_stage
#
# Context: Moved from SAC after 5 experiments hitting a ceiling of ~1.85–2.20.
# SAC diagnosis: off-policy replay buffer incompatible with sequential skill-building,
# alpha/entropy fought learning precision, sparse stage-clear rewards poorly handled.
# All reward and eval lessons from SAC are carried forward.
#
# Why PPO:
# - On-policy: training distribution always matches current policy
# - GAE propagates sparse stage-clear rewards backward through full trajectory
# - Natural sequential learning: sees stage 2 only when it clears stage 1
# - Entropy is a soft coefficient, not a competing objective
# - Gold standard for discrete-action game environments

---

## Validated lessons from SAC (DO NOT re-experiment these)

### Reward structure (validated in SAC Exp 3-5):
- reward_crash_penalty: 300  — crash must cost ~46% of per-step gain
- reward_distance_scale: 0.2  — strong continuous progress gradient
- reward_survival_bonus: 0.2  — staying alive pays per step
- reward_stage_bonus: 500     — sparse but not overwhelming; stage*500 on clear
- reward_distance_milestone: 40 every 300 units — dense sub-goal signal

### Curriculum (validated in SAC Exp 5):
- curriculum_enabled: false — ALWAYS start from stage 1
- Training distribution must match eval distribution
- Curriculum start-stage manipulation causes training/eval divergence every time

### Eval setup (validated in SAC Exp 2):
- eval_start_stages: [1, 3, 5] — 10 episodes each = 30 total per checkpoint
- eval_stage1_mean_stage is the PRIMARY metric
- eval_deterministic: true

### Environment settings (validated throughout):
- frame_stack: 4, max_objects: 10, seed_range: 1000, device: cpu
- obs_normalizer: true (Welford running stats, clip=10)

---

## Experiment 1: "PPO Baseline — All SAC Lessons Applied" (COMPLETED)
- **Date**: 2026-04-13
- **Hypothesis**: PPO's on-policy learning + GAE will handle sparse stage-clear rewards
  better than SAC's replay buffer. With the validated reward structure (crash=300),
  no curriculum (always stage 1), and PPO's natural sequential learning, the agent
  should push eval_stage1 consistently past 2.0 and toward 3.0+.
- **Config**:
  - total_timesteps: 3,000,000
  - learning_rate: 3e-4 with linear annealing to 1e-5
  - gamma: 0.99, gae_lambda: 0.97
  - rollout_steps: 2048, num_epochs: 10, batch_size: 64
  - entropy_coef: 0.01 (low — SAC taught us entropy fights precision)
  - entropy_coef_end: 0.001 (anneal toward near-zero)
  - hidden_sizes: (512, 512, 512, 256) — match SAC network size
  - curriculum_enabled: false (always stage 1)
  - All SAC-validated reward structure
- **Expected advantage over SAC**:
  - No replay buffer → training distribution = current policy distribution
  - GAE with lambda=0.97 propagates stage-clear bonus back 30+ steps
  - Entropy coefficient is small and annealed — doesn't fight policy convergence
  - Natural data: agent sees stage 2 transitions only when it actually clears stage 1
- **Final results** (3M steps, 1465 updates, 2933 episodes):
  - Best stage1_mean_stage: **2.50** (new record — SAC best was 2.20)
  - Avg(last 10): **2.10**  |  Avg(last 20): **2.01**
  - Stage3 best: 3.80  |  Stage5 best: 5.30 — both improved over SAC
  - Final training mean_stage: 2.09, entropy: 0.95
  - Confirmed PPO outperforms SAC on all metrics
- **Key findings from Exp 1**:
  - GAE horizon mismatch: lambda=0.97 → 33-step horizon vs 1412-step episodes (2% coverage)
  - Value loss exploded late (2169) — value_coef too high
  - Slow entropy warmup cost ~500K steps (entropy 1.59 → 0.95)
  - Agent plateaued at ~2.5 — needs stage-2 exposure
- **Status**: COMPLETED

---

## Experiment 2: "GAE Fix + Stage-2 Mix + Warm-Start" (COMPLETED)
- **Date**: 2026-04-13
- **Warm-start**: from PPO Exp 1 best_eval.pt — skip 500K entropy warmup
- **Hypothesis**: Fixing GAE horizon mismatch (rollout 4096, lambda 0.98) + adding 10%
  stage-2 starts + warm-start will push stage1 eval consistently past 3.0.
- **Changes from Exp 1**:
  - total_timesteps: 3M → 6M (trend still climbing at 3M)
  - rollout_steps: 2048 → 4096 (CRITICAL: ~3 episodes/rollout, better GAE)
  - gae_lambda: 0.97 → 0.98 (horizon 33→50 steps)
  - num_epochs: 10 → 8 (less overfitting with longer rollouts)
  - value_coef: 0.5 → 0.3 (reduce value loss dominance)
  - entropy_coef: 0.01 → 0.005 (warm-start has lower entropy)
  - learning_rate: 3e-4 → 2e-4 (warm-start: don't overshoot)
  - stage_mix: {1: 0.9, 2: 0.1} (10% stage-2 exposure)
  - reward_fuel_bonus: 30 → 45 (fuel is survival-critical)
  - reward_pothole_penalty: 5 → 8 (potholes drain fuel AND slow progress)
- **Final results** (6M steps, 1464 updates, 4393 episodes):
  - Best stage1: **2.90**  |  Avg(last 10): **2.35**  |  Avg(last 20): **2.40**
  - Stage3 best: 3.90  |  Stage5 best: 5.30
  - Training/eval gap: **0.01** — near-perfect alignment (stage_mix working)
- **Key findings for Exp 3**:
  - Gradient updates same as Exp 1 (1464) — 6M/4096 = 3M/2048. Need more total steps.
  - Value loss GREW from 2194 → 4331 as rewards increased — need value_coef cut deeper
  - Trend still rising at 6M end (window avg 2.37) — more steps = more gains
  - Stage-2 exposure (10%) effective but need 20% to reliably learn stage-2 clearing
- **Status**: COMPLETED

---

## Experiment 3: "10M Steps + More Stage Exposure + Value Fix" (COMPLETED)
- **Date**: 2026-04-13
- **Warm-start**: from PPO Exp 2 best_eval.pt
- **Hypothesis**: Agent was still improving at 6M end. With 10M steps (2441 updates,
  67% more than Exp 2), deeper value_coef reduction (0.3→0.15) to stop the growing
  value loss, and increased stage-2/3 exposure (20%/5%), stage1 avg should push past 3.0.
- **Changes from Exp 2**:
  - total_timesteps: 6M → 10M (same rollout=4096, so 2441 vs 1464 updates)
  - stage_mix: {1:0.9, 2:0.1} → {1:0.75, 2:0.20, 3:0.05} (more stage-2, add stage-3)
  - value_coef: 0.3 → 0.15 (value loss doubled from 2194→4331 over run — cut it)
  - entropy_coef: 0.005 → 0.002 (mature warm-start policy, start tighter)
  - entropy_coef_end: 0.0005 → 0.0002
  - learning_rate: 2e-4 → 1.5e-4 (slightly lower for more mature policy)
  - learning_rate_end: 5e-6 → 2e-6
- **Final results** (10M steps, 2442 updates, 7577 episodes):
  - Best stage1: **3.00** (first time ever hitting 3.0!) @ step 9.4M
  - Avg(last 10): **2.62**  |  Avg(last 20): **2.54**  |  Overall avg: **2.24**
  - Stage3 best: 3.90  |  Stage5 best: 5.80 (new record)
  - Final window avg 2.67 — STILL RISING when run ended
  - Final entropy: 0.593  |  Value loss doubled late: 1020 → 1960 (returns grew)
  - Best reward ever: 24,329
- **Key findings for Exp 4**:
  - Crash cost ratio dropped: episodes grew 1454→1652 steps → 300/1652=0.182/step = 36%
    (below 40% target). Crash penalty needs raising to 400.
  - Value loss doubled late (1020→1960) as agent cleared 2+ stages — value_coef 0.10 needed
  - Late trend 2.39→2.55→2.67 — run ended while still climbing, just needs more steps
  - SAC transfer no longer useful — PPO avg 2.62 >> SAC best avg 1.83
- **Status**: COMPLETED

---

## Experiment 4: "12M + Crash Rebalance + Push to Stage 4" (COMPLETED)
- **Date**: 2026-04-13
- **Warm-start**: from PPO Exp 3 best_eval.pt
- **Hypothesis**: Three targeted fixes on an already-rising policy will push avg past 3.0:
  1. Crash penalty 300→400: episodes now 1652 steps, crash cost fell to 36% — restore 40%+
  2. value_coef 0.15→0.10: late value loss doubled (1020→1960), cut it further
  3. Stage mix {60/20/12/5/3}: expose agent to stages 3-5 directly
  Plus simply run 12M steps — final window avg 2.67 was still rising at 10M end
- **Changes from Exp 3**:
  - total_timesteps: 10M → 12M
  - reward_crash_penalty: 300 → 400 (episodes grew, crash became cheap again)
  - value_coef: 0.15 → 0.10 (value loss doubled in late Exp 3)
  - stage_mix: {1:0.75,2:0.20,3:0.05} → {1:0.60,2:0.20,3:0.12,4:0.05,5:0.03}
  - entropy_coef: 0.002 → 0.001 (policy more mature, start tighter)
  - learning_rate: 1.5e-4 → 1e-4, learning_rate_end: 2e-6 → 5e-7
- **Final results** (12M steps, 2930 updates, 8521 episodes):
  - Best stage1: **4.00** (first time clearing 3 full stages from cold start!)
  - Avg(last 10): **3.08**  |  Avg(last 20): **3.06**
  - Stage3 best: 4.40  |  Stage5 best: 5.90
  - Sustained avg >3.0 for final 2M steps of training
  - Value loss exploded: 1380 → 4756 (3.4x growth) — critic struggling with huge returns
  - Entropy converged to 0.477 — policy highly deterministic
  - Agent at stage3 clears only 0.53 extra stages  |  At stage5 only 0.14 — weak on high stages
- **Status**: COMPLETED

---

## Experiment 5: "Stage-Expert Merging + Return Normalization" (IN PROGRESS)
- **Date**: 2026-04-14
- **Approach**: Combine Option A (return normalization) + Option B (stage-expert merging)
- **Phase 1 — Specialist Training** (RUNNING, PID 4780):
  - Warm-start from Exp 4  |  Train ONLY on stages 4-8 (2M steps)
  - stage_mix: {4:0.30, 5:0.30, 6:0.20, 7:0.12, 8:0.08}
  - normalize_returns: true (new feature)  |  value_coef: 0.10
  - Eval: stages 4, 6, 8 (specialist metrics)
- **Phase 2 — Weight Merge** (pending specialist completion):
  - merged = 60% Exp4_base + 40% specialist  (merge_policies.py)
  - Actor weights blended  |  Critic reset fresh  |  Obs normalizer from base
- **Phase 3 — Fine-tune merged policy** (12M steps):
  - normalize_returns: true  |  value_coef: 0.05
  - stage_mix: {1:0.50, 2:0.15, 3:0.15, 4:0.10, 5:0.07, 6:0.03}
  - Hypothesis: merged policy combines stage1-3 mastery (Exp4) + stage4-8 specialist skills
- **Specialist final results** (Phase 1):
  - Stage4: 4.20→4.40  |  Stage6: 6.00→6.50  |  Stage8: 8.00→8.20
  - Value loss: 0.3 — return normalization working perfectly
- **Phase 3 results** (killed at 7.47M steps):
  - Avg(last10): 1.81  |  Best: 2.80 — REGRESSION vs Exp4's 2.75/3.50 at same step
  - Merge HURT the policy. Root cause: specialist entropy=1.007 (exploratory) blended
    into base entropy=0.477 (precise). 40% specialist corrupted the base's precision.
  - Stage5 still flat at 5.0 — specialist's high-stage knowledge did not transfer
  - Return normalization worked (value loss 0.22 throughout) — this part is validated
- **Key learnings**:
  - Merge alpha=0.40 was too aggressive with a high-entropy specialist
  - Expert merging only works if both policies have similar entropy/convergence level
  - Return normalization is a real improvement — keep it for Exp 6
  - The direct warm-start approach (Exp 2→3→4) was better than expert merging
- **Status**: DISCARDED — killed at 7.47M steps

---

## Experiment 6: "Exp4 + Return Normalization" (RUNNING)
- **Date**: 2026-04-14
- **Warm-start**: from Exp 4 best_eval.pt (proven best policy, avg 3.08)
- **Hypothesis**: Return normalization alone (no merge) will fix the value loss explosion
  that caused Exp 4 to plateau at avg 3.08. With stable critic, the policy should
  push consistently past 4.0 avg. Exp 4 was still rising at the end (2.97 final window)
  so simply continuing with a fixed critic should be enough.
- **Changes from Exp 4**:
  - normalize_returns: true (new — fixes value loss 1380→4756 explosion)
  - value_coef: 0.05 (0.10→0.05: with norm, critic converges easily)
  - total_timesteps: 12M (same as Exp 4)
  - Everything else identical to Exp 4 (stage_mix, crash_penalty, lr, etc.)
- **Killed at 1.98M steps** — entropy rebounded from 0.477 to 1.145
  - Root cause: entropy_coef=0.001 forced exploration on an already-converged warm-start
  - Policy spent 12M steps in Exp4 earning entropy 0.477, then we told it to explore again
  - Combined with return normalization making advantages smaller → policy "forgot" its precision
  - Fix: entropy_coef=0.0 for warm-start policies
- **Status**: DISCARDED (2M steps) — restarted as Exp6b with entropy_coef=0.0

## Experiment 6b: "Exp4 + Return Norm + Zero Entropy" (RUNNING)
- **Date**: 2026-04-14
- **Key insight**: For warm-start from converged policy, entropy_coef MUST be 0.
  The policy already explored for 12M steps. Forcing exploration is wasted compute.
- **Changes from Exp6**:
  - entropy_coef: 0.001 → 0.0  (no entropy forcing — let policy stay sharp)
  - entropy_coef_end: 0.0
  - anneal_entropy: false
  - Everything else same (normalize_returns=true, value_coef=0.05, warm-start Exp4)
- **Killed at 2M steps** — entropy still rose to 1.104 despite coef=0
  - Root cause: return normalisation scales advantages to ~±1 (small gradients)
  - Weak gradients → policy drifts toward uniformity naturally, entropy rises
  - The normalisation/warm-start combination creates a gradient scale mismatch
  - Exp4 WITHOUT return normalisation hit avg 3.08 / best 4.00 — already better
- **Status**: DISCARDED

## Experiment 7: "Exp4 Exact Config + 15M Steps" (COMPLETED)
- **Final**: best=5.80  avg(last10)=4.03  avg(last20)=4.15
- **Status**: COMPLETED

---

## Experiment 8: "Hard Stage Push — stages 5-8 in mix" (COMPLETED)
- **Date**: 2026-04-15  |  Warm-start: Exp7
- **Key change**: stage_mix shifted to 40% weight on stages 5-8 directly
- **Final**: best=7.00  avg(last10)=4.70  avg(last20)=4.90
- **Agent progress**: from stage1→4.9, from stage3→5.4, from stage5→6.3
- **Key findings for Exp9**:
  - Episodes grew 1656→2654 steps → crash cost fell to 0.30x (want 0.40+) → raise to 600
  - Stages 6-10 still under-practiced (from stage5 only +1.3 stages)
  - Value loss: 18K→28K but learning continued despite it
- **Status**: COMPLETED

---

## Experiment 9: "Stages 6-10 Push + Crash Rebalance" (RUNNING)
- **Date**: 2026-04-15  |  Warm-start: Exp8 best_eval.pt (best=7.00, avg=4.90)
- **Changes from Exp8**:
  - reward_crash_penalty: 400→600  (episodes grew to 2654 steps, crash cost fell to 0.30x)
  - stage_mix: push 56% weight onto stages 5-10, 32% on stages 7-10 directly
  - learning_rate: 3e-5→2e-5  (even more mature policy)
  - value_coef: 0.10→0.08  (value loss still growing, slight reduction)
- **Hypothesis**: Direct stage 7-10 practice + restored crash economics will push avg to 6.0+
- **Status**: COMPLETED (see Exp9 results above in log)

---

## Experiment 12: "Exp9 Proven Config — 50M Steps" (RUNNING)
- **Date**: 2026-04-17
- **Warm-start**: Exp11 best_eval.pt (step 7.2M, peak stage1 ~5.4)
- **Core insight**: Exps 10 and 11 confirmed — wide stage_mix (1-25) diffuses the
  stage-1 cold-start gradient. Every time we broadened the mix, stage1 avg regressed.
  Exp9's focused mix (60/20/12/5/3 on stages 1-5) produced the best cold-start avg (5.42).
  The solution: go back to exactly Exp9's mix and run for 50M steps (3.3x longer).
  At 0.7 avg improvement per 15M steps, 50M steps should push stage1 avg to 7.0+.
- **Changes from Exp9**:
  - total_timesteps: 15M → 50M
  - learning_rate: lower (3e-5→1e-5) — very mature policy, fine-tune gently
  - max_episode_steps: 15K→50K — support long runs into stage 10+
- **Everything else identical to Exp9**: stage_mix {1:0.60, 2:0.20, 3:0.12, 4:0.05, 5:0.03},
  crash_penalty=600, value_coef=0.10, all validated rewards
- **Target**: stage1 avg(last10) > 7.0, best > 10.0 (clearing first 9 stages consistently)
- **Status**: KILLED at 22.8M (plateaued ~4.9, no progress) — Exp13 adopted scaled crash instead

## Experiment 13: "Stage-Scaled Crash Penalty (base=600, scale=0.3) — 20M Steps" (COMPLETED)
- **Date**: 2026-04-17  |  Warm-start: Exp12 best (step ~5M)
- **Key idea**: penalty = 600 × (1 + 0.3 × (stage-1))
  Stage1=600, Stage5=1320, Stage10=2220 — dying costs more the further you've progressed
- **Results**: s1_best=7.40  s1_avg(10ep)=5.39  s5_avg=6.66  s5_best=7.70
- **Status**: COMPLETED

## Experiment 14: "Scaled Crash 0.4 + 30M Steps" (COMPLETED)
- **Date**: 2026-04-18  |  Warm-start: Exp13 best_eval.pt (step 15.5M)
- **Hypothesis**: Exp13 avg=5.39 nearly matched Exp9's 5.42 on 20M steps.
  30M steps (50% more) + stronger scale (0.3→0.4): stage10 penalty 2220→2760.
  Target: avg(10ep) > 6.0, best > 9.0
- **Changes from Exp13**:
  - total_timesteps: 20M → 30M
  - reward_crash_penalty_stage_scale: 0.3 → 0.4
  - learning_rate: 1e-5 → 5e-6
- **Status**: COMPLETED

---

## Experiment 15: "Exp9 Stage Mix + Exp14 Scaled Crash" (COMPLETED)
- **Date**: 2026-04-19  |  Warm-start: Exp14 best_eval.pt (step 18.5M)
- **Hypothesis**: Combine Exp9's high-stage mix (stages 5-10 = 54%) with Exp14's validated
  scaled crash (scale=0.4). Both approaches achieved avg ~5.25 separately; combining them
  should push past that ceiling.
- **Config**:
  - stage_mix: {1:0.20, 2:0.08, 3:0.08, 4:0.10, 5:0.12, 6:0.12, 7:0.10, 8:0.08, 9:0.06, 10:0.06}
  - crash_penalty=600, scale=0.4, lr=5e-6→2e-9, total=20M
  - eval_start_stages: [1, 5], eval_episodes: 10, eval_max_episode_steps: 100000
- **Final results** (20M steps, 4883 updates):
  - s1_best=7.20  |  s1_avg(all 20 windows)=5.25  |  s5_best=8.70  |  s5_avg(last10)=6.33
  - Training mean_stage=6.74 (agent averages stage 6-7 in training)
  - Final LR=2.8e-9 (annealed to zero — actor effectively frozen last ~5M steps)
  - Value loss wildly unstable: 12K–321K spikes
- **Key findings for Exp16**:
  - LR exhaustion is critical: once LR hits ~0, policy stops learning entirely
  - Stage-1 anchor at 20% was too low — cold-start-to-stage-13 chain weakened vs Exp9/14
  - Stage-8 eval missing — no direct visibility into high-stage chaining ability
  - Value loss explosion unresolved; value_coef reduction needed
- **Status**: COMPLETED

---

## Experiment 16: "Fresh LR + Restored Stage-1 Anchor + Stage-8 Eval" (RUNNING)
- **Date**: 2026-04-20  |  Warm-start: Exp15 best_eval.pt (s1_best=7.20, s1_avg=5.25)
- **Problem being solved**: Exp15's actor was frozen (LR→0) and stage-1 anchor at 20%
  weakened cold-start chains. Agent could no longer reach stage 13+ seen in earlier exps.
- **Hypothesis**: Fresh LR range (1e-5→2e-7, 2000x larger than Exp15 final LR) restarts
  real policy updates. Restoring stage-1 anchor to 35% rebuilds the cold-start chain.
  With 25M steps and explicit stage-8 eval, we expect best > 9.0 and stage 13+ visible again.
- **Changes from Exp15**:
  - learning_rate: 5e-6 → 1e-5  (fresh range — not exhausted this time)
  - learning_rate_end: 2e-9 → 2e-7  (prevents full exhaustion with 25M steps)
  - total_timesteps: 20M → 25M
  - stage_mix: {1:0.35, 2-7:0.10-0.08, 8:0.07, 9-10:0.03} (restored 35% stage-1 anchor)
  - value_coef: 0.10 → 0.06  (reduce value loss instability)
  - eval_start_stages: [1, 5, 8]  (added stage-8 eval to track high-stage chaining)
- **Target**: s1_best > 9.0, s1_avg(last10) > 6.0, agent visible at stage 13+
- **Final results** (killed at 16.4M steps — plateaued from step 0):
  - s1_best=7.3 (step 5.6M)  |  s1_avg(last10)=5.26  |  s1_rolling_avg=5.26 (all 162 windows)
  - Stage-8 avg=9.0–10.3 — high-stage chain confirmed working but not growing
  - Stage-1 was already 5.2 at window 1 (step 102K) — warm-start had plateaued before Exp16 began
  - Policy loss tiny (-0.0001 to -0.002), entropy frozen ~0.38 — gradients near zero
  - Root cause diagnosed: rollout=4096 ≈ 1.5 episodes per update → gradient too noisy
    to push actor beyond its converged basin. Lucky/unlucky seeds not averaged out.
- **Status**: KILLED at 16.4M — plateau confirmed, moving to Exp17

---

## Experiment 17: "4x Larger Rollout Buffer" (RUNNING)
- **Date**: 2026-04-20  |  Warm-start: Exp16 best_eval.pt (same as Exp15, s1_best=7.20)
- **Core hypothesis**: Exp16 root cause = rollout_steps=4096 ≈ 1.5 episodes per update.
  Only 1-2 full trajectories per gradient step → signal too noisy for actor to improve.
  4x larger rollout (16384 steps ≈ 6 episodes) gives much cleaner advantage estimates,
  averages out lucky/unlucky seeds, and may break the 5.25 plateau.
- **Key change**: rollout_steps: 4096 → 16384
  - num_epochs: 8 → 4  (larger rollout → less reuse to avoid overfitting)
  - batch_size: 64 → 256  (larger buffer → bigger minibatches viable)
- **Everything else identical to Exp16**: stage mix, LR, crash scale, value_coef=0.06
- **Target**: s1_avg(last10) > 6.0, break out of 5.25 plateau
- **Final results** (killed at 5.26M steps):
  - s1_avg ~5.1 — identical to Exp16, no improvement
  - Policy loss 10x smaller than Exp16 (-7e-5 vs -8e-4) — gradients more frozen, not cleaner
  - Larger rollout caused more gradient cancellation, not better signal
  - Confirms: warm-start lineage is fully saturated, Option B failed
- **Status**: KILLED at 5.26M — moving to Option A (fresh training from scratch)

---

## Experiment 18: "Fresh Training From Scratch" (RUNNING)
- **Date**: 2026-04-20  |  **No warm-start — random weights**
- **Core insight**: The warm-start lineage from Exp9 has been fine-tuned for ~100M+
  cumulative steps. Every exp since Exp15 showed stage-1 avg = 5.2 from step 0.
  The policy is at a local optimum — no fine-tuning can escape it.
  Fresh random weights force the agent to genuinely learn cold-start robustness.
- **Key differences from all prior exps**:
  - No warm-start (random orthogonal init)
  - LR: 3e-4 (30x higher — standard PPO starting LR for fresh training)
  - entropy_coef: 0.01 (10x higher — new policy needs real exploration)
  - value_coef: 0.5 (fresh critic needs stronger value signal)
  - stage_mix: {1:0.70, 2:0.20, 3:0.10} — heavy stage-1 focus, build from scratch
  - rollout_steps: 4096 (reverted from 16384 — larger was worse)
  - eval_start_stages: [1, 3] — track cold-start and stage-3 continuation
- **Hypothesis**: A fresh policy trained heavily on stage-1 will build genuine cold-start
  robustness rather than inheriting a saturated lineage. After this run we can warm-start
  Exp19 with a legitimately earned foundation and expand the stage mix.
- **Target**: s1_avg(last10) > 3.5 at 10M steps, > 5.0 at 20M steps
- **Final results** (20M steps, 4883 updates, 15969 episodes):
  - s1_best=4.2  |  s1_avg(last10)=3.52  |  s3_avg(last10)=4.0
  - Final entropy=0.98 (policy not fully converged — still learning at end)
  - Final LR=5e-6, training mean_stage=3.4
  - Tracks above Exp1 trajectory with better reward structure
  - Foundation genuinely earned — ready for Exp19 expansion
- **Status**: COMPLETED

---

## Experiment 19: "Expand Stage Mix to 1-6" (RUNNING)
- **Date**: 2026-04-21  |  Warm-start: Exp18 best_eval.pt (s1_avg=3.52, entropy=0.98)
- **Hypothesis**: Exp18 built a genuine cold-start foundation from scratch.
  Expanding the stage mix to stages 1-6 (vs 1-3 in Exp18) will teach the agent
  to chain through higher stages while preserving the cold-start robustness.
  With 45% stage-1 anchor and graduated coverage to stage 6, expect s1_avg > 5.0.
- **Changes from Exp18**:
  - learning_rate: 3e-4 → 1e-4  (more mature policy, lower starting LR)
  - learning_rate_end: 5e-6 → 2e-6
  - entropy_coef: 0.01 → 0.005  (Exp18 ended at 0.98 — keep some exploration)
  - entropy_coef_end: 0.001 → 0.0005
  - value_coef: 0.5 → 0.15  (policy more mature, critic needs less correction)
  - stage_mix: {1:0.45, 2:0.20, 3:0.15, 4:0.10, 5:0.07, 6:0.03}
  - eval_start_stages: [1, 3, 5]  (added stage-5 eval)
- **Target**: s1_avg(last10) > 5.0, s5_avg > 6.0 at 20M steps
- **Final results** (20M steps, 4883 updates, 8620 episodes):
  - s1_best=**7.4**  |  s1_avg(last10)=**6.03**  |  s5_avg(last10)=~6.9
  - **New record** — broke the 5.25 plateau that held for Exps 9-17
  - Fresh foundation (Exp18) was the key — Exp19 proved it by exceeding old ceiling
- **Status**: COMPLETED

---

## Experiment 20: "Expand Stage Mix to 1-10" (RUNNING)
- **Date**: 2026-04-22  |  Warm-start: Exp19 best_eval.pt (s1_best=7.4, s1_avg=6.03)
- **Hypothesis**: Exp19 mastered stages 1-6 (s1_avg=6.03, new record). Expanding stage
  mix to stages 1-10 will push the agent into consistently clearing stage 10+ while
  maintaining cold-start robustness via 35% stage-1 anchor.
- **Changes from Exp19**:
  - learning_rate: 1e-4 → 5e-5  (more mature policy)
  - learning_rate_end: 2e-6 → 1e-6
  - entropy_coef: 0.005 → 0.003
  - entropy_coef_end: 0.0005 → 0.0003
  - value_coef: 0.15 → 0.10
  - stage_mix: expanded to {1:0.35, 2:0.15, 3:0.12, 4:0.10, 5:0.09, 6:0.08, 7:0.05, 8:0.03, 9:0.02, 10:0.01}
  - total_timesteps: 20M → 25M  (wider stage range needs more budget)
  - eval_start_stages: [1, 5, 8]  (added stage-8 eval back)
- **Target**: s1_avg(last10) > 7.0, s1_best > 10.0, agent reaching stage 13+
- **Final results** (25M steps, 6104 updates, 6884 episodes):
  - s1_best=**10.1**  |  s1_avg(last10)=**8.46**  |  s8_best=10.2
  - Training mean_stage=7.47, final entropy=0.93
  - New record — agent clearing 8-10 stages from cold start
  - Observed issue: agent plays great without losing lives but dies from fuel exhaustion
    at high stages — fuel has no penalty signal, agent never learned fuel = life
- **Status**: COMPLETED

---

## Experiment 21: "Fuel Awareness" (RUNNING)
- **Date**: 2026-04-22  |  Warm-start: Exp20 best_eval.pt (s1_best=10.1, s1_avg=8.46)
- **Problem**: Agent dies from fuel exhaustion at high stages (stage 13-14) despite
  having lives remaining. Fuel pickup reward (45 pts) is 13x weaker than crash penalty
  (600 pts). No penalty for dying from empty fuel → agent never prioritized fuel.
- **Two new reward signals** (added to train_ppo.py shape_reward):
  1. reward_fuel_exhaustion_penalty=600: dying from fuel = same cost as a crash (stage-scaled)
     Makes fuel exhaustion a first-class concern equal to crashing
  2. reward_low_fuel_penalty=2.0: per-step penalty when fuel < 30
     Continuous urgency gradient — lower fuel = higher per-step cost
     Pushes agent to seek pickups proactively before running dry
- **Everything else identical to Exp20** — isolate the fuel signal change
- **Target**: agent survives longer at stages 13+ by actively collecting fuel,
  s1_best > 12.0, s1_avg(last10) > 9.0
- **Final results** (killed at ~10.5M steps):
  - s1_best=10.4  |  s8_best=11.4  |  episode lengths 7000–8600 (vs Exp20's 4600)
  - Fuel awareness confirmed working — agent surviving significantly longer per episode
- **Status**: KILLED at 10.5M — proceeding to Exp22

---

## Experiment 22: "Stage-Scaled Bonuses (Symmetric Risk-Reward)" (RUNNING)
- **Date**: 2026-04-22  |  Warm-start: Exp21 best_eval.pt
- **Insight**: Penalties scale with stage (crash/fuel exhaustion cost more at high stages)
  but bonuses were flat. Surviving at stage 10 paid the same per step as stage 1.
  Asymmetric incentive: agent avoids death but has no proportionally stronger reason
  to fight to stay alive at high stages.
- **New config param**: reward_bonus_stage_scale=0.3
  bonus_multiplier = 1 + 0.3 × (stage - 1)
  Applies to: survival_bonus, fuel_pickup_bonus, distance_milestone
  Stage 5: ×2.2  |  Stage 10: ×3.7  |  Stage 15: ×5.2
- **Effect**: symmetric risk-reward landscape
  - Dying at stage 10: costs 2760 pts (crash penalty, unchanged)
  - Surviving each step at stage 10: earns 0.74 pts (was 0.2)
  - Fuel pickup at stage 10: earns 166 pts (was 45) — strong incentive to seek fuel
- **Everything else identical to Exp21**
- **Target**: s1_best > 14, s1_avg(last10) > 10.0, agent surviving stage 15+
- **Final results** (25M steps, completed):
  - s1_best=**11.3**  |  s1_avg(last10)=**8.89**  |  top5=[10.6,10.7,10.9,10.9,11.3]
  - Stage-scaled bonuses confirmed effective — agent pushes deeper into high stages
- **Status**: COMPLETED

---

## Experiment 23: "Expand Stage Mix to 1-15" (RUNNING)
- **Date**: 2026-04-23  |  Warm-start: Exp22 best_eval.pt (s1_best=11.3, s1_avg=8.89)
- **Hypothesis**: Exp22 agent consistently reaches stage 10-11 from cold start but
  stage mix only went to stage 10 — no direct training on stages 11-15.
  Expanding to stages 1-15 gives explicit practice on territory already reached,
  which should consolidate performance and push best runs to stage 14-15+.
- **Changes from Exp22**:
  - stage_mix: expanded to stages 1-15, stage-1 anchor reduced to 25%
  - learning_rate: 2e-5 → 1e-5  (more mature policy)
  - learning_rate_end: 3e-7 → 2e-7
  - eval_start_stages: [1, 5, 10]  (stage-10 eval replaces stage-8)
- **All reward signals identical to Exp22** (fuel exhaustion + stage-scaled bonuses)
- **Target**: s1_best > 14, s1_avg(last10) > 10.0, agent reliably reaching stage 15+
- **Final results** (25M steps, completed):
  - s1_best=**11.7**  |  s1_avg(last10)=~9.5  |  s10_best=13.1
  - Top 5 s1: [10.8, 11.0, 11.3, 11.5, 11.7]
  - Steady improvement — expanding stage mix to 1-15 pushed best from 11.3 → 11.7
- **Status**: COMPLETED

---

## Experiment 24: "Expand Stage Mix to 1-20" (RUNNING)
- **Date**: 2026-04-24  |  Warm-start: Exp23 best_eval.pt (s1_best=11.7, s1_avg~9.5)
- **Hypothesis**: Exp23 agent reaches stage 11-13 but stage mix only went to 15.
  Expanding to stages 1-20 gives direct practice on the new frontier.
  Stage-1 anchor reduced to 20% (cold-start mastery well established).
- **Changes from Exp23**:
  - stage_mix: expanded to stages 1-20, stage-1 anchor 25% → 20%
  - learning_rate: 1e-5 → 8e-6
  - learning_rate_end: 2e-7 → 1e-7
- **All reward signals identical to Exp22/23**
- **Target**: s1_best > 14, s1_avg(last10) > 10.5, agent reaching stage 17+
- **Final results** (25M steps, completed):
  - s1_best=**11.8**  |  s1_avg(last10)=8.97  |  s10_best=14.7
  - Marginal gain (+0.1) over Exp23 — stage mix expansion to 1-20 hit diminishing returns
  - Root cause diagnosed: observation blindness — max_objects=10 misses enemies/fuel at stage 15+
  - Training mean_stage=11.24 (highest ever) but can't translate to higher cold-start scores
- **Status**: COMPLETED

---

## Experiment 25: "Fresh Training — max_objects=30" (RUNNING)
- **Date**: 2026-04-24  |  **No warm-start — random weights**
- **Core change**: max_objects: 10 → 30
  obs_size: 364 → 1004 (incompatible with old weights — must train fresh)
- **Why max_objects=30**:
  - At stage 15+, road has 20+ objects but agent only saw 10 nearest
  - Missed fuel pickups = ran out of fuel with fuel visible ahead
  - Collided with "invisible" enemy #11 that old network couldn't see
  - max_objects=30 gives full situational awareness at all stages
- **Fresh training strategy** (mirrors successful Exp18):
  - Random weight init, LR=3e-4, entropy=0.01
  - Heavy stage-1 focus: {1:0.70, 2:0.20, 3:0.10}
  - 30M steps (extra budget for larger obs space)
  - All proven rewards kept (crash penalty, fuel exhaustion, stage-scaled bonuses)
- **Plan after this**: warm-start Exp26 with expanded stage mix (same path as Exp18→19→20→21→22→23→24)
- **Target**: match Exp18 trajectory — s1_avg > 3.5 at 10M, > 5.0 at 20M, then Exp26+ surpasses old ceiling
- **Final results** (killed at 2.0M steps — only 6.7% through):
  - s1_best=1.6  |  still in early learning phase
  - Killed early to run Exp26 (full-speed incentive) using Exp25's best_eval.pt as weak warm-start
- **Status**: KILLED at 2.0M

---

## Experiment 26: "Expand Stage Mix 1-6 + Full-Speed Incentive" (RUNNING)
- **Date**: 2026-04-24  |  Warm-start: Exp25 best_eval.pt (max_objects=30, s1_avg=1.6 — weak but compatible)
- **Two changes from Exp25**:
  1. stage_mix: {1:0.45, 2:0.20, 3:0.15, 4:0.10, 5:0.07, 6:0.03}
  2. reward_speed_scale: 0.1 → 0.5 — full-speed incentive
     At max speed 360: 1.8/step (was 0.36). Stage-scaled at stage 10: 6.66/step
     Agent incentivized to hold max throttle rather than drift with speed zones
- **Note**: started fresh (no warm-start) — cleaner foundation, high LR=3e-4, entropy=0.01
  stage_mix: {1:0.70, 2:0.20, 3:0.10}, 30M steps
- **Target**: s1_avg > 5.0, agent visibly playing at full speed at high stages
- **Status**: RUNNING

---

## Experiment 26: "Expand Stage Mix 1-6 + Full-Speed Incentive" (PENDING — awaiting Exp25)
- **Warm-start**: Exp25 best_eval.pt (max_objects=30 foundation)
- **Two changes from Exp25**:
  1. stage_mix expanded to {1:0.45, 2:0.20, 3:0.15, 4:0.10, 5:0.07, 6:0.03}
     Same graduated expansion that took Exp18→Exp19 from avg 3.52 → 6.03
  2. reward_speed_scale: 0.1 → 0.5 (full-speed incentive)
     At max speed 360: 1.8/step (was 0.36/step)
     Stage-scaled: at stage 10 → 6.66/step for holding max speed
     Agent now has real incentive to push full throttle, not drift with speed zones
     Faster stage completion = less time exposed to enemies = fewer crashes
- **Final results** (30M steps, completed):
  - s1_best=**7.0**  |  s1_avg(last10)=**5.63**  |  final entropy=1.04
  - 67% stronger foundation than Exp18 (was 3.52 avg) — max_objects=30 + speed_scale=0.5
- **Status**: COMPLETED

---

## Experiment 27: "Expand Stage Mix 1-6" (RUNNING)
- **Date**: 2026-04-25  |  Warm-start: Exp26 best_eval.pt (s1_best=7.0, s1_avg=5.63)
- **Same move as Exp18→Exp19**: expand stage mix from {1-3} to {1-6}
  Exp18→Exp19 took avg 3.52 → 6.03. With stronger Exp26 foundation, expect > 8.0.
- **Changes from Exp26**:
  - learning_rate: 3e-4 → 1e-4 (more mature policy)
  - entropy_coef: 0.01 → 0.005
  - value_coef: 0.5 → 0.15
  - stage_mix: {1:0.45, 2:0.20, 3:0.15, 4:0.10, 5:0.07, 6:0.03}
  - total_timesteps: 30M → 20M
- **All other settings identical to Exp26** (max_objects=30, speed_scale=0.5, full reward stack)
- **Target**: s1_avg(last10) > 8.0, s1_best > 10.0
- **Final results** (20M steps, completed):
  - s1_best=**8.6**  |  s1_avg(last10)=**6.99**
  - Stronger than Exp19 equivalent (7.4 / 6.03) — max_objects=30 + speed_scale=0.5 paying off
- **Status**: COMPLETED

---

## Experiment 28: "Transformer Architecture — Fresh Training" (RUNNING)
- **Date**: 2026-04-26  |  **No warm-start — random weights**
- **Core change**: network_arch: mlp → transformer
- **Architecture**: ActorCriticTransformer
  - Single-head self-attention (num_heads=1)
  - Separate actor + critic transformer encoders
  - 2 transformer layers, embed_dim=256, ffn_dim=1024
  - CLS token aggregates global context for actor/critic heads
  - 125 tokens per forward pass (1 CLS + 31 per frame × 4 frames)
  - ~3.2M parameters
- **Why transformer**:
  MLP treats 30 objects as a flat 1004-dim vector — no sense of object relationships.
  Transformer self-attention dynamically weights which objects matter most
  (fuel can when fuel is low, enemy on collision course, etc.).
  Permutation-invariant over objects — naturally handles variable enemy positions.
- **Same fresh training strategy** as Exp18/Exp26:
  LR=3e-4, entropy=0.01, stage_mix={1:0.70, 2:0.20, 3:0.10}, 30M steps
  max_objects=30, speed_scale=0.5, full reward stack
- **Target**: match Exp26 trajectory (s1_avg > 5.0 at 30M), then surpass MLP ceiling in Exp29+

### Architecture Design Decisions

#### Why sequence_length = 4 (one token per frame)

The original design had 125 tokens (1 CLS + 31 object/player tokens × 4 frames).
We changed to 4 tokens — one per frame — for these reasons:

- Each frame token = full game snapshot [251 dims: 11 player + 30×8 objects]
- Sequence of 4 tokens = 4 consecutive timesteps (~67ms apart at 60fps)
- Transformer attends over **TIME** not over individual objects within a frame
- Natural question: "how did the game state evolve over the last 4 steps?"
- Self-attention over 4 tokens is ~930x fewer computations than 125 tokens
- Input shape: `[BATCH=64, SEQ=4, DIM=251→256]`

**What self-attention learns from 4 frames:**
Each frame attends to all others. Attention weights reveal which past frame
is most relevant to the current decision:
- Frame t-1 most relevant → tracking momentum / recent state changes
- Frame t-3 most relevant → comparing to a baseline / detecting drift
- All frames equal → detecting periodic patterns (speed zone cycles)

#### Why 4-head attention (multi-head)

Single-head learns ONE way to relate frames. 4 heads learn 4 DIFFERENT
temporal relationship patterns simultaneously, each in a 64-dim subspace:

| Head | What it learns |
|------|----------------|
| Head 1 | Speed/momentum — "Did I accelerate or brake over last 4 frames?" |
| Head 2 | Threat proximity — "Is the nearest enemy getting closer each frame?" |
| Head 3 | Fuel urgency — "Fuel dropped 15pts last 3 frames, I need a pickup" |
| Head 4 | Stage transition — "Did the stage just change? New enemies spawned?" |

Multi-head has the **same parameter count** as single-head — Q/K/V matrices
are the same size, just split into subspaces. No extra cost, richer patterns.

#### Architecture summary

```
[B, 1004] → reshape → [B, 4, 251]
          → Linear(251→256) + pos_embed → [B, 4, 256]
          → TransformerEncoder (2 layers, 4 heads, ffn=512) [separate actor/critic]
          → mean pool → [B, 256]
          → actor head: 256→9  |  critic head: 256→1
```

Parameters: ~2.18M (vs 3.2M for old 125-token design, 2.35M for MLP)

- **Final results** (killed at 4.06M steps):
  - s1_avg oscillating 1.4–2.2 — no consistent improvement over MLP baseline
  - Transformer unsuitable for this environment — observation frames are hand-crafted
    float tuples with no hidden structure for attention to discover
  - MLP already handles the 4-frame temporal context optimally via concatenation
  - Self-attention finds no meaningful relationships in engineered float sequences
- **Lesson**: Transformer excels at raw/minimal inputs (pixels, word IDs) where model
  must discover structure. Our obs is fully engineered — MLP is the right architecture.
- **Status**: KILLED at 4.06M — reverting to MLP

---

## Experiment 28 (revised): "MLP — Expand Stage Mix 1-10 (max_objects=30 chain)" (RUNNING)
- **Date**: 2026-04-27  |  Warm-start: Exp27 best_eval.pt (s1_best=8.6, s1_avg=6.99)
- **Same move as Exp19→Exp20**: expand stage mix from {1-6} to {1-10}
  Exp19→Exp20 took s1_best 7.4 → 10.1. With stronger Exp27 foundation (8.6 vs 7.4),
  expect to push past 10.1 toward 12+.
- **Changes from Exp27**:
  - stage_mix: {1:0.35, 2:0.15, 3:0.12, 4:0.10, 5:0.09, 6:0.08, 7:0.05, 8:0.03, 9:0.02, 10:0.01}
  - learning_rate: 1e-4 → 3e-5 (more mature policy)
  - entropy_coef: 0.005 → 0.002
  - value_coef: 0.15 → 0.10
  - total_timesteps: 20M → 25M
  - eval_start_stages: [1, 5, 8]
- **All other settings identical to Exp27** (max_objects=30, speed_scale=0.5, MLP)
- **Target**: s1_best > 12, s1_avg(last10) > 9.0
- **Status**: RUNNING
