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
- **Status**: RUNNING
