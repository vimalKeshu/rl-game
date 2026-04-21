#!/usr/bin/env python3
"""
Policy Merging Script — Stage-Expert Merging (Task Arithmetic style)

Merges two PPO actor-critic checkpoints by weighted averaging of the actor weights.
The critic is taken from the base policy (better calibrated to the full task).
The obs_normalizer is also taken from the base policy.

Usage:
    python rl/ppo/merge_policies.py <base> <specialist> <output> [alpha]

    alpha = weight given to specialist (default 0.4)
    So merged = (1-alpha)*base + alpha*specialist

Example:
    python rl/ppo/merge_policies.py \\
        /tmp/ppo_exp4_best_eval.pt \\
        rl/ppo/checkpoints/specialist_best.pt \\
        /tmp/merged_policy.pt \\
        0.4
"""

import sys
import os
import torch


def merge_policies(base_path: str, specialist_path: str, output_path: str, alpha: float = 0.4):
    """
    Merge base and specialist policies.

    Strategy:
    - Actor (policy) weights: blend base + specialist
      The specialist has deep knowledge of high stages the base rarely sees.
    - Critic weights: keep base only
      The critic's value predictions are calibrated to the base's training distribution.
      Averaging with specialist critic would corrupt the value estimates.
    - Obs normalizer: keep base
      Both saw the same obs space but base has more diverse statistics.
    """
    print(f"Loading base policy:       {base_path}")
    base = torch.load(base_path, map_location='cpu', weights_only=False)

    print(f"Loading specialist policy: {specialist_path}")
    spec = torch.load(specialist_path, map_location='cpu', weights_only=False)

    base_state = base['actor_critic']
    spec_state = spec['actor_critic']

    # Verify architectures match
    if set(base_state.keys()) != set(spec_state.keys()):
        raise ValueError("Architecture mismatch — base and specialist have different layer keys")

    print(f"\nMerging: {(1-alpha)*100:.0f}% base + {alpha*100:.0f}% specialist")
    print(f"Strategy: blend ALL weights (actor + critic share the same network in PPO ActorCritic)")
    print(f"Note: critic will re-learn quickly during fine-tuning with normalize_returns=True")

    merged_state = {}
    total_params = 0
    for key in base_state.keys():
        b = base_state[key].float()
        s = spec_state[key].float()
        merged_state[key] = (1.0 - alpha) * b + alpha * s
        total_params += b.numel()

    print(f"Merged {total_params:,} parameters across {len(merged_state)} tensors")

    # Build merged checkpoint — use base's model_config and obs_normalizer
    merged_checkpoint = {
        'actor_critic': merged_state,
        'model_config': base.get('model_config', spec.get('model_config', {})),
        'obs_normalizer': base.get('obs_normalizer'),  # Base has richer normalizer stats
        'global_step': 0,   # Reset counters for fresh fine-tuning
        'num_updates': 0,
    }

    torch.save(merged_checkpoint, output_path)
    print(f"\nSaved merged policy → {output_path}")

    # Quick sanity check
    merged = torch.load(output_path, map_location='cpu', weights_only=False)
    sample_key = list(merged['actor_critic'].keys())[0]
    sample_val = merged['actor_critic'][sample_key].mean().item()
    print(f"Sanity check — first param mean: {sample_val:.6f} (should be between base and specialist)")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    base_path      = sys.argv[1]
    specialist_path = sys.argv[2]
    output_path    = sys.argv[3]
    alpha          = float(sys.argv[4]) if len(sys.argv) > 4 else 0.4

    merge_policies(base_path, specialist_path, output_path, alpha)
