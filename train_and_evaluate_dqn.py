#!/usr/bin/env python3
"""
train_and_evaluate_dqn.py  (project root entry point)

DQN Training and Evaluation Pipeline
======================================
End-to-end pipeline:
  1. (Optional) --debug flag for a fast 2-3 min smoke test
  2. Train DQN for RL_TRAINING_EPISODES (default 2000) episodes
  3. Evaluate trained DQN vs TopK / LRU / LFU / Random on held-out data
  4. Save results + plots to results/

Fix history (2026):
  CRITICAL-1 : Now imports NOMADQNTrainer / CachePolicyEvaluator from
               stable_dqn_sim (was importing broken noma_caching_sim).
  CRITICAL-2 : episode_done logic removed; stable_dqn_sim handles it
               internally via req_remaining countdown.
  HIGH-1     : Uses NOMADQNTrainer.create_dqn_cache() so ALL config
               params (beta_frames, tau, hidden_dims, ...) are forwarded.
  HIGH-2     : Evaluation uses CachePolicyEvaluator (run_batch_episode)
               for a fair apples-to-apples comparison.
  MEDIUM-1   : Plot functions imported from stable_dqn_sim.
  LOW-1      : Docstring episode count corrected (50 -> 2000).

Usage:
    # Quick sanity check (~2-3 min on CPU):
    python train_and_evaluate_dqn.py --debug

    # Full run (~4-6 hrs on CPU, ~1 hr on GPU):
    python train_and_evaluate_dqn.py

Author: Cache-Aided NOMA Team
Date: December 2025  |  Revised: March 2026
"""

import sys
import os
import time
import pandas as pd
from pathlib import Path

# -- path setup ---------------------------------------------------------------
# This file lives at project root, so src/ is already one level down.
src_path = Path(__file__).parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# -- imports from the FIXED simulator (CRITICAL-1 fix) ------------------------
from src import config as cfg
from src.simulation.stable_dqn_sim import (
    NOMADQNTrainer,
    CachePolicyEvaluator,
    plot_training_curves,    # MEDIUM-1 fix: was plot_dqn_training
    plot_policy_comparison,  # MEDIUM-1 fix: was plot_comparison_results
)


# =============================================================================
# PHASE 1 -- TRAINING
# =============================================================================

def train_dqn(verbose: bool = True):
    """
    Train DQN cache using NOMADQNTrainer.

    CRITICAL-2 fix: episode_done is no longer computed or passed here.
    NOMADQNTrainer.train() / run_episode() manage episode boundaries
    internally via a req_remaining countdown.

    HIGH-1 fix: NOMADQNTrainer.create_dqn_cache() forwards ALL config
    parameters (hidden_dims, tau, gradient_clip, priority_beta_frames,
    warm_up_steps, train_freq) -- none are silently dropped.

    Returns:
        trained_cache : DQNCache (eval mode off; call set_eval_mode(True) to use)
        train_df      : pd.DataFrame -- per-episode training metrics
        test_df       : pd.DataFrame -- periodic test-phase metrics
    """
    print("\n" + "="*70)
    print("PHASE 1: DQN TRAINING")
    print("="*70)
    print(f"  Episodes     : {cfg.RL_TRAINING_EPISODES}")
    print(f"  Steps/episode: {cfg.RL_STEPS_PER_EPISODE:,}  "
          f"(= {cfg.NUM_USERS} users x {cfg.REQUESTS_PER_USER} req)")
    print(f"  Total steps  : {cfg.RL_TRAINING_STEPS:,}")
    print(f"  eps decay    : {cfg.RL_EPSILON_START} -> {cfg.RL_EPSILON_END} "
          f"over {cfg.RL_EPSILON_DECAY_STEPS:,} steps\n")

    trainer = NOMADQNTrainer(cfg, verbose=verbose)
    trained_cache, train_df = trainer.train(
        num_episodes=cfg.RL_TRAINING_EPISODES,
        test_interval=50,
        save_best=True,
    )
    test_df = pd.DataFrame(trainer.test_history)
    return trained_cache, train_df, test_df


# =============================================================================
# PHASE 2 -- EVALUATION
# =============================================================================

def evaluate_all(trained_cache, verbose: bool = True):
    """
    Evaluate DQN vs all baseline policies.

    HIGH-2 fix: uses CachePolicyEvaluator which calls run_batch_episode()
    for every policy -- identical NOMA pairing for a fair comparison.

    Returns:
        pd.DataFrame -- combined results across all policies & runs
    """
    print("\n" + "="*70)
    print("PHASE 2: POLICY COMPARISON")
    print("="*70)
    print(f"  Runs per policy: {cfg.NUM_RUNS}")
    print(f"  Eval requests  : {cfg.RL_EVAL_REQUESTS}\n")

    evaluator = CachePolicyEvaluator(cfg, verbose=verbose)
    combined_df = evaluator.compare_all_policies(
        num_runs=cfg.NUM_RUNS,
        pretrained_dqn=trained_cache,
    )
    return combined_df


# =============================================================================
# MAIN
# =============================================================================

def main(debug: bool = False):
    """
    Full pipeline: validate config -> train -> evaluate -> save -> plot.

    Args:
        debug : If True, calls set_debug_config() for a fast smoke test
                (~2-3 min on CPU).  Use this FIRST to confirm the
                pipeline works before the full 20M-step run.
    """
    if debug:
        print("\n[DEBUG MODE] Using small-scale config for smoke test")
        cfg.set_debug_config()

    print("\n" + "#"*70)
    print("#" + " "*10 + "DQN TRAINING & EVALUATION PIPELINE" + " "*14 + "#")
    print("#"*70)
    print(f"\n  Cache size : {cfg.CACHE_SIZE}")
    print(f"  Num files  : {cfg.NUM_FILES}")
    print(f"  Num users  : {cfg.NUM_USERS}")
    print(f"  Episodes   : {cfg.RL_TRAINING_EPISODES}")
    print(f"  Eval runs  : {cfg.NUM_RUNS}")
    print(f"  CIC reward : {cfg.RL_REWARD_CIC_ENABLED}  (correct = 2.0)")

    # -- validate config BEFORE spending time training -----------------------
    print()
    if not cfg.validate_config():
        print("\n[ERROR] Config validation failed. Fix config.py before running.")
        sys.exit(1)

    os.makedirs('results',     exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    t0 = time.time()

    # -- Phase 1: Train -------------------------------------------------------
    trained_cache, train_df, test_df = train_dqn()

    train_df.to_csv('results/dqn_training_history.csv', index=False)
    print("\n[OK] Training history  -> results/dqn_training_history.csv")
    if len(test_df) > 0:
        test_df.to_csv('results/dqn_test_history.csv', index=False)
        print("[OK] Test history      -> results/dqn_test_history.csv")

    plot_training_curves(
        train_df, test_df,
        save_path='results/dqn_training_curves.png',
    )
    print("[OK] Training curves   -> results/dqn_training_curves.png")

    trained_cache.save_model('checkpoints/dqn_trained.pth')
    print("[OK] Trained model     -> checkpoints/dqn_trained.pth")

    # -- Phase 2: Evaluate ----------------------------------------------------
    comparison_df = evaluate_all(trained_cache)

    comparison_df.to_csv('results/comparison_results.csv', index=False)
    print("\n[OK] Comparison data   -> results/comparison_results.csv")

    plot_policy_comparison(
        comparison_df,
        save_path='results/comparison_plots.png',
    )
    print("[OK] Comparison plots  -> results/comparison_plots.png")

    # -- Summary --------------------------------------------------------------
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70 + "\n")

    summary = comparison_df.groupby('policy').agg({
        'hit_rate':            ['mean', 'std'],
        'outage_probability':  ['mean', 'std'],
        'cic_benefit_rate':    ['mean', 'std'],
        'spectral_efficiency': ['mean', 'std'],
    }).round(4)
    print(summary)

    print("\n" + "-"*70)
    print("KEY FINDINGS:")
    print("-"*70)
    for policy in ['dqn', 'topk', 'lru', 'lfu', 'random']:
        pdata = comparison_df[comparison_df['policy'] == policy]
        if len(pdata) == 0:
            continue
        hit    = pdata['hit_rate'].mean()
        cic    = pdata['cic_benefit_rate'].mean()
        outage = pdata['outage_probability'].mean()
        tag    = "  <-- DQN" if policy == 'dqn' else ""
        print(f"  {policy.upper():8s}: Hit={hit:.1%}, CIC={cic:.1%}, Outage={outage:.1%}{tag}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    print("\n" + "#"*70)
    print("#" + " "*28 + "DONE" + " "*26 + "#")
    print("#"*70 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Train and evaluate DQN cache for Cache-Aided NOMA")
    parser.add_argument(
        '--debug', action='store_true',
        help='Run set_debug_config() for a fast 2-3 min smoke test')
    args = parser.parse_args()
    main(debug=args.debug)


