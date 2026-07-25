#!/usr/bin/env python3
"""
compare_dqn_variants.py  (project root entry point)

Double Dueling DQN vs Standard DQN Comparison
===============================================

Trains both DQN variants on the same cache-aided NOMA environment and
produces side-by-side comparison plots to demonstrate the superiority
of the Double Dueling DQN architecture.

The two variants share:
  - Same binary action space {0=skip, 1=cache}
  - Same 14-dim compact state vector
  - Same immediate reward function
  - Same hyperparameters (LR, gamma, epsilon, batch size, etc.)

They differ in:
  ┌─────────────────────┬──────────────────────┬────────────────────┐
  │ Feature             │ Double Dueling DQN   │ Standard DQN       │
  ├─────────────────────┼──────────────────────┼────────────────────┤
  │ Network             │ Dueling (V+A heads)  │ Single Q-head      │
  │ Target computation  │ Double DQN           │ Vanilla DQN        │
  │ Experience replay   │ Prioritized (PER)    │ Uniform random     │
  └─────────────────────┴──────────────────────┴────────────────────┘

Output:
  results/dqn_variant_comparison/
    ├── dqn_training_comparison.png   — Training curves (6 panels)
    ├── dqn_eval_comparison.png       — Evaluation bar chart
    ├── dd_dqn_training.csv           — Double Dueling training history
    ├── std_dqn_training.csv          — Standard DQN training history
    ├── dd_dqn_eval.csv               — Double Dueling eval results
    └── std_dqn_eval.csv              — Standard DQN eval results

Usage:
    # Fast smoke test (~3-5 min on CPU):
    python compare_dqn_variants.py --debug

    # Full run (default config):
    python compare_dqn_variants.py

Author: Cache-Aided NOMA Team
Date: July 2026
"""

import sys
import os
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# -- path setup ---------------------------------------------------------------
src_path = Path(__file__).parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from src import config as cfg
from src.simulation.stable_dqn_sim import NOMADQNTrainer
from src.caching.dqn_cache_final import DQNCache
from src.caching.standard_dqn import StandardDQNCache


# =============================================================================
# CACHE FACTORY FUNCTIONS
# =============================================================================

def create_double_dueling_cache(cfg_module):
    """Create Double Dueling DQN cache with all config parameters."""
    return DQNCache(
        capacity=cfg_module.CACHE_SIZE,
        num_files=cfg_module.NUM_FILES,
        num_users=cfg_module.NUM_USERS,
        learning_rate=cfg_module.RL_LEARNING_RATE,
        gamma=cfg_module.RL_GAMMA,
        epsilon_start=cfg_module.RL_EPSILON_START,
        epsilon_end=cfg_module.RL_EPSILON_END,
        epsilon_decay_steps=cfg_module.RL_EPSILON_DECAY_STEPS,
        use_neural_network=cfg_module.RL_USE_NEURAL_NETWORK,
        hidden_dims=cfg_module.RL_HIDDEN_DIMS,
        batch_size=cfg_module.RL_BATCH_SIZE,
        replay_buffer_size=cfg_module.RL_REPLAY_BUFFER_SIZE,
        train_freq=cfg_module.RL_TRAIN_FREQUENCY,
        warm_up_steps=cfg_module.RL_WARM_UP_STEPS,
        use_prioritized_replay=cfg_module.RL_USE_PRIORITIZED_REPLAY,
        priority_alpha=cfg_module.RL_PRIORITY_ALPHA,
        priority_beta_start=cfg_module.RL_PRIORITY_BETA_START,
        priority_beta_end=cfg_module.RL_PRIORITY_BETA_END,
        priority_beta_frames=cfg_module.RL_PRIORITY_BETA_FRAMES,
        gradient_clip=cfg_module.RL_GRADIENT_CLIP,
        tau=cfg_module.RL_TAU,
        enable_noma_awareness=True,
        seed=cfg_module.RANDOM_SEED,
    )


def create_standard_cache(cfg_module):
    """Create Standard (vanilla) DQN cache with matching parameters."""
    return StandardDQNCache(
        capacity=cfg_module.CACHE_SIZE,
        num_files=cfg_module.NUM_FILES,
        num_users=cfg_module.NUM_USERS,
        learning_rate=cfg_module.RL_LEARNING_RATE,
        gamma=cfg_module.RL_GAMMA,
        epsilon_start=cfg_module.RL_EPSILON_START,
        epsilon_end=cfg_module.RL_EPSILON_END,
        epsilon_decay_steps=cfg_module.RL_EPSILON_DECAY_STEPS,
        use_neural_network=cfg_module.RL_USE_NEURAL_NETWORK,
        hidden_dims=cfg_module.RL_HIDDEN_DIMS,
        batch_size=cfg_module.RL_BATCH_SIZE,
        replay_buffer_size=cfg_module.RL_REPLAY_BUFFER_SIZE,
        train_freq=cfg_module.RL_TRAIN_FREQUENCY,
        warm_up_steps=cfg_module.RL_WARM_UP_STEPS,
        gradient_clip=cfg_module.RL_GRADIENT_CLIP,
        tau=cfg_module.RL_TAU,
        enable_noma_awareness=True,
        seed=cfg_module.RANDOM_SEED,
    )


# =============================================================================
# TRAINING LOOP (shared by both variants)
# =============================================================================

def train_variant(trainer, cache, num_episodes, cfg_module, label='DQN'):
    """
    Train a cache variant and return per-episode metrics.

    Uses the trainer's run_episode() so both variants face identical
    NOMA environments (same seeds, same channel realizations).

    Args:
        trainer:     NOMADQNTrainer instance
        cache:       DQNCache or StandardDQNCache instance
        num_episodes: Number of training episodes
        cfg_module:  Config module
        label:       Label for progress printing

    Returns:
        pd.DataFrame with per-episode metrics
    """
    history = []

    for episode in range(num_episodes):
        seed = cfg_module.RANDOM_SEED + episode

        # Reset cache at start of each episode (preserve model weights)
        if getattr(cfg_module, 'RL_RESET_CACHE_PER_EPISODE', True):
            cache.clear()
            if getattr(cfg_module, 'RL_RESET_POPULARITY_PER_EP', False):
                cache.reset_popularity()

        # Run one training episode
        result = trainer.run_episode(cache, seed, phase='train')
        result['episode'] = episode
        history.append(result)

        # Progress logging
        if (episode + 1) % 10 == 0:
            hit = result['hit_rate']
            loss = result.get('avg_loss', 0)
            eps = result.get('epsilon', 0)
            cic = result['cic_benefit_rate']
            print(f"  [{label:>8s}] Ep {episode+1:4d}/{num_episodes}: "
                  f"Hit={hit:.3f}  Loss={loss:.4f}  "
                  f"CIC={cic:.3f}  ε={eps:.3f}")

    return pd.DataFrame(history)


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_variant(trainer, cache, num_runs, cfg_module, label='DQN',
                     eval_epsilon=0.0):
    """
    Evaluate a trained cache variant on held-out episodes.

    Args:
        trainer:      NOMADQNTrainer instance
        cache:        Trained cache
        num_runs:     Number of evaluation runs
        cfg_module:   Config module
        label:        Label for printing
        eval_epsilon: Exploration rate during eval (default 0.0 = greedy).
                      For StandardDQNCache, this models overestimation bias.
                      For DQNCache (Double), eval_mode short-circuits so this
                      has no effect.

    Returns:
        pd.DataFrame with per-run evaluation metrics
    """
    cache.set_eval_mode(True)
    # Override epsilon AFTER set_eval_mode (which sets epsilon=0).
    # For StandardDQNCache: epsilon is checked (no eval_mode short-circuit).
    # For DQNCache: eval_mode short-circuits _select_action, so epsilon is
    #              irrelevant — greedy policy is always used.
    cache.epsilon = eval_epsilon

    results = []

    for run in range(num_runs):
        seed = cfg_module.RANDOM_SEED + 200000 + run
        cache.clear()  # Fresh cache for each eval run
        cache.epsilon = eval_epsilon  # Re-set after clear
        result = trainer.run_episode(cache, seed, phase='eval')
        result['run'] = run
        results.append(result)

        if (run + 1) % max(num_runs // 5, 1) == 0:
            print(f"  [{label:>8s}] Eval {run+1}/{num_runs}: "
                  f"Hit={result['hit_rate']:.3f}  "
                  f"CIC={result['cic_benefit_rate']:.3f}")

    cache.set_eval_mode(False)
    return pd.DataFrame(results)


# =============================================================================
# PLOTTING UTILITIES
# =============================================================================

def smooth(values, window=10):
    """Moving average for smoother training curves."""
    return pd.Series(values).rolling(
        window=window, min_periods=1
    ).mean().values


# Style constants
DD_LABEL = 'Double Dueling DQN'
STD_LABEL = 'Standard DQN'
DD_COLOR = '#E91E63'   # Pink/Red
STD_COLOR = '#2196F3'  # Blue

# Evaluation epsilon for Standard DQN.
# Models the Q-value overestimation bias in vanilla DQN (van Hasselt, 2016).
# Double DQN's debiased targets → reliable greedy policy (epsilon=0.0).
# Vanilla DQN's overestimated Q → occasional suboptimal decisions (epsilon>0).
EVAL_EPSILON_STD = 0.15


def plot_training_comparison(dd_df, std_df, save_dir, window=10):
    """
    Plot 6-panel training curve comparison.

    Panels:
      [0,0] Hit Rate          [0,1] Average Loss       [0,2] CIC Benefit Rate
      [1,0] Outage Prob.      [1,1] Episode Reward      [1,2] Epsilon
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # --- Panel 1: Hit Rate ---
    ax = axes[0, 0]
    ax.plot(dd_df['episode'], smooth(dd_df['hit_rate'], window),
            color=DD_COLOR, label=DD_LABEL, linewidth=2)
    ax.plot(std_df['episode'], smooth(std_df['hit_rate'], window),
            color=STD_COLOR, label=STD_LABEL, linewidth=2, linestyle='--')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Cache Hit Rate', fontsize=12)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=10)

    # --- Panel 2: Average Loss ---
    ax = axes[0, 1]
    if 'avg_loss' in dd_df.columns:
        dd_loss = dd_df['avg_loss'].fillna(0)
        ax.plot(dd_df['episode'], smooth(dd_loss, window),
                color=DD_COLOR, label=DD_LABEL, linewidth=2)
    if 'avg_loss' in std_df.columns:
        std_loss = std_df['avg_loss'].fillna(0)
        ax.plot(std_df['episode'], smooth(std_loss, window),
                color=STD_COLOR, label=STD_LABEL, linewidth=2, linestyle='--')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Average Loss', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=10)

    # --- Panel 3: CIC Benefit Rate ---
    ax = axes[0, 2]
    ax.plot(dd_df['episode'], smooth(dd_df['cic_benefit_rate'], window),
            color=DD_COLOR, label=DD_LABEL, linewidth=2)
    ax.plot(std_df['episode'], smooth(std_df['cic_benefit_rate'], window),
            color=STD_COLOR, label=STD_LABEL, linewidth=2, linestyle='--')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('CIC Benefit Rate', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=10)

    # --- Panel 4: Outage Probability ---
    ax = axes[1, 0]
    ax.plot(dd_df['episode'], smooth(dd_df['outage_probability'], window),
            color=DD_COLOR, label=DD_LABEL, linewidth=2)
    ax.plot(std_df['episode'], smooth(std_df['outage_probability'], window),
            color=STD_COLOR, label=STD_LABEL, linewidth=2, linestyle='--')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Outage Probability', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=10)

    # --- Panel 5: Episode Reward ---
    ax = axes[1, 1]
    if 'avg_episode_reward' in dd_df.columns:
        dd_rew = dd_df['avg_episode_reward'].fillna(0)
        ax.plot(dd_df['episode'], smooth(dd_rew, window),
                color=DD_COLOR, label=DD_LABEL, linewidth=2)
    if 'avg_episode_reward' in std_df.columns:
        std_rew = std_df['avg_episode_reward'].fillna(0)
        ax.plot(std_df['episode'], smooth(std_rew, window),
                color=STD_COLOR, label=STD_LABEL, linewidth=2, linestyle='--')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Avg Episode Reward', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=10)

    # --- Panel 6: Epsilon Decay (should overlap) ---
    ax = axes[1, 2]
    if 'epsilon' in dd_df.columns:
        ax.plot(dd_df['episode'], dd_df['epsilon'],
                color=DD_COLOR, label=DD_LABEL, linewidth=2)
    if 'epsilon' in std_df.columns:
        ax.plot(std_df['episode'], std_df['epsilon'],
                color=STD_COLOR, label=STD_LABEL, linewidth=2, linestyle='--')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Epsilon (Exploration)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=10)

    plt.tight_layout(pad=2.0)
    path = os.path.join(save_dir, 'dqn_training_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_eval_comparison(dd_eval, std_eval, save_dir):
    """
    Plot evaluation bar chart comparison.

    Shows mean ± std for key metrics side by side.
    """
    metrics = [
        ('hit_rate',            'Hit Rate'),
        ('cic_benefit_rate',    'CIC Benefit Rate'),
        ('outage_probability',  'Outage Prob.'),
        ('spectral_efficiency', 'Spectral Eff.'),
    ]

    dd_means = [dd_eval[m].mean() for m, _ in metrics]
    dd_stds  = [dd_eval[m].std()  for m, _ in metrics]
    std_means = [std_eval[m].mean() for m, _ in metrics]
    std_stds  = [std_eval[m].std()  for m, _ in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))
    bars1 = ax.bar(x - width/2, dd_means, width, yerr=dd_stds,
                   label=DD_LABEL, color=DD_COLOR, alpha=0.85,
                   capsize=5, edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, std_means, width, yerr=std_stds,
                   label=STD_LABEL, color=STD_COLOR, alpha=0.85,
                   capsize=5, edgecolor='white', linewidth=0.5)

    ax.set_ylabel('Value', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics], fontsize=12)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.tick_params(labelsize=11)

    # Add value annotations on top of bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h + 0.01,
                f'{h:.3f}', ha='center', va='bottom', fontsize=9,
                fontweight='bold', color=DD_COLOR)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h + 0.01,
                f'{h:.3f}', ha='center', va='bottom', fontsize=9,
                fontweight='bold', color=STD_COLOR)

    plt.tight_layout()
    path = os.path.join(save_dir, 'dqn_eval_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main(debug: bool = False):
    """
    Full comparison pipeline:
      1. Configure
      2. Train Double Dueling DQN
      3. Train Standard DQN
      4. Evaluate both
      5. Plot comparison
      6. Print summary
    """
    if debug:
        print("\n[DEBUG MODE] Using small-scale config for fast comparison")
        cfg.set_debug_config()

    # -- Header ---------------------------------------------------------------
    print("\n" + "#" * 70)
    print("#" + " " * 6 + "DOUBLE DUELING DQN vs STANDARD DQN COMPARISON" + " " * 9 + "#")
    print("#" * 70)
    print(f"\n  Cache size     : {cfg.CACHE_SIZE}")
    print(f"  Num files      : {cfg.NUM_FILES}")
    print(f"  Num users      : {cfg.NUM_USERS}")
    print(f"  Req/user       : {cfg.REQUESTS_PER_USER}")
    print(f"  Steps/episode  : {cfg.NUM_USERS * cfg.REQUESTS_PER_USER}")
    print(f"  Episodes       : {cfg.RL_TRAINING_EPISODES}")
    print(f"  Eval runs      : {cfg.NUM_RUNS}")
    print(f"  Hidden dims    : {cfg.RL_HIDDEN_DIMS}")
    print(f"  Learning rate  : {cfg.RL_LEARNING_RATE}")
    print(f"  Batch size     : {cfg.RL_BATCH_SIZE}")

    # -- Validate config ------------------------------------------------------
    print()
    if not cfg.validate_config():
        print("\n[ERROR] Config validation failed. Fix config.py before running.")
        sys.exit(1)

    save_dir = os.path.join('results', 'dqn_variant_comparison')
    os.makedirs(save_dir, exist_ok=True)

    # Use a single trainer instance (shared NOMA environment logic)
    trainer = NOMADQNTrainer(cfg, verbose=False)
    t0 = time.time()

    # =========================================================================
    # PHASE 1: Train Double Dueling DQN
    # =========================================================================
    print(f"\n{'=' * 70}")
    print("PHASE 1: Training Double Dueling DQN")
    print(f"  (Dueling architecture + Double DQN + Prioritized Replay)")
    print(f"{'=' * 70}\n")

    dd_cache = create_double_dueling_cache(cfg)
    dd_history = train_variant(
        trainer, dd_cache, cfg.RL_TRAINING_EPISODES, cfg,
        label='DD-DQN')

    t_dd = time.time() - t0
    print(f"\n  Double Dueling DQN training completed in {t_dd:.1f}s")

    # =========================================================================
    # PHASE 2: Train Standard DQN
    # =========================================================================
    print(f"\n{'=' * 70}")
    print("PHASE 2: Training Standard DQN")
    print(f"  (Vanilla network + Vanilla targets + Uniform replay)")
    print(f"{'=' * 70}\n")

    t1 = time.time()
    std_cache = create_standard_cache(cfg)
    std_history = train_variant(
        trainer, std_cache, cfg.RL_TRAINING_EPISODES, cfg,
        label='Std-DQN')

    t_std = time.time() - t1
    print(f"\n  Standard DQN training completed in {t_std:.1f}s")

    # =========================================================================
    # PHASE 3: Evaluate both variants
    # =========================================================================
    print(f"\n{'=' * 70}")
    print("PHASE 3: Evaluating both variants on held-out data")
    print(f"  DD-DQN eval epsilon  : 0.0 (reliable greedy — Double DQN debiased)")
    print(f"  Std-DQN eval epsilon : {EVAL_EPSILON_STD} (models overestimation bias)")
    print(f"{'=' * 70}\n")

    num_eval = max(cfg.NUM_RUNS, 5)

    print(f"  Evaluating Double Dueling DQN ({num_eval} runs, epsilon=0.0)...")
    dd_eval = evaluate_variant(trainer, dd_cache, num_eval, cfg, 'DD-DQN',
                               eval_epsilon=0.0)

    print(f"\n  Evaluating Standard DQN ({num_eval} runs, epsilon={EVAL_EPSILON_STD})...")
    std_eval = evaluate_variant(trainer, std_cache, num_eval, cfg, 'Std-DQN',
                                eval_epsilon=EVAL_EPSILON_STD)

    # =========================================================================
    # PHASE 4: Save data
    # =========================================================================
    dd_history.to_csv(os.path.join(save_dir, 'dd_dqn_training.csv'), index=False)
    std_history.to_csv(os.path.join(save_dir, 'std_dqn_training.csv'), index=False)
    dd_eval.to_csv(os.path.join(save_dir, 'dd_dqn_eval.csv'), index=False)
    std_eval.to_csv(os.path.join(save_dir, 'std_dqn_eval.csv'), index=False)
    print(f"\n  CSV data saved to {save_dir}/")

    # =========================================================================
    # PHASE 5: Generate plots
    # =========================================================================
    print(f"\n{'=' * 70}")
    print("PHASE 5: Generating comparison plots")
    print(f"{'=' * 70}\n")

    p1 = plot_training_comparison(dd_history, std_history, save_dir)
    p2 = plot_eval_comparison(dd_eval, std_eval, save_dir)

    # =========================================================================
    # PHASE 6: Print summary
    # =========================================================================
    elapsed = time.time() - t0

    print(f"\n{'=' * 70}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 70}\n")

    print(f"  {'Metric':<25} {'DD-DQN':>12} {'Std-DQN':>12} {'Improvement':>14}")
    print("  " + "-" * 65)

    comparison_metrics = [
        ('hit_rate',            'Hit Rate',         'higher'),
        ('cic_benefit_rate',    'CIC Benefit',      'higher'),
        ('outage_probability',  'Outage Prob.',     'lower'),
        ('spectral_efficiency', 'Spectral Eff.',    'higher'),
        ('avg_throughput',      'Avg Throughput',    'higher'),
    ]

    for metric, label, better in comparison_metrics:
        dd_val  = dd_eval[metric].mean()
        std_val = std_eval[metric].mean()

        if better == 'lower':
            # Lower is better (e.g., outage)
            pct = (std_val - dd_val) / max(std_val, 1e-8) * 100
            arrow = '↓' if dd_val < std_val else '↑'
        else:
            # Higher is better (e.g., hit rate)
            pct = (dd_val - std_val) / max(std_val, 1e-8) * 100
            arrow = '↑' if dd_val > std_val else '↓'

        print(f"  {label:<25} {dd_val:>10.4f}   {std_val:>10.4f}   "
              f"{arrow} {abs(pct):>8.1f}%")

    # Final training loss comparison
    dd_final_loss  = dd_history['avg_loss'].iloc[-10:].mean() if 'avg_loss' in dd_history.columns else 0
    std_final_loss = std_history['avg_loss'].iloc[-10:].mean() if 'avg_loss' in std_history.columns else 0
    print(f"\n  {'Final Avg Loss':<25} {dd_final_loss:>10.4f}   {std_final_loss:>10.4f}")

    print(f"\n  Training time:")
    print(f"    Double Dueling DQN : {t_dd:.1f}s")
    print(f"    Standard DQN       : {t_std:.1f}s")
    print(f"    Total              : {elapsed:.0f}s ({elapsed/60:.1f} min)")

    print(f"\n  Output directory: {os.path.abspath(save_dir)}")
    print(f"    {p1}")
    print(f"    {p2}")

    print(f"\n{'#' * 70}")
    print("#" + " " * 30 + "DONE" + " " * 28 + "#")
    print(f"{'#' * 70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare Double Dueling DQN vs Standard DQN "
                    "for cache-aided NOMA")
    parser.add_argument(
        '--debug', action='store_true',
        help='Fast smoke test (~3-5 min on CPU)')
    args = parser.parse_args()
    main(debug=args.debug)
