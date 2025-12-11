#!/usr/bin/env python3
"""
DQN Training and Evaluation Script

This script properly trains DQN cache before evaluation:
1. Train for RL_TRAINING_EPISODES (50) episodes
2. Switch to evaluation mode
3. Evaluate on NUM_RUNS (100) test episodes
4. Compare with baseline policies

Author: Cache-Aided NOMA Team
Date: December 2025
"""

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from src import config as cfg
from src.caching import create_cache
from src.simulation.noma_caching_sim import (
    NOMACachingSimulator,
    plot_comparison_results,
    plot_dqn_training
)


def train_dqn(cfg):
    """
    Train DQN cache with proper training workflow.
    
    Returns:
        trained_cache: Trained DQN cache instance
        training_df: DataFrame with training history
    """
    print("\n" + "="*70)
    print("PHASE 1: DQN TRAINING")
    print("="*70)
    print(f"Training for {cfg.RL_TRAINING_EPISODES} episodes...\n")
    
    # Create DQN cache
    dqn_cache = create_cache(
        'dqn',
        capacity=cfg.CACHE_SIZE,
        num_files=cfg.NUM_FILES,
        num_users=cfg.NUM_USERS,
        learning_rate=cfg.RL_LEARNING_RATE,
        gamma=cfg.RL_GAMMA,
        epsilon_start=cfg.RL_EPSILON_START,
        epsilon_end=cfg.RL_EPSILON_END,
        epsilon_decay_steps=cfg.RL_EPSILON_DECAY_STEPS,
        batch_size=cfg.RL_BATCH_SIZE,
        replay_buffer_size=cfg.RL_REPLAY_BUFFER_SIZE,
        use_prioritized_replay=cfg.RL_USE_PRIORITIZED_REPLAY,
        priority_alpha=cfg.RL_PRIORITY_ALPHA,
        priority_beta_start=cfg.RL_PRIORITY_BETA_START,
        priority_beta_end=cfg.RL_PRIORITY_BETA_END,
        seed=cfg.RANDOM_SEED
    )
    
    simulator = NOMACachingSimulator(cfg)
    training_history = []
    
    # Training loop
    for episode in range(cfg.RL_TRAINING_EPISODES):
        seed = cfg.RANDOM_SEED + episode + 1000  # Different seeds for training
        episode_done = (episode == cfg.RL_TRAINING_EPISODES - 1)
        
        # Run training episode
        results = simulator.run_single_episode(dqn_cache, seed, episode_done)
        
        # Get DQN stats
        dqn_stats = dqn_cache.get_stats()
        results.update(dqn_stats)
        results['episode'] = episode
        
        training_history.append(results)
        
        # Print progress
        if (episode + 1) % 10 == 0 or episode == 0:
            print(f"  Episode {episode+1:3d}/{cfg.RL_TRAINING_EPISODES}: "
                  f"Hit={results['hit_rate']:.3f}, "
                  f"CIC={results['cic_benefit_rate']:.3f}, "
                  f"ε={dqn_stats['epsilon']:.3f}, "
                  f"Loss={dqn_stats.get('avg_loss', 0):.4f}")
    
    print("\n✅ Training complete!")
    print(f"   Final hit rate: {training_history[-1]['hit_rate']:.3f}")
    print(f"   Final CIC benefit: {training_history[-1]['cic_benefit_rate']:.3f}")
    print(f"   Final epsilon: {dqn_stats['epsilon']:.3f}")
    
    return dqn_cache, pd.DataFrame(training_history)


def evaluate_dqn(dqn_cache, cfg, num_runs=None):
    """
    Evaluate trained DQN cache.
    
    Args:
        dqn_cache: Trained DQN cache instance
        cfg: Configuration
        num_runs: Number of evaluation runs (default: cfg.NUM_RUNS)
    
    Returns:
        DataFrame with evaluation results
    """
    if num_runs is None:
        num_runs = cfg.NUM_RUNS
    
    print("\n" + "="*70)
    print("PHASE 2: DQN EVALUATION")
    print("="*70)
    print(f"Evaluating for {num_runs} episodes...\n")
    
    # Switch to evaluation mode (no exploration, no training)
    dqn_cache.set_eval_mode(True)
    
    simulator = NOMACachingSimulator(cfg)
    eval_results = []
    
    for run in range(num_runs):
        # Use different seed range for evaluation
        seed = cfg.RANDOM_SEED + 50000 + run
        
        # Run evaluation episode
        results = simulator.run_single_episode(dqn_cache, seed, episode_done=False)
        results['policy'] = 'dqn'
        results['run'] = run
        results['seed'] = seed
        
        eval_results.append(results)
        
        # Print progress
        if (run + 1) % 20 == 0 or run == 0:
            print(f"  Run {run+1:3d}/{num_runs}: "
                  f"Hit={results['hit_rate']:.3f}, "
                  f"Outage={results['outage_probability']:.3f}, "
                  f"CIC={results['cic_benefit_rate']:.3f}")
    
    return pd.DataFrame(eval_results)


def evaluate_baselines(cfg, num_runs=None):
    """
    Evaluate baseline policies for comparison.
    
    Returns:
        DataFrame with baseline results
    """
    if num_runs is None:
        num_runs = cfg.NUM_RUNS
    
    print("\n" + "="*70)
    print("PHASE 3: BASELINE COMPARISON")
    print("="*70)
    
    policies = ['topk', 'lru', 'lfu', 'random']
    all_results = []
    
    for policy in policies:
        print(f"\nTesting {policy.upper()} policy...")
        
        cache = create_cache(policy, capacity=cfg.CACHE_SIZE)
        simulator = NOMACachingSimulator(cfg)
        
        for run in range(num_runs):
            seed = cfg.RANDOM_SEED + 50000 + run  # Same seeds as DQN eval
            
            results = simulator.run_single_episode(cache, seed, episode_done=False)
            results['policy'] = policy
            results['run'] = run
            results['seed'] = seed
            
            all_results.append(results)
        
        # Print summary for this policy
        policy_results = [r for r in all_results if r['policy'] == policy]
        avg_hit = np.mean([r['hit_rate'] for r in policy_results])
        avg_cic = np.mean([r['cic_benefit_rate'] for r in policy_results])
        avg_outage = np.mean([r['outage_probability'] for r in policy_results])
        
        print(f"  {policy.upper()}: Hit={avg_hit:.3f}, CIC={avg_cic:.3f}, Outage={avg_outage:.3f}")
    
    return pd.DataFrame(all_results)


def main():
    """
    Main execution: Train DQN, then compare with baselines.
    """
    print("\n" + "#"*70)
    print("#" + " "*10 + "DQN TRAINING & EVALUATION PIPELINE" + " "*14 + "#")
    print("#"*70)
    
    print("\nConfiguration:")
    print(f"  Cache size: {cfg.CACHE_SIZE}")
    print(f"  Num files: {cfg.NUM_FILES}")
    print(f"  Num users: {cfg.NUM_USERS}")
    print(f"  Training episodes: {cfg.RL_TRAINING_EPISODES}")
    print(f"  Evaluation runs: {cfg.NUM_RUNS}")
    print(f"  CIC reward: {cfg.RL_REWARD_CIC_ENABLED}")
    
    t0 = time.time()
    
    # Phase 1: Train DQN
    dqn_cache, training_df = train_dqn(cfg)
    
    # Save training history
    import os
    os.makedirs('results', exist_ok=True)
    training_df.to_csv('results/dqn_training_history.csv', index=False)
    print("\n✅ Training history saved to results/dqn_training_history.csv")
    
    # Plot training curves
    plot_dqn_training(training_df, 'results/dqn_training_curves.png')
    
    # Save trained model
    os.makedirs('checkpoints', exist_ok=True)
    dqn_cache.save_model('checkpoints/dqn_trained.pth')
    print("✅ Trained model saved to checkpoints/dqn_trained.pth")
    
    # Phase 2: Evaluate DQN
    dqn_eval_df = evaluate_dqn(dqn_cache, cfg)
    
    # Phase 3: Evaluate baselines
    baseline_df = evaluate_baselines(cfg)
    
    # Combine results
    all_results_df = pd.concat([dqn_eval_df, baseline_df], ignore_index=True)
    
    # Aggregate statistics
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70 + "\n")
    
    summary = all_results_df.groupby('policy').agg({
        'hit_rate': ['mean', 'std'],
        'outage_probability': ['mean', 'std'],
        'cic_benefit_rate': ['mean', 'std'],
        'spectral_efficiency': ['mean', 'std']
    }).round(4)
    
    print(summary)
    
    # Highlight key comparisons
    print("\n" + "-"*70)
    print("KEY FINDINGS:")
    print("-"*70)
    
    policies = ['dqn', 'lru', 'lfu', 'topk']
    for policy in policies:
        policy_data = all_results_df[all_results_df['policy'] == policy]
        if len(policy_data) > 0:
            hit = policy_data['hit_rate'].mean()
            cic = policy_data['cic_benefit_rate'].mean()
            outage = policy_data['outage_probability'].mean()
            print(f"{policy.upper():8s}: Hit={hit:.1%}, CIC={cic:.1%}, Outage={outage:.1%}")
    
    # Save results
    all_results_df.to_csv('results/comparison_results.csv', index=False)
    print("\n✅ Results saved to results/comparison_results.csv")
    
    # Plot comparison
    plot_comparison_results(all_results_df, 'results/comparison_plots.png')
    
    print(f"\n⏱️  Total time: {time.time() - t0:.1f}s")
    
    print("\n" + "#"*70)
    print("#" + " "*22 + "COMPLETE" + " "*28 + "#")
    print("#"*70 + "\n")


if __name__ == "__main__":
    main()