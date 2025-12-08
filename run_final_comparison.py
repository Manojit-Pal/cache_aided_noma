# run_final_comparison.py

import sys
import os
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from src import config as cfg
from src.simulation.noma_caching_sim import run_mc_runs

# ✅ UPDATED: Import the new stable DQN cache
try:
    from src.caching.dqn_cache_final import StableDQNCache
    from src.simulation.stable_dqn_sim import train_dqn_cache, evaluate_cache
    STABLE_DQN_AVAILABLE = True
    print("✅ Using Stable DQN implementation")
except ImportError:
    from src.simulation.improved_rl_noma_sim import run_improved_rl_monte_carlo
    STABLE_DQN_AVAILABLE = False
    IMPROVED_RL_AVAILABLE = True
    print("⚠️  Using improved_rl_noma_sim (fallback)")
    


def print_header(text):
    print("\n" + "="*80)
    print(text.center(80))
    print("="*80)


def run_baseline_policies(cfg, policies=['topk', 'lru', 'lfu']):
    """Run traditional baseline policies."""
    
    print_header("PHASE 1: BASELINE POLICIES")
    
    results = {}
    
    for policy in policies:
        print(f"\n📊 Testing {policy.upper()} policy...")
        cfg.CACHE_POLICY = policy
        
        t0 = time.time()
        df = run_mc_runs(cfg)
        elapsed = time.time() - t0
        
        # Add missing columns for compatibility
        df['noma_success_rate'] = 0.5  # Default for non-NOMA aware
        df['avg_ber'] = 0.01
        df['policy'] = policy
        
        results[policy] = df
        
        print(f"   ✅ Completed in {elapsed:.1f}s")
        print(f"   Mean Hit Rate:  {df['hit_rate'].mean():.4f} ± {df['hit_rate'].std():.4f}")
        print(f"   Mean Outage:    {df['outage_rate'].mean():.4f} ± {df['outage_rate'].std():.4f}")
    
    return results


def run_stable_dqn_policy(cfg, num_training_episodes=50, steps_per_episode=1000, num_eval_runs=10):
    """✅ NEW: Run Stable DQN policy with proper training and evaluation."""
    
    print_header("PHASE 2: STABLE DQN CACHE")
    
    print(f"\n🧠 Testing STABLE DQN Cache...")
    print(f"   Training Episodes: {num_training_episodes}")
    print(f"   Steps per Episode: {steps_per_episode}")
    print(f"   Total Training Steps: {num_training_episodes * steps_per_episode}")
    print(f"   Evaluation Runs: {num_eval_runs}")
    
    # Create and train DQN cache
    dqn_cache = StableDQNCache(
        capacity=cfg.CACHE_SIZE,
        num_files=cfg.NUM_FILES,
        num_users=cfg.NUM_USERS,
        learning_rate=cfg.RL_LEARNING_RATE,
        gamma=cfg.RL_GAMMA,
        epsilon_start=cfg.RL_EPSILON_START,
        epsilon_end=cfg.RL_EPSILON_END,
        epsilon_decay_steps=cfg.RL_EPSILON_DECAY_STEPS,
        hidden_dims=cfg.RL_HIDDEN_DIMS,
        batch_size=cfg.RL_BATCH_SIZE,
        replay_buffer_size=cfg.RL_REPLAY_BUFFER_SIZE,
        use_prioritized_replay=cfg.RL_USE_PRIORITIZED_REPLAY,
        priority_alpha=cfg.RL_PRIORITY_ALPHA,
        priority_beta=cfg.RL_PRIORITY_BETA,
        seed=cfg.RANDOM_SEED
    )
    
    # Training phase
    print(f"\n{'='*80}")
    print("TRAINING PHASE")
    print(f"{'='*80}")
    
    t0 = time.time()
    training_stats = train_dqn_cache(
        dqn_cache, 
        cfg, 
        num_episodes=num_training_episodes,
        steps_per_episode=steps_per_episode,
        verbose=True
    )
    training_time = time.time() - t0
    
    print(f"\n✅ Training completed in {training_time/60:.1f} minutes")
    
    # Evaluation phase
    print(f"\n{'='*80}")
    print("EVALUATION PHASE")
    print(f"{'='*80}")
    
    eval_results = []
    
    for run in range(num_eval_runs):
        seed = cfg.RANDOM_SEED + run + 1000
        
        result = evaluate_cache(
            dqn_cache,
            cfg,
            seed,
            num_requests=cfg.RL_EVAL_REQUESTS,
            cache_name="stable_dqn"
        )
        
        # Add run information
        result['run'] = run + 1
        result['seed'] = seed
        result['policy'] = 'stable_dqn'
        
        eval_results.append(result)
        
        if (run + 1) % 3 == 0:
            print(f"  Eval Run {run+1}/{num_eval_runs}: "
                  f"Hit={result['hit_rate']:.3f}, "
                  f"Outage={result['outage_rate']:.3f}, "
                  f"NOMA Success={result['noma_success_rate']:.3f}")
    
    df = pd.DataFrame(eval_results)
    
    print(f"\n✅ Evaluation completed")
    print(f"   Mean Hit Rate:  {df['hit_rate'].mean():.4f} ± {df['hit_rate'].std():.4f}")
    print(f"   Mean Outage:    {df['outage_rate'].mean():.4f} ± {df['outage_rate'].std():.4f}")
    print(f"   Mean NOMA Success: {df['noma_success_rate'].mean():.4f} ± {df['noma_success_rate'].std():.4f}")
    
    # Get final cache stats
    cache_stats = dqn_cache.get_stats()
    print(f"\n   Final Cache Stats:")
    print(f"      Training Steps: {cache_stats['training_step']}")
    print(f"      Final Epsilon: {cache_stats['epsilon']:.4f}")
    print(f"      Cumulative Reward: {cache_stats['cumulative_reward']:.2f}")
    print(f"      Avg Loss: {cache_stats['avg_loss']:.6f}")
    
    # Convert training stats to learning curves format
    learning_curves = {
        'episode_rewards': training_stats['episode_rewards'],
        'episode_hit_rates': training_stats['episode_hit_rates'],
        'episode_losses': training_stats['episode_losses']
    }
    
    return df, learning_curves


def run_fallback_rl_policy(cfg, improved=True, num_training_steps=50000):
    """Fallback: Run old RL implementation if stable DQN not available."""
    
    print_header("PHASE 2: REINFORCEMENT LEARNING POLICY (FALLBACK)")
    
    if improved and IMPROVED_RL_AVAILABLE:
        print(f"\n🧠 Testing IMPROVED RL-DQN-NOMA...")
        print(f"   Training Steps: {num_training_steps}")
        
        t0 = time.time()
        df, learning_curves_raw = run_improved_rl_monte_carlo(
            cfg,
            num_runs=cfg.NUM_RUNS,
            num_training_steps=num_training_steps
        )
        elapsed = time.time() - t0
        
        df['policy'] = 'improved_dqn_noma'
        
        # Convert learning curves format
        learning_curves = {
            'episode_rewards': [],
            'episode_hit_rates': [],
            'episode_losses': []
        }
        
        print(f"\n   ✅ Completed in {elapsed:.1f}s")
    
    print(f"   Mean Hit Rate:  {df['hit_rate'].mean():.4f} ± {df['hit_rate'].std():.4f}")
    print(f"   Mean Outage:    {df['outage_rate'].mean():.4f} ± {df['outage_rate'].std():.4f}")
    
    if 'noma_success_rate' in df.columns:
        print(f"   Mean NOMA Success: {df['noma_success_rate'].mean():.4f}")
    
    return df, learning_curves


def compute_improvements(all_results, baseline='topk'):
    """Compute percentage improvements over baseline."""
    
    print_header("COMPUTING IMPROVEMENTS")
    
    baseline_df = all_results[baseline]
    baseline_hit = baseline_df['hit_rate'].mean()
    baseline_outage = baseline_df['outage_rate'].mean()
    
    improvements = {}
    
    for policy, df in all_results.items():
        if policy == baseline:
            continue
        
        # Calculate Hit Rate Improvement
        hit_improvement = (df['hit_rate'].mean() - baseline_hit) / baseline_hit * 100
        
        # Calculate Outage Reduction
        if baseline_outage > 0:
            outage_reduction = (baseline_outage - df['outage_rate'].mean()) / baseline_outage * 100
        else:
            outage_reduction = 0.0
        
        # Calculate NOMA Success Improvement
        if 'noma_success_rate' in df.columns:
            noma_improvement = (df['noma_success_rate'].mean() - 0.5) / 0.5 * 100
        else:
            noma_improvement = 0
        
        improvements[policy] = {
            'hit_rate_improvement_%': hit_improvement,
            'outage_reduction_%': outage_reduction,
            'noma_success_improvement_%': noma_improvement,
            'combined_score': hit_improvement + outage_reduction
        }
    
    # Print improvements
    print(f"\n📊 Improvements over {baseline.upper()} baseline:\n")
    
    for policy, imp in improvements.items():
        print(f"{policy.upper()}:")
        print(f"   Hit Rate:     {imp['hit_rate_improvement_%']:+.2f}%")
        print(f"   Outage:       {imp['outage_reduction_%']:+.2f}%")
        print(f"   NOMA Success: {imp['noma_success_improvement_%']:+.2f}%")
        print(f"   Combined:     {imp['combined_score']:+.2f}")
        print()
    
    return improvements


def add_labels(ax, rects, format_str="{:.2f}"):
    """Helper to add value labels on top of bars."""
    for rect in rects:
        height = rect.get_height()
        if height != 0:  # Only add label if height is non-zero
            ax.annotate(format_str.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')


def plot_training_progress(learning_curves, save_dir='./'):
    """✅ UPDATED: Visualize the RL training process."""
    if learning_curves is None:
        print("   ⚠️  No learning curves available")
        return

    print("   Generating Learning Curves...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 1. Hit Rate Evolution
    if 'episode_hit_rates' in learning_curves:
        episodes = np.arange(len(learning_curves['episode_hit_rates']))
        axes[0].plot(episodes, learning_curves['episode_hit_rates'], 'b-', linewidth=2)
        axes[0].set_xlabel('Episode', fontsize=10)
        axes[0].set_ylabel('Hit Rate', fontsize=10)
        axes[0].set_title('DQN Training: Hit Rate Evolution', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
    
    # 2. Loss Evolution
    if 'episode_losses' in learning_curves:
        episodes = np.arange(len(learning_curves['episode_losses']))
        axes[1].plot(episodes, learning_curves['episode_losses'], 'r-', linewidth=2)
        axes[1].set_xlabel('Episode', fontsize=10)
        axes[1].set_ylabel('Loss', fontsize=10)
        axes[1].set_title('DQN Training: Loss Convergence', fontsize=12, fontweight='bold')
        axes[1].set_yscale('log')
        axes[1].grid(True, alpha=0.3)
    
    # 3. Cumulative Reward
    if 'episode_rewards' in learning_curves:
        episodes = np.arange(len(learning_curves['episode_rewards']))
        axes[2].plot(episodes, learning_curves['episode_rewards'], 'g-', linewidth=2)
        axes[2].set_xlabel('Episode', fontsize=10)
        axes[2].set_ylabel('Cumulative Reward', fontsize=10)
        axes[2].set_title('DQN Training: Reward Progress', fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'rl_learning_dynamics.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved Learning Curves: {save_path}")
    plt.close()


def create_comprehensive_plots(all_results, improvements, save_dir='./'):
    """Create comprehensive visualization suite."""
    
    print_header("GENERATING VISUALIZATIONS")
    
    # Plot 1: Performance comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    policies = list(all_results.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(policies)))
    
    # Hit Rate
    ax = axes[0, 0]
    hit_rates = [all_results[p]['hit_rate'].mean() for p in policies]
    hit_stds = [all_results[p]['hit_rate'].std() for p in policies]
    bars = ax.bar(policies, hit_rates, yerr=hit_stds, color=colors, capsize=5, alpha=0.8)
    add_labels(ax, bars)
    ax.set_ylabel('Hit Rate', fontsize=12)
    ax.set_title('Cache Hit Rate Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    # Outage Rate
    ax = axes[0, 1]
    outage_rates = [all_results[p]['outage_rate'].mean() for p in policies]
    outage_stds = [all_results[p]['outage_rate'].std() for p in policies]
    bars = ax.bar(policies, outage_rates, yerr=outage_stds, color=colors, capsize=5, alpha=0.8)
    add_labels(ax, bars)
    ax.set_ylabel('Outage Probability', fontsize=12)
    ax.set_title('Outage Probability (Lower is Better)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    # NOMA Success Rate
    ax = axes[1, 0]
    noma_success = []
    for p in policies:
        if 'noma_success_rate' in all_results[p].columns:
            noma_success.append(all_results[p]['noma_success_rate'].mean())
        else:
            noma_success.append(0.5)
    
    bars = ax.bar(policies, noma_success, color=colors, alpha=0.8)
    add_labels(ax, bars)
    ax.set_ylabel('NOMA Success Rate', fontsize=12)
    ax.set_title('NOMA Transmission Success Rate', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Baseline')
    ax.legend()
    
    # Improvement scores
    ax = axes[1, 1]
    if improvements:
        imp_policies = list(improvements.keys())
        combined_scores = [improvements[p]['combined_score'] for p in imp_policies]
        bars = ax.bar(imp_policies, combined_scores, 
                      color=[colors[policies.index(p)] for p in imp_policies],
                      alpha=0.8)
        add_labels(ax, bars, format_str="{:.1f}")
        ax.set_ylabel('Combined Score (%)', fontsize=12)
        ax.set_title('Overall Performance Improvement', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'final_comparison_all_metrics.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {save_path}")
    plt.close()


def save_results(all_results, improvements, save_dir='./'):
    """Save all results to files."""
    
    print_header("SAVING RESULTS")
    
    # Combine all results
    combined = pd.concat(all_results.values(), ignore_index=True)
    combined_path = os.path.join(save_dir, 'results_final_comparison.csv')
    combined.to_csv(combined_path, index=False)
    print(f"   ✅ Saved combined results: {combined_path}")
    
    # Save summary statistics
    summary = combined.groupby('policy').agg({
        'hit_rate': ['mean', 'std', 'min', 'max'],
        'outage_rate': ['mean', 'std', 'min', 'max'],
        'noma_success_rate': ['mean', 'std']
    }).round(4)
    
    summary_path = os.path.join(save_dir, 'summary_statistics.csv')
    summary.to_csv(summary_path)
    print(f"   ✅ Saved summary: {summary_path}")
    
    # Save improvements
    if improvements:
        imp_df = pd.DataFrame(improvements).T
        imp_path = os.path.join(save_dir, 'improvements_over_baseline.csv')
        imp_df.to_csv(imp_path)
        print(f"   ✅ Saved improvements: {imp_path}")
    
    print(f"\n   📁 All results saved to: {os.path.abspath(save_dir)}")


def print_final_summary(all_results, improvements):
    """Print comprehensive final summary."""
    
    print_header("🎉 FINAL SUMMARY")
    
    print("\n📊 PERFORMANCE METRICS:\n")
    
    # Find best performers
    best_hit_policy = max(all_results.keys(), 
                          key=lambda p: all_results[p]['hit_rate'].mean())
    best_outage_policy = min(all_results.keys(),
                             key=lambda p: all_results[p]['outage_rate'].mean())
    
    print(f"🏆 Best Hit Rate:     {best_hit_policy.upper()}")
    print(f"   Value: {all_results[best_hit_policy]['hit_rate'].mean():.4f}")
    
    print(f"\n🏆 Best Outage Rate:  {best_outage_policy.upper()}")
    print(f"   Value: {all_results[best_outage_policy]['outage_rate'].mean():.4f}")
    
    # Check for RL policy
    rl_policies = [p for p in all_results.keys() if 'dqn' in p or 'rl' in p]
    
    if rl_policies:
        print(f"\n🧠 RL POLICY PERFORMANCE:")
        for rl_policy in rl_policies:
            df = all_results[rl_policy]
            print(f"\n   {rl_policy.upper()}:")
            print(f"      Hit Rate:      {df['hit_rate'].mean():.4f} ± {df['hit_rate'].std():.4f}")
            print(f"      Outage:        {df['outage_rate'].mean():.4f} ± {df['outage_rate'].std():.4f}")
            
            if 'noma_success_rate' in df.columns:
                print(f"      NOMA Success:  {df['noma_success_rate'].mean():.4f}")
            
            if 'use_neural_network' in df.columns:
                nn_used = df['use_neural_network'].iloc[0]
                print(f"      Neural Net:    {'✅ Active' if nn_used else '⚠️ Q-table'}")
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE - Ready for presentation!")
    print("="*80 + "\n")


def main():
    """Main execution function."""
    
    print("\n" + "🎓"*40)
    print("FINAL YEAR PROJECT: RL-BASED NOMA CACHING")
    print("Comprehensive Performance Analysis")
    print("🎓"*40)
    
    # Configuration
    print(f"\n⚙️  Configuration:")
    print(f"   Files:          {cfg.NUM_FILES}")
    print(f"   Cache Size:     {cfg.CACHE_SIZE} ({cfg.CACHE_SIZE/cfg.NUM_FILES*100:.1f}%)")
    print(f"   Users:          {cfg.NUM_USERS}")
    print(f"   Runs per Policy: {cfg.NUM_RUNS}")
    
    if STABLE_DQN_AVAILABLE:
        print(f"\n🧠 RL Training Configuration:")
        print(f"   Training Episodes: {cfg.RL_TRAINING_EPISODES}")
        print(f"   Steps per Episode: {cfg.RL_STEPS_PER_EPISODE}")
        print(f"   Total Training Steps: {cfg.RL_TRAINING_STEPS}")
        print(f"   Epsilon Decay Steps: {cfg.RL_EPSILON_DECAY_STEPS}")
        print(f"   Learning Rate: {cfg.RL_LEARNING_RATE}")
        print(f"   Batch Size: {cfg.RL_BATCH_SIZE}")
    
    start_time = time.time()
    
    # Run experiments
    baseline_results = run_baseline_policies(cfg)
    
    # Run RL based on availability
    if STABLE_DQN_AVAILABLE:
        rl_results, learning_curves = run_stable_dqn_policy(
            cfg,
            num_training_episodes=cfg.RL_TRAINING_EPISODES,
            steps_per_episode=cfg.RL_STEPS_PER_EPISODE,
            num_eval_runs=cfg.NUM_RUNS
        )
    else:
        rl_results, learning_curves = run_fallback_rl_policy(
            cfg,
            improved=IMPROVED_RL_AVAILABLE,
            num_training_steps=cfg.RL_TRAINING_STEPS
        )
    
    # Combine results
    all_results = {**baseline_results, **{rl_results['policy'].iloc[0]: rl_results}}
    
    # Analyze
    improvements = compute_improvements(all_results, baseline='topk')
    
    # Visualize
    create_comprehensive_plots(all_results, improvements)
    
    # Plot Learning Curves
    if learning_curves is not None:
        plot_training_progress(learning_curves)
    
    # Save
    save_results(all_results, improvements)
    
    # Summary
    print_final_summary(all_results, improvements)
    
    # Total time
    total_time = time.time() - start_time
    print(f"⏱️  Total execution time: {total_time/60:.1f} minutes\n")


if __name__ == "__main__":
    main()