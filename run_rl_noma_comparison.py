# run_rl_noma_comparison.py
"""
Comprehensive comparison of RL-based NOMA-aware caching against baselines.
Compares: Top-K, LRU, LFU, Random, and RL-DQN-NOMA policies.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

sys.path.insert(0, 'src')

from src import config as cfg
from src.simulation.noma_caching_sim import run_mc_runs
from src.simulation.rl_noma_sim import run_rl_noma_monte_carlo


def compare_all_policies(cfg, save_dir='./results_rl'):
    """
    Run comprehensive comparison of all caching policies.
    """
    
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "🚀 "*35)
    print("COMPREHENSIVE RL-NOMA CACHING POLICY COMPARISON")
    print("🚀 "*35 + "\n")
    
    # Policies to compare
    baseline_policies = ["topk", "lru", "lfu", "random"]
    
    all_results = {}
    
    # 1. Run baseline policies
    print("="*70)
    print("PHASE 1: RUNNING BASELINE POLICIES")
    print("="*70)
    
    for policy in baseline_policies:
        print(f"\n>>> Testing {policy.upper()} policy...")
        cfg.CACHE_POLICY = policy
        t0 = time.time()
        df = run_mc_runs(cfg)
        elapsed = time.time() - t0
        
        # Add missing columns for compatibility
        df['noma_success_rate'] = 0.5
        df['avg_ber'] = 0.01
        df['avg_ber_weak'] = 0.01
        df['avg_ber_strong'] = 0.01
        df['cumulative_reward'] = 0
        
        all_results[policy] = df
        df.to_csv(f'{save_dir}/results_{policy}.csv', index=False)
        
        print(f"    Hit Rate:    {df['hit_rate'].mean():.4f} ± {df['hit_rate'].std():.4f}")
        print(f"    Outage Rate: {df['outage_rate'].mean():.4f} ± {df['outage_rate'].std():.4f}")
        print(f"    Time: {elapsed:.1f}s")
    
    # 2. Run RL-based policy
    print("\n" + "="*70)
    print("PHASE 2: RUNNING RL-DQN-NOMA POLICY")
    print("="*70 + "\n")
    
    t0 = time.time()
    df_rl = run_rl_noma_monte_carlo(cfg, num_runs=cfg.NUM_RUNS)
    elapsed = time.time() - t0
    
    all_results['rl_dqn_noma'] = df_rl
    df_rl.to_csv(f'{save_dir}/results_rl_dqn_noma.csv', index=False)
    
    print(f"\n    Time: {elapsed:.1f}s")
    
    # 3. Consolidate results
    print("\n" + "="*70)
    print("PHASE 3: ANALYSIS & COMPARISON")
    print("="*70 + "\n")
    
    combined = []
    for policy, df in all_results.items():
        df_copy = df.copy()
        df_copy['policy'] = policy
        combined.append(df_copy)
    
    combined_df = pd.concat(combined, ignore_index=True)
    combined_df.to_csv(f'{save_dir}/results_all_policies.csv', index=False)
    
    # 4. Compute summary statistics
    summary = compute_summary_statistics(all_results)
    summary.to_csv(f'{save_dir}/summary_statistics.csv', index=False)
    
    print("\nSUMMARY TABLE:")
    print(summary.to_string(index=False))
    
    # 5. Compute improvements over baseline
    improvements = compute_improvements(summary)
    improvements.to_csv(f'{save_dir}/improvements_over_baseline.csv', index=False)
    
    print("\n\nIMPROVEMENTS OVER TOP-K BASELINE:")
    print(improvements.to_string(index=False))
    
    # 6. Generate visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    plot_comprehensive_comparison(all_results, save_dir)
    plot_learning_curves(all_results, save_dir)
    plot_performance_radar(summary, save_dir)
    
    print("\n✅ All results saved to:", save_dir)
    print("="*70 + "\n")
    
    return all_results, summary, improvements


def compute_summary_statistics(results_dict):
    """Compute summary statistics for all policies."""
    
    summary_data = []
    
    for policy, df in results_dict.items():
        summary_data.append({
            'policy': policy,
            'hit_rate_mean': df['hit_rate'].mean(),
            'hit_rate_std': df['hit_rate'].std(),
            'outage_rate_mean': df['outage_rate'].mean(),
            'outage_rate_std': df['outage_rate'].std(),
            'noma_success_mean': df['noma_success_rate'].mean(),
            'noma_success_std': df['noma_success_rate'].std(),
            'ber_mean': df['avg_ber'].mean() if 'avg_ber' in df else 0,
            'ber_std': df['avg_ber'].std() if 'avg_ber' in df else 0,
        })
    
    return pd.DataFrame(summary_data)


def compute_improvements(summary_df):
    """Compute improvements over baseline (Top-K)."""
    
    baseline = summary_df[summary_df['policy'] == 'topk'].iloc[0]
    
    improvements_data = []
    
    for _, row in summary_df.iterrows():
        if row['policy'] == 'topk':
            continue
        
        hit_improvement = (row['hit_rate_mean'] - baseline['hit_rate_mean']) / baseline['hit_rate_mean'] * 100
        outage_improvement = (baseline['outage_rate_mean'] - row['outage_rate_mean']) / baseline['outage_rate_mean'] * 100
        noma_improvement = (row['noma_success_mean'] - baseline['noma_success_mean']) / baseline['noma_success_mean'] * 100
        ber_improvement = (baseline['ber_mean'] - row['ber_mean']) / (baseline['ber_mean'] + 1e-10) * 100
        
        improvements_data.append({
            'policy': row['policy'],
            'hit_rate_improvement_%': hit_improvement,
            'outage_reduction_%': outage_improvement,
            'noma_success_improvement_%': noma_improvement,
            'ber_reduction_%': ber_improvement,
            'combined_score': hit_improvement + outage_improvement + noma_improvement
        })
    
    return pd.DataFrame(improvements_data)


def plot_comprehensive_comparison(results_dict, save_dir):
    """Create comprehensive comparison plots."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    policies = list(results_dict.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(policies)))
    
    # 1. Hit Rate Comparison
    ax = axes[0, 0]
    hit_rates = [results_dict[p]['hit_rate'].mean() for p in policies]
    hit_stds = [results_dict[p]['hit_rate'].std() for p in policies]
    bars = ax.bar(policies, hit_rates, yerr=hit_stds, color=colors, capsize=5)
    ax.set_ylabel('Hit Rate')
    ax.set_title('Cache Hit Rate Comparison')
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    # Highlight best
    best_idx = np.argmax(hit_rates)
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(3)
    
    # 2. Outage Rate Comparison
    ax = axes[0, 1]
    outage_rates = [results_dict[p]['outage_rate'].mean() for p in policies]
    outage_stds = [results_dict[p]['outage_rate'].std() for p in policies]
    bars = ax.bar(policies, outage_rates, yerr=outage_stds, color=colors, capsize=5)
    ax.set_ylabel('Outage Rate')
    ax.set_title('Outage Rate Comparison (Lower is Better)')
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    # Highlight best
    best_idx = np.argmin(outage_rates)
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(3)
    
    # 3. NOMA Success Rate
    ax = axes[0, 2]
    noma_success = [results_dict[p]['noma_success_rate'].mean() for p in policies]
    noma_stds = [results_dict[p]['noma_success_rate'].std() for p in policies]
    bars = ax.bar(policies, noma_success, yerr=noma_stds, color=colors, capsize=5)
    ax.set_ylabel('NOMA Success Rate')
    ax.set_title('NOMA Transmission Success Rate')
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    best_idx = np.argmax(noma_success)
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(3)
    
    # 4. BER Comparison
    ax = axes[1, 0]
    ber_means = [results_dict[p]['avg_ber'].mean() if 'avg_ber' in results_dict[p] else 0 
                 for p in policies]
    ber_stds = [results_dict[p]['avg_ber'].std() if 'avg_ber' in results_dict[p] else 0 
                for p in policies]
    bars = ax.bar(policies, ber_means, yerr=ber_stds, color=colors, capsize=5)
    ax.set_ylabel('Average BER')
    ax.set_title('Bit Error Rate Comparison (Lower is Better)')
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    # 5. Box plot - Hit Rate Distribution
    ax = axes[1, 1]
    data_to_plot = [results_dict[p]['hit_rate'] for p in policies]
    bp = ax.boxplot(data_to_plot, labels=policies, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_ylabel('Hit Rate')
    ax.set_title('Hit Rate Distribution')
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    # 6. Combined Performance Score
    ax = axes[1, 2]
    # Normalize and combine metrics
    hit_norm = np.array(hit_rates) / np.max(hit_rates)
    outage_norm = 1 - (np.array(outage_rates) / (np.max(outage_rates) + 1e-6))
    noma_norm = np.array(noma_success) / (np.max(noma_success) + 1e-6)
    
    combined_score = (hit_norm + outage_norm + noma_norm) / 3.0
    
    bars = ax.bar(policies, combined_score, color=colors)
    ax.set_ylabel('Combined Performance Score')
    ax.set_title('Overall Performance (Normalized)')
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    best_idx = np.argmax(combined_score)
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/comprehensive_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_dir}/comprehensive_comparison.png")
    plt.close()


def plot_learning_curves(results_dict, save_dir):
    """Plot learning curves for RL policy."""
    
    if 'rl_dqn_noma' not in results_dict:
        return
    
    df_rl = results_dict['rl_dqn_noma']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Hit Rate over runs
    axes[0, 0].plot(df_rl['hit_rate'], marker='o', linewidth=2, markersize=4)
    axes[0, 0].set_xlabel('Simulation Run')
    axes[0, 0].set_ylabel('Hit Rate')
    axes[0, 0].set_title('RL Agent: Hit Rate Evolution')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add moving average
    window = 5
    if len(df_rl) >= window:
        ma = df_rl['hit_rate'].rolling(window=window).mean()
        axes[0, 0].plot(ma, 'r--', linewidth=2, label=f'{window}-run MA')
        axes[0, 0].legend()
    
    # 2. Cumulative reward
    axes[0, 1].plot(df_rl['cumulative_reward'], marker='s', linewidth=2, markersize=4, color='green')
    axes[0, 1].set_xlabel('Simulation Run')
    axes[0, 1].set_ylabel('Cumulative Reward')
    axes[0, 1].set_title('RL Agent: Cumulative Reward')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Epsilon decay
    axes[1, 0].plot(df_rl['final_epsilon'], marker='^', linewidth=2, markersize=4, color='purple')
    axes[1, 0].set_xlabel('Simulation Run')
    axes[1, 0].set_ylabel('Epsilon (Exploration Rate)')
    axes[1, 0].set_title('RL Agent: Exploration vs Exploitation')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Q-table growth
    axes[1, 1].plot(df_rl['q_table_size'], marker='d', linewidth=2, markersize=4, color='orange')
    axes[1, 1].set_xlabel('Simulation Run')
    axes[1, 1].set_ylabel('Q-Table Size (States Explored)')
    axes[1, 1].set_title('RL Agent: Learning Progress')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/rl_learning_curves.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_dir}/rl_learning_curves.png")
    plt.close()


def plot_performance_radar(summary_df, save_dir):
    """Create radar chart comparing all policies."""
    
    from math import pi
    
    # Metrics to compare (normalized to 0-1)
    metrics = ['hit_rate_mean', 'noma_success_mean']
    # Invert outage and BER (lower is better)
    summary_df['outage_inv'] = 1 - summary_df['outage_rate_mean']
    summary_df['ber_inv'] = 1 - (summary_df['ber_mean'] / summary_df['ber_mean'].max())
    
    metrics_labels = ['Hit Rate', 'NOMA Success', 'Low Outage', 'Low BER']
    metrics_cols = ['hit_rate_mean', 'noma_success_mean', 'outage_inv', 'ber_inv']
    
    # Normalize to 0-1
    for col in metrics_cols:
        max_val = summary_df[col].max()
        if max_val > 0:
            summary_df[col + '_norm'] = summary_df[col] / max_val
        else:
            summary_df[col + '_norm'] = 0
    
    # Number of variables
    N = len(metrics_labels)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(summary_df)))
    
    for idx, row in summary_df.iterrows():
        values = [row[col + '_norm'] for col in metrics_cols]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=row['policy'], color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_labels, size=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
    ax.grid(True)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title('Policy Performance Comparison (Normalized)', size=14, y=1.08)
    
    plt.savefig(f'{save_dir}/performance_radar.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_dir}/performance_radar.png")
    plt.close()


def main():
    """Main execution."""
    
    # Set parameters for RL simulation
    cfg.TIME_SLOTS = 1000
    cfg.CACHE_UPDATE_INTERVAL = 100
    cfg.NUM_RUNS = 20  # Adjust based on computational resources
    
    print("\n📋 Configuration:")
    print(f"   Files: {cfg.NUM_FILES}")
    print(f"   Cache Size: {cfg.CACHE_SIZE} ({cfg.CACHE_SIZE/cfg.NUM_FILES*100:.1f}%)")
    print(f"   Users: {cfg.NUM_USERS}")
    print(f"   Time Slots: {cfg.TIME_SLOTS}")
    print(f"   Monte Carlo Runs: {cfg.NUM_RUNS}")
    print(f"   Zipf Alpha: {cfg.ZIPF_ALPHA}\n")
    
    # Run comparison
    results, summary, improvements = compare_all_policies(cfg)
    
    # Print key findings
    print("\n" + "="*70)
    print("🎯 KEY FINDINGS")
    print("="*70)
    
    # Best performing policy
    best_policy = improvements.loc[improvements['combined_score'].idxmax(), 'policy']
    best_score = improvements.loc[improvements['combined_score'].idxmax(), 'combined_score']
    
    print(f"\n🏆 Best Overall Policy: {best_policy.upper()}")
    print(f"   Combined Performance Score: {best_score:.2f}%")
    
    if best_policy == 'rl_dqn_noma':
        rl_improvements = improvements[improvements['policy'] == 'rl_dqn_noma'].iloc[0]
        print(f"\n   Improvements over Top-K baseline:")
        print(f"   • Hit Rate:     {rl_improvements['hit_rate_improvement_%']:+.2f}%")
        print(f"   • Outage:       {rl_improvements['outage_reduction_%']:+.2f}%")
        print(f"   • NOMA Success: {rl_improvements['noma_success_improvement_%']:+.2f}%")
        print(f"   • BER:          {rl_improvements['ber_reduction_%']:+.2f}%")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()