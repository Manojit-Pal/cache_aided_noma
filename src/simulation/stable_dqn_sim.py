# src/simulation/stable_dqn_sim.py
"""
Simulation runner for Stable DQN Cache in NOMA systems.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from scipy.special import erfc

from src.utils import set_seed, sample_zipf_catalog
from src.noma import channel_model
from src.noma.power_allocation import allocate_power_gridsearch
from src.noma.noma_base import simulate_noma_pair

# Import caches
from src.caching.dqn_cache_final import StableDQNCache
from src.caching.static_cache import StaticTopKCache
from src.caching.dynamic_cache import LRUCache, LFUCache


def compute_ber_bpsk(sinr):
    """Compute BER for BPSK."""
    return 0.5 * erfc(np.sqrt(np.maximum(sinr, 0.0)))


# ============================================================================
# TRAINING PHASE
# ============================================================================

def train_dqn_cache(
    cache: StableDQNCache,
    cfg,
    num_episodes: int = 50,
    steps_per_episode: int = 1000,
    verbose: bool = True
) -> Dict:
    """
    Train DQN cache over multiple episodes.
    
    Args:
        cache: StableDQNCache instance
        cfg: Configuration object
        num_episodes: Number of training episodes
        steps_per_episode: Requests per episode
        verbose: Print progress
    
    Returns:
        Training statistics dictionary
    """
    print(f"\n{'='*70}")
    print("TRAINING PHASE")
    print(f"{'='*70}")
    print(f"Episodes: {num_episodes}")
    print(f"Steps per episode: {steps_per_episode}")
    print(f"Total training steps: {num_episodes * steps_per_episode}")
    
    # Generate training environment
    user_pos = channel_model.generate_user_positions(
        cfg.NUM_USERS, cfg.CELL_RADIUS, seed=cfg.RANDOM_SEED
    )
    distances = user_pos[:, 2]
    pl = np.array([
        channel_model.pathloss(d, cfg.PATHLOSS_EXPONENT, cfg.MIN_DISTANCE)
        for d in distances
    ])
    
    # Training metrics
    episode_rewards = []
    episode_hit_rates = []
    episode_losses = []
    
    total_steps = 0
    start_time = time.time()
    
    for episode in range(num_episodes):
        # Track reward specific to THIS episode
        start_cum_reward = cache.get_stats()['cumulative_reward']
        
        episode_hits = 0
        episode_requests = 0
        
        # Update channel for this episode
        small_scale = channel_model.rayleigh_gain(cfg.NUM_USERS)
        channel_gains = pl * small_scale
        
        for step in range(steps_per_episode):
            # Update channel periodically (mobility)
            if step % 50 == 0:
                small_scale = channel_model.rayleigh_gain(cfg.NUM_USERS)
                channel_gains = pl * small_scale
            
            # Generate request
            file_id = sample_zipf_catalog(cfg.NUM_FILES, cfg.ZIPF_ALPHA, size=1)[0]
            user_id = np.random.choice(cfg.NUM_USERS)
            
            episode_requests += 1
            total_steps += 1
            
            # Check if last step of episode
            is_episode_end = (step == steps_per_episode - 1)
            
            # Check cache
            cache_hit = cache.is_hit(file_id)
            
            if cache_hit:
                episode_hits += 1
                # Observe cache hit
                cache.observe_request(
                    user_id=user_id,
                    file_id=file_id,
                    cache_hit=True,
                    channel_gain=channel_gains[user_id],
                    episode_done=is_episode_end
                )
            else:
                # Cache miss - NOMA transmission
                partner_user = np.random.choice(
                    [u for u in range(cfg.NUM_USERS) if u != user_id]
                )
                
                # Determine weak/strong users
                if channel_gains[user_id] < channel_gains[partner_user]:
                    weak_user, strong_user = user_id, partner_user
                else:
                    weak_user, strong_user = partner_user, user_id
                
                gain_weak = channel_gains[weak_user]
                gain_strong = channel_gains[strong_user]
                
                # Power allocation
                p_w, p_s, feasible, _ = allocate_power_gridsearch(
                    gain_weak, gain_strong, cfg, grid_points=cfg.POWER_ALLOC_GRID
                )
                
                # Simulate NOMA transmission
                weak_success, strong_success, sinr_w, _, sinr_s_after = \
                    simulate_noma_pair(gain_weak, gain_strong, cfg, p_w, p_s)
                
                # Determine outcome for requesting user
                noma_success = weak_success if user_id == weak_user else strong_success
                user_sinr = sinr_w if user_id == weak_user else sinr_s_after
                ber = compute_ber_bpsk(user_sinr)
                outage = not noma_success
                
                # Learn from transmission
                cache.observe_request(
                    user_id=user_id,
                    file_id=file_id,
                    cache_hit=False,
                    noma_success=noma_success,
                    channel_gain=channel_gains[user_id],
                    sinr_weak=sinr_w,
                    sinr_strong=sinr_s_after,
                    ber=ber,
                    outage=outage,
                    episode_done=is_episode_end
                )
        
        # Episode statistics
        stats = cache.get_stats()
        
        # Calculate ACTUAL episode reward (Difference between end and start)
        current_cum_reward = stats['cumulative_reward']
        actual_episode_reward = current_cum_reward - start_cum_reward
        
        episode_hit_rate = episode_hits / episode_requests if episode_requests > 0 else 0
        
        episode_rewards.append(actual_episode_reward)
        episode_hit_rates.append(episode_hit_rate)
        episode_losses.append(stats['avg_loss'])
        
        # Print progress
        if verbose and (episode + 1) % 5 == 0:
            elapsed = time.time() - start_time
            print(f"Episode {episode+1}/{num_episodes} | "
                  f"Hit Rate: {episode_hit_rate:.3f} | "
                  f"Reward: {actual_episode_reward:.1f} | "  # Changed to display episode reward
                  f"Epsilon: {stats['epsilon']:.3f} | "
                  f"Loss: {stats['avg_loss']:.6f} | "
                  f"Time: {elapsed:.1f}s")
            start_time = time.time()
    
    print(f"\n✅ Training complete!")
    print(f"   Final epsilon: {cache.epsilon:.4f}")
    print(f"   Final hit rate: {episode_hit_rates[-1]:.4f}")
    
    return {
        'episode_rewards': episode_rewards,
        'episode_hit_rates': episode_hit_rates,
        'episode_losses': episode_losses,
        'total_steps': total_steps
    }


# ============================================================================
# EVALUATION PHASE
# ============================================================================

def evaluate_cache(
    cache,
    cfg,
    seed: int,
    num_requests: int = 5000,
    cache_name: str = "Unknown"
) -> Dict:
    """
    Evaluate a cache policy.
    
    Args:
        cache: Cache instance
        cfg: Configuration
        seed: Random seed
        num_requests: Number of requests to evaluate
        cache_name: Name for logging
    
    Returns:
        Performance metrics dictionary
    """
    set_seed(seed)
    
    # Set evaluation mode if DQN cache
    if isinstance(cache, StableDQNCache):
        cache.set_eval_mode(True)
    
    # Generate evaluation environment
    user_pos = channel_model.generate_user_positions(
        cfg.NUM_USERS, cfg.CELL_RADIUS, seed=seed
    )
    distances = user_pos[:, 2]
    pl = np.array([
        channel_model.pathloss(d, cfg.PATHLOSS_EXPONENT, cfg.MIN_DISTANCE)
        for d in distances
    ])
    small_scale = channel_model.rayleigh_gain(cfg.NUM_USERS)
    channel_gains = pl * small_scale
    
    # Metrics
    total_hits = 0
    total_requests = 0
    total_transmissions = 0
    total_outages = 0
    ber_list = []
    noma_successes = 0
    
    for step in range(num_requests):
        if step % 50 == 0:
            small_scale = channel_model.rayleigh_gain(cfg.NUM_USERS)
            channel_gains = pl * small_scale
        
        file_id = sample_zipf_catalog(cfg.NUM_FILES, cfg.ZIPF_ALPHA, size=1)[0]
        user_id = np.random.choice(cfg.NUM_USERS)
        
        total_requests += 1
        cache_hit = cache.is_hit(file_id)
        
        if cache_hit:
            total_hits += 1
        else:
            total_transmissions += 1
            
            # NOMA transmission
            partner_user = np.random.choice(
                [u for u in range(cfg.NUM_USERS) if u != user_id]
            )
            
            if channel_gains[user_id] < channel_gains[partner_user]:
                weak_user, strong_user = user_id, partner_user
            else:
                weak_user, strong_user = partner_user, user_id
            
            gain_weak = channel_gains[weak_user]
            gain_strong = channel_gains[strong_user]
            
            p_w, p_s, feasible, _ = allocate_power_gridsearch(
                gain_weak, gain_strong, cfg, grid_points=cfg.POWER_ALLOC_GRID
            )
            
            weak_success, strong_success, sinr_w, _, sinr_s_after = \
                simulate_noma_pair(gain_weak, gain_strong, cfg, p_w, p_s)
            
            noma_success = weak_success if user_id == weak_user else strong_success
            user_sinr = sinr_w if user_id == weak_user else sinr_s_after
            ber = compute_ber_bpsk(user_sinr)
            
            ber_list.append(ber)
            
            if not noma_success:
                total_outages += 1
            else:
                noma_successes += 1
    
    # Restore training mode if DQN cache
    if isinstance(cache, StableDQNCache):
        cache.set_eval_mode(False)
    
    hit_rate = total_hits / total_requests if total_requests > 0 else 0
    outage_rate = total_outages / total_transmissions if total_transmissions > 0 else 0
    noma_success_rate = noma_successes / total_transmissions if total_transmissions > 0 else 0
    avg_ber = np.mean(ber_list) if ber_list else 0
    
    return {
        'cache_policy': cache_name,
        'hit_rate': hit_rate,
        'outage_rate': outage_rate,
        'noma_success_rate': noma_success_rate,
        'avg_ber': avg_ber,
        'total_hits': total_hits,
        'total_transmissions': total_transmissions,
        'total_outages': total_outages
    }


# ============================================================================
# COMPREHENSIVE COMPARISON
# ============================================================================

def run_comprehensive_comparison(
    cfg,
    num_training_episodes: int = 50,
    steps_per_episode: int = 1000,
    num_eval_runs: int = 10,
    eval_requests_per_run: int = 5000
) -> pd.DataFrame:
    """
    Run comprehensive comparison of all cache policies.
    
    Compares:
    - Stable DQN (our method)
    - Top-K (static baseline)
    - LRU (dynamic baseline)
    - LFU (dynamic baseline)
    
    Returns:
        DataFrame with all results
    """
    print(f"\n{'='*80}")
    print("COMPREHENSIVE CACHE POLICY COMPARISON")
    print(f"{'='*80}")
    
    all_results = []
    
    # ========================================================================
    # 1. Train and Evaluate DQN Cache
    # ========================================================================
    
    print(f"\n{'='*80}")
    print("PHASE 1: STABLE DQN CACHE")
    print(f"{'='*80}")
    
    dqn_cache = StableDQNCache(
        capacity=cfg.CACHE_SIZE,
        num_files=cfg.NUM_FILES,
        num_users=cfg.NUM_USERS,
        learning_rate=0.0001,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay_steps=num_training_episodes * steps_per_episode // 2,
        hidden_dims=[128, 64],
        batch_size=64,
        use_prioritized_replay=True,
        seed=cfg.RANDOM_SEED
    )
    
    # Train
    training_stats = train_dqn_cache(
        dqn_cache, cfg, num_training_episodes, steps_per_episode
    )
    
    # Evaluate multiple times
    print(f"\nEvaluating DQN cache ({num_eval_runs} runs)...")
    for run in range(num_eval_runs):
        seed = cfg.RANDOM_SEED + run + 1000
        result = evaluate_cache(dqn_cache, cfg, seed, eval_requests_per_run, "DQN")
        result['run'] = run + 1
        result['seed'] = seed
        all_results.append(result)
        
        if (run + 1) % 3 == 0:
            print(f"  Run {run+1}/{num_eval_runs}: "
                  f"Hit={result['hit_rate']:.3f}, "
                  f"Outage={result['outage_rate']:.3f}")
    
    # ========================================================================
    # 2. Evaluate Baseline Caches
    # ========================================================================
    
    baseline_policies = {
        'TopK': lambda: StaticTopKCache(cfg.CACHE_SIZE),
        'LRU': lambda: LRUCache(cfg.CACHE_SIZE),
        'LFU': lambda: LFUCache(cfg.CACHE_SIZE)
    }
    
    for policy_name, cache_constructor in baseline_policies.items():
        print(f"\n{'='*80}")
        print(f"EVALUATING: {policy_name}")
        print(f"{'='*80}")
        
        for run in range(num_eval_runs):
            seed = cfg.RANDOM_SEED + run + 1000
            set_seed(seed)
            
            cache = cache_constructor()
            
            # Populate TopK cache
            if policy_name == 'TopK':
                requests = sample_zipf_catalog(
                    cfg.NUM_FILES, cfg.ZIPF_ALPHA, 
                    size=cfg.NUM_USERS * cfg.REQUESTS_PER_USER
                )
                from collections import Counter
                cnt = Counter(requests)
                ranking = [item for item, _ in cnt.most_common()]
                cache.populate(ranking)
            
            result = evaluate_cache(cache, cfg, seed, eval_requests_per_run, policy_name)
            result['run'] = run + 1
            result['seed'] = seed
            all_results.append(result)
            
            if (run + 1) % 3 == 0:
                print(f"  Run {run+1}/{num_eval_runs}: "
                      f"Hit={result['hit_rate']:.3f}, "
                      f"Outage={result['outage_rate']:.3f}")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Print summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    
    summary = df.groupby('cache_policy').agg({
        'hit_rate': ['mean', 'std'],
        'outage_rate': ['mean', 'std'],
        'noma_success_rate': ['mean', 'std'],
        'avg_ber': ['mean', 'std']
    }).round(4)
    
    print(summary)
    
    # Calculate improvements
    print(f"\n{'='*80}")
    print("IMPROVEMENTS OVER TOP-K BASELINE")
    print(f"{'='*80}")
    
    topk_hit = df[df['cache_policy'] == 'TopK']['hit_rate'].mean()
    topk_outage = df[df['cache_policy'] == 'TopK']['outage_rate'].mean()
    
    for policy in ['DQN', 'LRU', 'LFU']:
        policy_df = df[df['cache_policy'] == policy]
        hit_improvement = (policy_df['hit_rate'].mean() - topk_hit) / topk_hit * 100
        outage_reduction = (topk_outage - policy_df['outage_rate'].mean()) / topk_outage * 100
        
        print(f"\n{policy}:")
        print(f"  Hit Rate Improvement: {hit_improvement:+.2f}%")
        print(f"  Outage Reduction: {outage_reduction:+.2f}%")
    
    return df, training_stats


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_results(df: pd.DataFrame, training_stats: Dict, save_dir: str = './'):
    """Generate comprehensive plots."""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Figure 1: Performance Comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    policies = df['cache_policy'].unique()
    colors = {'DQN': 'blue', 'TopK': 'red', 'LRU': 'green', 'LFU': 'orange'}
    
    # Hit Rate
    ax = axes[0, 0]
    for policy in policies:
        data = df[df['cache_policy'] == policy]['hit_rate']
        ax.bar(policy, data.mean(), yerr=data.std(), 
               color=colors.get(policy, 'gray'), capsize=5, alpha=0.8)
    ax.set_ylabel('Hit Rate')
    ax.set_title('Cache Hit Rate Comparison')
    ax.grid(axis='y', alpha=0.3)
    
    # Outage Rate
    ax = axes[0, 1]
    for policy in policies:
        data = df[df['cache_policy'] == policy]['outage_rate']
        ax.bar(policy, data.mean(), yerr=data.std(),
               color=colors.get(policy, 'gray'), capsize=5, alpha=0.8)
    ax.set_ylabel('Outage Probability')
    ax.set_title('Outage Probability (Lower is Better)')
    ax.grid(axis='y', alpha=0.3)
    
    # NOMA Success Rate
    ax = axes[1, 0]
    for policy in policies:
        data = df[df['cache_policy'] == policy]['noma_success_rate']
        ax.bar(policy, data.mean(), yerr=data.std(),
               color=colors.get(policy, 'gray'), capsize=5, alpha=0.8)
    ax.set_ylabel('NOMA Success Rate')
    ax.set_title('NOMA Transmission Success Rate')
    ax.grid(axis='y', alpha=0.3)
    
    # BER
    ax = axes[1, 1]
    for policy in policies:
        data = df[df['cache_policy'] == policy]['avg_ber']
        ax.bar(policy, data.mean(), yerr=data.std(),
               color=colors.get(policy, 'gray'), capsize=5, alpha=0.8)
    ax.set_ylabel('Average BER')
    ax.set_title('Bit Error Rate')
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/cache_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_dir}/cache_comparison.png")
    plt.close()
    
    # Figure 2: Training Progress
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    episodes = np.arange(len(training_stats['episode_hit_rates']))
    
    # Hit Rate Evolution
    axes[0].plot(episodes, training_stats['episode_hit_rates'], 'b-', linewidth=2)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Hit Rate')
    axes[0].set_title('DQN Training: Hit Rate Evolution')
    axes[0].grid(True, alpha=0.3)
    
    # Loss Evolution
    axes[1].plot(episodes, training_stats['episode_losses'], 'r-', linewidth=2)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('DQN Training: Loss Convergence')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)
    
    # Episode Reward (FIXED: Now showing episode reward instead of cumulative)
    axes[2].plot(episodes, training_stats['episode_rewards'], 'g-', linewidth=2)
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Episode Reward')
    axes[2].set_title('DQN Training: Reward Progress')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/dqn_training_progress.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_dir}/dqn_training_progress.png")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    from src import config as cfg
    
    print("\n" + "🚀"*40)
    print("STABLE DQN CACHE FOR NOMA SYSTEMS")
    print("🚀"*40)
    
    # Run comparison
    df, training_stats = run_comprehensive_comparison(
        cfg,
        num_training_episodes=50,
        steps_per_episode=1000,
        num_eval_runs=10,
        eval_requests_per_run=5000
    )
    
    # Save results
    df.to_csv('results_dqn_comparison.csv', index=False)
    print(f"\n✅ Results saved: results_dqn_comparison.csv")
    
    # Generate plots
    plot_results(df, training_stats)
    
    print(f"\n{'='*80}")
    print("✅ SIMULATION COMPLETE!")
    print(f"{'='*80}\n")
    



    