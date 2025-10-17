# src/simulation/rl_noma_sim.py
"""
Enhanced simulation framework for RL-based caching in NOMA networks.
Integrates deep Q-learning with NOMA transmission and channel feedback.
"""

import numpy as np
import pandas as pd
from collections import Counter
from typing import Dict, List

from src.utils import set_seed, sample_zipf_catalog
from src.noma import channel_model
from src.noma.power_allocation import allocate_power_gridsearch
from src.noma.noma_base import simulate_noma_pair
from scipy.special import erfc

# Import RL cache
from src.caching.rl_noma_cache import DQNNomaCache


def compute_ber_bpsk(sinr):
    """Compute BER for BPSK modulation."""
    return 0.5 * erfc(np.sqrt(np.maximum(sinr, 0.0)))


def run_rl_noma_experiment(seed, cfg, training_mode=True):
    """
    Run single experiment with RL-based NOMA-aware caching.
    
    Args:
        seed: Random seed
        cfg: Configuration object
        training_mode: If True, RL agent learns from outcomes
    
    Returns:
        Dictionary with metrics
    """
    set_seed(seed)
    
    # Initialize RL cache
    cache = DQNNomaCache(
        capacity=cfg.CACHE_SIZE,
        num_files=cfg.NUM_FILES,
        num_users=cfg.NUM_USERS,
        learning_rate=0.001,
        gamma=0.95,
        epsilon_start=1.0 if training_mode else 0.1,  # Less exploration in eval
        epsilon_end=0.01,
        epsilon_decay=0.995
    )
    
    # Generate channel gains
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
    
    # Simulation time slots
    time_slots = cfg.TIME_SLOTS if hasattr(cfg, 'TIME_SLOTS') else 1000
    requests_per_slot = max(1, (cfg.NUM_USERS * cfg.REQUESTS_PER_USER) // time_slots)
    
    # Metrics tracking
    total_requests = 0
    total_hits = 0
    total_transmissions = 0
    total_outages = 0
    total_ber_weak = []
    total_ber_strong = []
    noma_success_count = 0
    
    cache_update_interval = getattr(cfg, 'CACHE_UPDATE_INTERVAL', 100)
    
    for time_slot in range(time_slots):
        # Generate requests for this time slot
        requests = sample_zipf_catalog(
            cfg.NUM_FILES, cfg.ZIPF_ALPHA, size=requests_per_slot
        )
        requesting_users = np.random.choice(
            cfg.NUM_USERS, size=requests_per_slot, replace=True
        )
        
        # Update channel gains (simulate mobility)
        if time_slot % 10 == 0:
            small_scale = channel_model.rayleigh_gain(cfg.NUM_USERS)
            channel_gains = pl * small_scale
        
        # Process each request
        for file_id, user_id in zip(requests, requesting_users):
            total_requests += 1
            
            # Check cache
            cache_hit = cache.is_hit(file_id)
            
            if cache_hit:
                total_hits += 1
                
                # Observe cache hit (reward agent)
                if training_mode:
                    cache.observe_request(
                        user_id=user_id,
                        file_id=file_id,
                        cache_hit=True,
                        channel_gain=channel_gains[user_id]
                    )
            else:
                # Cache miss - need NOMA transmission
                total_transmissions += 1
                
                # Find another user for pairing
                other_users = [u for u in range(cfg.NUM_USERS) if u != user_id]
                if other_users and cfg.PAIR_USERS:
                    partner_user = np.random.choice(other_users)
                    
                    # Ensure weak/strong ordering
                    if channel_gains[user_id] < channel_gains[partner_user]:
                        weak_user = user_id
                        strong_user = partner_user
                    else:
                        weak_user = partner_user
                        strong_user = user_id
                    
                    gain_weak = channel_gains[weak_user]
                    gain_strong = channel_gains[strong_user]
                    
                    # Power allocation
                    p_w, p_s, feasible, alloc_info = allocate_power_gridsearch(
                        gain_weak, gain_strong, cfg, grid_points=cfg.POWER_ALLOC_GRID
                    )
                    
                    # Simulate NOMA transmission
                    weak_success, strong_success, sinr_w, sinr_s_decode, sinr_s_after = \
                        simulate_noma_pair(gain_weak, gain_strong, cfg, p_w, p_s)
                    
                    # Determine outcome for current user
                    if user_id == weak_user:
                        noma_success = weak_success
                        user_sinr = sinr_w
                    else:
                        noma_success = strong_success
                        user_sinr = sinr_s_after
                    
                    # Compute BER
                    ber_weak = compute_ber_bpsk(sinr_w)
                    ber_strong = compute_ber_bpsk(sinr_s_after)
                    total_ber_weak.append(ber_weak)
                    total_ber_strong.append(ber_strong)
                    
                    # Determine outage
                    outage = not noma_success
                    if outage:
                        total_outages += 1
                    else:
                        noma_success_count += 1
                    
                    # Learn from outcome
                    if training_mode:
                        cache.observe_request(
                            user_id=user_id,
                            file_id=file_id,
                            cache_hit=False,
                            noma_success=noma_success,
                            channel_gain=channel_gains[user_id],
                            sinr_weak=sinr_w,
                            sinr_strong=sinr_s_after,
                            ber=ber_weak if user_id == weak_user else ber_strong,
                            outage=outage
                        )
                
                else:
                    # Single user transmission (no pairing available)
                    P = cfg.TX_POWER
                    N0 = cfg.NOISE_POWER
                    sinr_single = P * channel_gains[user_id] / N0
                    sinr_threshold = 2 ** cfg.TARGET_RATE_BPS - 1
                    noma_success = sinr_single >= sinr_threshold
                    
                    ber = compute_ber_bpsk(sinr_single)
                    total_ber_weak.append(ber)
                    
                    outage = not noma_success
                    if outage:
                        total_outages += 1
                    else:
                        noma_success_count += 1
                    
                    # Learn from outcome
                    if training_mode:
                        cache.observe_request(
                            user_id=user_id,
                            file_id=file_id,
                            cache_hit=False,
                            noma_success=noma_success,
                            channel_gain=channel_gains[user_id],
                            ber=ber,
                            outage=outage
                        )
        
        # Periodic cache optimization
        if training_mode and time_slot % cache_update_interval == 0 and time_slot > 0:
            cache.populate()
            
            # Experience replay for better learning
            if hasattr(cache, 'experience_replay'):
                cache.experience_replay(batch_size=32)
    
    # Calculate final metrics
    hit_rate = total_hits / total_requests if total_requests > 0 else 0
    outage_rate = total_outages / total_transmissions if total_transmissions > 0 else 0
    noma_success_rate = noma_success_count / total_transmissions if total_transmissions > 0 else 0
    
    avg_ber_weak = np.mean(total_ber_weak) if total_ber_weak else 0
    avg_ber_strong = np.mean(total_ber_strong) if total_ber_strong else 0
    
    # Get RL statistics
    rl_stats = cache.get_stats()
    
    result = {
        'seed': seed,
        'policy': 'rl_dqn_noma',
        'total_requests': total_requests,
        'hits': total_hits,
        'hit_rate': hit_rate,
        'total_transmissions': total_transmissions,
        'outages': total_outages,
        'outage_rate': outage_rate,
        'noma_success_rate': noma_success_rate,
        'avg_ber_weak': avg_ber_weak,
        'avg_ber_strong': avg_ber_strong,
        'avg_ber': (avg_ber_weak + avg_ber_strong) / 2,
        'q_table_size': rl_stats['q_table_size'],
        'final_epsilon': rl_stats['epsilon'],
        'cumulative_reward': rl_stats['cumulative_reward'],
        'cache_contents_count': rl_stats['cache_size']
    }
    
    return result, cache


def run_rl_noma_monte_carlo(cfg, num_runs=None):
    """
    Run multiple Monte Carlo experiments with RL-NOMA caching.
    
    Args:
        cfg: Configuration
        num_runs: Number of runs (defaults to cfg.NUM_RUNS)
    
    Returns:
        DataFrame with results
    """
    if num_runs is None:
        num_runs = cfg.NUM_RUNS
    
    results = []
    
    print("="*70)
    print("RL-NOMA CACHING SIMULATION")
    print("="*70)
    
    for run in range(num_runs):
        seed = cfg.RANDOM_SEED + run
        
        # Training mode for learning
        result, trained_cache = run_rl_noma_experiment(
            seed, cfg, training_mode=True
        )
        
        results.append(result)
        
        print(f"Run {run+1}/{num_runs}: "
              f"hit_rate={result['hit_rate']:.4f}, "
              f"outage={result['outage_rate']:.4f}, "
              f"noma_success={result['noma_success_rate']:.4f}, "
              f"reward={result['cumulative_reward']:.1f}, "
              f"epsilon={result['final_epsilon']:.3f}")
    
    df = pd.DataFrame(results)
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print(f"Mean Hit Rate:        {df['hit_rate'].mean():.4f} ± {df['hit_rate'].std():.4f}")
    print(f"Mean Outage Rate:     {df['outage_rate'].mean():.4f} ± {df['outage_rate'].std():.4f}")
    print(f"Mean NOMA Success:    {df['noma_success_rate'].mean():.4f} ± {df['noma_success_rate'].std():.4f}")
    print(f"Mean BER:             {df['avg_ber'].mean():.6f} ± {df['avg_ber'].std():.6f}")
    print(f"Mean Q-Table Size:    {df['q_table_size'].mean():.0f}")
    print(f"Mean Cumulative Reward: {df['cumulative_reward'].mean():.1f}")
    print("="*70)
    
    return df