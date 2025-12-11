# src/simulation/noma_caching_sim.py
"""
Cache-Aided NOMA System Simulator for 6G Networks

This module implements a comprehensive simulation environment for evaluating
caching policies in NOMA systems with:

- **Cache-Aided Interference Cancellation (CIC)**: Novel contribution
- **Successive Interference Cancellation (SIC)**: NOMA core mechanism  
- **Dynamic user pairing**: Channel-based strategies
- **Adaptive power allocation**: Cache-aware optimization
- **Deep RL integration**: DQN-based caching

Key Research Contributions:
1. First DRL-based cache management for NOMA with CIC exploitation
2. NOMA-aware reward design for deep RL
3. Joint optimization of caching + power allocation
4. Performance comparison against baselines (LRU, LFU, Random, TopK)

Author: Cache-Aided NOMA Team
Date: December 2025
"""

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional
import os

# Project imports
from src import config
from src.utils import set_seed, sample_zipf_catalog

# NOMA module imports
from src.noma import (
    generate_user_positions,
    compute_channel_gains,
    pair_users,
    allocate_power,
    simulate_sic_process,
    sinr_threshold_from_rate,
    rate_from_sinr
)

# Caching module imports
from src.caching import (
    create_cache,
    CacheBase,
    StaticTopKCache,
    LRUCache,
    LFUCache,
    RandomCache
)

# Try to import DQN cache
try:
    from src.caching import DQNCache
    HAS_DQN = True
except ImportError:
    HAS_DQN = False
    print("⚠️  DQN cache not available - will skip DQN experiments")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compute_popularity_ranking(requests: np.ndarray) -> Tuple[List[int], Dict[int, int]]:
    """
    Compute file popularity ranking from request history.
    
    Args:
        requests: Array of file IDs
    
    Returns:
        ranking: List of file IDs sorted by popularity (most popular first)
        counts: Dictionary of file_id -> request count
    """
    cnt = Counter(requests)
    sorted_items = [item for item, _ in cnt.most_common()]
    return sorted_items, dict(cnt)


def generate_time_varying_channels(num_users: int, cell_radius: float, 
                                   time_slots: int, doppler_freq: float,
                                   pathloss_exp: float, seed: int = None) -> np.ndarray:
    """
    Generate time-varying channel gains for mobility modeling.
    
    Args:
        num_users: Number of users
        cell_radius: Cell radius in meters
        time_slots: Number of time slots
        doppler_freq: Doppler frequency in Hz
        pathloss_exp: Path loss exponent
        seed: Random seed
    
    Returns:
        channel_gains: Array of shape (time_slots, num_users)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate initial positions
    user_pos = generate_user_positions(num_users, cell_radius, seed=seed)
    
    # Time-varying channel
    from src.noma.channel_model import TimeVaryingChannel
    tv_channel = TimeVaryingChannel(doppler_freq)
    
    gains = np.zeros((time_slots, num_users))
    for t in range(time_slots):
        gains[t, :] = tv_channel.generate_at_time(user_pos, t, pathloss_exp)
    
    return gains


# ============================================================================
# CORE SIMULATION ENGINE
# ============================================================================

class NOMACachingSimulator:
    """
    Main simulator for Cache-Aided NOMA systems.
    
    This class orchestrates the complete simulation including:
    - Channel generation
    - User pairing
    - Cache management
    - NOMA transmission with SIC/CIC
    - Performance metrics tracking
    """
    
    def __init__(self, cfg):
        """
        Initialize simulator with configuration.
        
        Args:
            cfg: Configuration object (from src.config)
        """
        self.cfg = cfg
        
        # Initialize metrics storage
        self.reset_metrics()
    
    def reset_metrics(self):
        """Reset all performance metrics."""
        self.metrics = {
            # Basic cache metrics
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            
            # NOMA transmission metrics
            'noma_transmissions': 0,
            'noma_successes': 0,
            'noma_failures': 0,
            'outages': 0,
            
            # SIC/CIC metrics (YOUR NOVEL CONTRIBUTION)
            'sic_attempts': 0,
            'sic_successes': 0,
            'sic_failures': 0,
            'cic_opportunities': 0,
            'cic_enabled_weak': 0,  # Weak user benefits from cache
            'cic_enabled_strong': 0,  # Strong user benefits from cache
            
            # Pairing metrics
            'pairs_formed': 0,
            'single_transmissions': 0,
            
            # Detailed outcomes
            'weak_user_successes': 0,
            'strong_user_successes': 0,
            'both_success': 0,
            'partial_success': 0,
            'both_fail': 0,
            
            # Rate metrics
            'total_throughput': 0.0,
            'weak_user_throughput': 0.0,
            'strong_user_throughput': 0.0,
            
            # Energy metrics
            'total_energy': 0.0,
        }
        
        # History for analysis
        self.transmission_history = []
        self.cic_events = []
    
    def run_single_episode(self, cache: CacheBase, seed: int, 
                          episode_done: bool = False) -> Dict:
        """
        Run a single simulation episode.
        
        Args:
            cache: Cache instance to evaluate
            seed: Random seed for reproducibility
            episode_done: Whether this is the last episode (for DQN)
        
        Returns:
            Dictionary of performance metrics
        """
        set_seed(seed)
        self.reset_metrics()
        
        # ========================================================================
        # STEP 1: CHANNEL SETUP
        # ========================================================================
        
        # Generate user positions
        user_positions = generate_user_positions(
            self.cfg.NUM_USERS, 
            self.cfg.CELL_RADIUS, 
            seed=seed
        )
        
        # Compute channel gains
        # ✅ BUG FIX #4, #5, #7: Correct ALL parameter names and remove unsupported 'seed'
        # Function signature: compute_channel_gains(positions, exponent, min_distance, 
        #                     fading_type, K_factor_db, los_probability)
        #                     NO 'seed' parameter!
        channel_gains = compute_channel_gains(
            user_positions,
            exponent=self.cfg.PATHLOSS_EXPONENT,       # ✅ 'exponent' not 'pathloss_exponent'
            fading_type=self.cfg.FADING_TYPE,          # ✅ Correct
            K_factor_db=self.cfg.RICIAN_K_FACTOR_DB,   # ✅ 'K_factor_db' not 'rician_k_db'
            los_probability=self.cfg.LOS_PROBABILITY   # ✅ Correct
            # ✅ Removed 'seed' - not supported by compute_channel_gains()
        )
        
        # ========================================================================
        # STEP 2: GENERATE REQUESTS
        # ========================================================================
        
        total_requests = self.cfg.NUM_USERS * self.cfg.REQUESTS_PER_USER
        file_requests = sample_zipf_catalog(
            self.cfg.NUM_FILES,
            self.cfg.ZIPF_ALPHA,
            size=total_requests
        )
        
        # Randomly assign requests to users
        requesting_users = np.random.choice(
            self.cfg.NUM_USERS,
            size=total_requests,
            replace=True
        )
        
        # ========================================================================
        # STEP 3: POPULATE CACHE (for non-learning policies)
        # ========================================================================
        
        if isinstance(cache, StaticTopKCache) and len(cache) == 0:
            # Pre-populate with top-K files
            ranking, _ = compute_popularity_ranking(file_requests)
            cache.populate(ranking)
        
        # ========================================================================
        # STEP 4: PROCESS REQUESTS WITH NOMA-AWARE CACHING
        # ========================================================================
        
        # Group requests by user for NOMA pairing
        user_requests = defaultdict(list)
        for req_idx, (file_id, user_id) in enumerate(zip(file_requests, requesting_users)):
            user_requests[user_id].append({
                'file_id': file_id,
                'req_idx': req_idx,
                'user_id': user_id
            })
        
        # Process each request
        miss_users = []  # Users with cache misses (need NOMA)
        miss_requests = {}  # file_id for each miss user
        
        for user_id in range(self.cfg.NUM_USERS):
            if user_id not in user_requests:
                continue
            
            # Take first request for this user (simplified model)
            request = user_requests[user_id][0]
            file_id = request['file_id']
            
            self.metrics['total_requests'] += 1
            
            # Check cache
            cache_hit = cache.is_hit(file_id, update_stats=True)
            
            if cache_hit:
                self.metrics['cache_hits'] += 1
                # Direct delivery from cache (no NOMA needed)
                rate = self.cfg.CACHE_DELIVERY_RATE
                self.metrics['total_throughput'] += rate
            else:
                self.metrics['cache_misses'] += 1
                miss_users.append(user_id)
                miss_requests[user_id] = file_id
        
        # ========================================================================
        # STEP 5: NOMA PAIRING FOR CACHE MISSES
        # ========================================================================
        
        if len(miss_users) == 0:
            # All requests were cache hits!
            return self._compile_results(cache)
        
        # Pair users according to strategy
        # ✅ BUG FIX #6: Correct call with users list first, then channel_gains
        pairs, leftover_user = pair_users(
            miss_users,           # ✅ First: list of users to pair
            channel_gains,        # ✅ Second: complete channel gains array
            method=self.cfg.PAIRING_METHOD  # ✅ Third: method as keyword arg
        )
        
        self.metrics['pairs_formed'] = len(pairs)
        self.metrics['single_transmissions'] = 1 if leftover_user is not None else 0
        
        # SINR threshold
        sinr_threshold = sinr_threshold_from_rate(self.cfg.TARGET_RATE_BPS)
        
        # ========================================================================
        # STEP 6: SIMULATE NOMA TRANSMISSIONS WITH SIC/CIC
        # ========================================================================
        
        for (weak_user, strong_user) in pairs:
            self._simulate_noma_pair(
                weak_user=weak_user,
                strong_user=strong_user,
                weak_file=miss_requests[weak_user],
                strong_file=miss_requests[strong_user],
                channel_gains=channel_gains,
                sinr_threshold=sinr_threshold,
                cache=cache,
                episode_done=episode_done
            )
        
        # Handle leftover user (single transmission)
        if leftover_user is not None:
            self._simulate_single_user(
                user_id=leftover_user,
                file_id=miss_requests[leftover_user],
                channel_gains=channel_gains,
                sinr_threshold=sinr_threshold,
                cache=cache,
                episode_done=episode_done
            )
        
        # ========================================================================
        # STEP 7: RETURN RESULTS
        # ========================================================================
        
        return self._compile_results(cache)
    
    def _simulate_noma_pair(self, weak_user: int, strong_user: int,
                            weak_file: int, strong_file: int,
                            channel_gains: np.ndarray, 
                            sinr_threshold: float,
                            cache: CacheBase,
                            episode_done: bool = False):
        """
        Simulate NOMA transmission for a paired users with SIC/CIC.
        
        This is the core of your novel contribution!
        """
        gain_w = channel_gains[weak_user]
        gain_s = channel_gains[strong_user]
        
        self.metrics['noma_transmissions'] += 1
        
        # ========================================================================
        # POWER ALLOCATION (Cache-aware if enabled)
        # ========================================================================
        
        # Check cache status for CIC potential
        weak_cached = cache.is_hit(weak_file, update_stats=False)
        strong_cached = cache.is_hit(strong_file, update_stats=False)
        
        # ✅ BUG FIX #1: Correct allocate_power() call
        p_weak, p_strong, feasible, _ = allocate_power(
            gain_w=gain_w,
            gain_s=gain_s,
            cfg=self.cfg,  # ✅ Pass cfg object (contains all params)
            method=self.cfg.POWER_ALLOC_METHOD,
            weak_cached=weak_cached,
            strong_cached=strong_cached,
            grid_points=self.cfg.POWER_ALLOC_GRID  # For gridsearch method
        )
        
        # ========================================================================
        # SIC/CIC SIMULATION (YOUR NOVEL CONTRIBUTION)
        # ========================================================================
        
        sic_results = simulate_sic_process(
            P_tx=self.cfg.TX_POWER,
            p_weak=p_weak,
            p_strong=p_strong,
            gain_w=gain_w,
            gain_s=gain_s,
            noise=self.cfg.NOISE_POWER,
            target_sinr=sinr_threshold,
            imperfection_factor=self.cfg.SIC_IMPERFECTION,
            weak_cached=weak_cached,
            strong_cached=strong_cached
        )
        
        # Extract results
        weak_success = sic_results['weak_success']
        strong_success = sic_results['strong_success']
        cic_applied = sic_results['cic_applied']
        
        # ========================================================================
        # UPDATE METRICS
        # ========================================================================
        
        # SIC tracking
        self.metrics['sic_attempts'] += 1
        if sic_results['can_decode_weak']:
            self.metrics['sic_successes'] += 1
        else:
            self.metrics['sic_failures'] += 1
        
        # CIC tracking (NOVEL CONTRIBUTION)
        if weak_cached:
            self.metrics['cic_opportunities'] += 1
            if strong_success:
                self.metrics['cic_enabled_strong'] += 1
                self.cic_events.append({
                    'type': 'strong_benefits',
                    'weak_user': weak_user,
                    'strong_user': strong_user,
                    'weak_file': weak_file,
                    'sinr_improvement': sic_results['sinr_s_after'] / 
                                       (sic_results['sinr_s_after'] - 
                                        (self.cfg.SIC_IMPERFECTION * self.cfg.TX_POWER * p_weak * gain_s))
                })
        
        if strong_cached:
            self.metrics['cic_opportunities'] += 1
            if weak_success:
                self.metrics['cic_enabled_weak'] += 1
                self.cic_events.append({
                    'type': 'weak_benefits',
                    'weak_user': weak_user,
                    'strong_user': strong_user,
                    'strong_file': strong_file
                })
        
        # Transmission outcomes
        if weak_success:
            self.metrics['weak_user_successes'] += 1
        if strong_success:
            self.metrics['strong_user_successes'] += 1
        
        # ✅ BUG FIX #2: Correct outage counting (no double-counting)
        # Count transmission outcome
        if weak_success and strong_success:
            self.metrics['both_success'] += 1
            self.metrics['noma_successes'] += 1
            # No outages
        elif weak_success or strong_success:
            self.metrics['partial_success'] += 1
            self.metrics['noma_successes'] += 1
            # One outage (either weak or strong)
            if not weak_success:
                self.metrics['outages'] += 1
            if not strong_success:
                self.metrics['outages'] += 1
        else:
            self.metrics['both_fail'] += 1
            self.metrics['noma_failures'] += 1
            # Both users in outage
            self.metrics['outages'] += 2
        
        # Throughput
        self.metrics['total_throughput'] += sic_results['sum_rate']
        self.metrics['weak_user_throughput'] += sic_results['rate_w']
        self.metrics['strong_user_throughput'] += sic_results['rate_s']
        
        # Energy
        self.metrics['total_energy'] += self.cfg.TX_POWER * (p_weak + p_strong)
        
        # ========================================================================
        # ✅ BUG FIX #3: DQN LEARNING (Corrected Parameters)
        # ========================================================================
        
        # DQN cache has a specific request() signature - only pass what it accepts
        if hasattr(cache, 'request'):  # DQN cache with NOMA-aware learning
            # Weak user - pass only supported parameters
            cache.request(
                item=weak_file,
                user_id=weak_user,
                channel_gain=gain_w,
                paired_user=strong_user,
                paired_file=strong_file,
                noma_success=weak_success,
                outage=not weak_success,
                ber=None,  # Optional: could calculate from SINR
                sinr_weak=sic_results['sinr_w'],
                sinr_strong=sic_results['sinr_s_after'],
                episode_done=episode_done
            )
            
            # Strong user - pass only supported parameters
            cache.request(
                item=strong_file,
                user_id=strong_user,
                channel_gain=gain_s,
                paired_user=weak_user,
                paired_file=weak_file,
                noma_success=strong_success,
                outage=not strong_success,
                ber=None,
                sinr_weak=sic_results['sinr_w'],
                sinr_strong=sic_results['sinr_s_after'],
                episode_done=episode_done
            )
        
        # Store transmission history
        self.transmission_history.append({
            'weak_user': weak_user,
            'strong_user': strong_user,
            'weak_file': weak_file,
            'strong_file': strong_file,
            'weak_success': weak_success,
            'strong_success': strong_success,
            'cic_applied': cic_applied,
            'weak_cached': weak_cached,
            'strong_cached': strong_cached,
            'sum_rate': sic_results['sum_rate']
        })
    
    def _simulate_single_user(self, user_id: int, file_id: int,
                             channel_gains: np.ndarray,
                             sinr_threshold: float,
                             cache: CacheBase,
                             episode_done: bool = False):
        """
        Simulate single user transmission (no pairing).
        """
        gain = channel_gains[user_id]
        
        # Single user gets full power
        sinr = self.cfg.TX_POWER * gain / self.cfg.NOISE_POWER
        success = sinr >= sinr_threshold
        
        if success:
            self.metrics['noma_successes'] += 1
            rate = rate_from_sinr(sinr)
            self.metrics['total_throughput'] += rate
        else:
            self.metrics['noma_failures'] += 1
            self.metrics['outages'] += 1
        
        self.metrics['noma_transmissions'] += 1
        self.metrics['total_energy'] += self.cfg.TX_POWER
        
        # ✅ BUG FIX #3: DQN learning for single user (corrected parameters)
        if hasattr(cache, 'request'):
            cache.request(
                item=file_id,
                user_id=user_id,
                channel_gain=gain,
                noma_success=success,
                outage=not success,
                episode_done=episode_done
            )
    
    def _compile_results(self, cache: CacheBase) -> Dict:
        """
        Compile final results for this episode.
        """
        total_req = max(self.metrics['total_requests'], 1)
        total_noma = max(self.metrics['noma_transmissions'], 1)
        total_sic = max(self.metrics['sic_attempts'], 1)
        
        results = {
            # Cache performance
            'hit_rate': self.metrics['cache_hits'] / total_req,
            'miss_rate': self.metrics['cache_misses'] / total_req,
            
            # NOMA performance
            'outage_probability': self.metrics['outages'] / (2 * total_noma),
            'noma_success_rate': self.metrics['noma_successes'] / total_noma,
            
            # SIC performance
            'sic_success_rate': self.metrics['sic_successes'] / total_sic,
            
            # CIC performance (NOVEL CONTRIBUTION)
            'cic_opportunity_rate': self.metrics['cic_opportunities'] / total_noma,
            'cic_benefit_rate': (self.metrics['cic_enabled_weak'] + 
                                 self.metrics['cic_enabled_strong']) / total_noma,
            
            # Throughput
            'avg_throughput': self.metrics['total_throughput'] / total_req,
            'spectral_efficiency': self.metrics['total_throughput'] / total_noma,
            
            # Energy efficiency
            'energy_per_bit': (self.metrics['total_energy'] / 
                              max(self.metrics['total_throughput'], 1)),
            
            # Raw counts
            'total_requests': self.metrics['total_requests'],
            'cache_hits': self.metrics['cache_hits'],
            'noma_transmissions': self.metrics['noma_transmissions'],
            'outages': self.metrics['outages'],
            'cic_events': len(self.cic_events),
        }
        
        # Add cache-specific stats if available
        if hasattr(cache, 'stats'):
            cache_stats = cache.stats()
            results.update({f'cache_{k}': v for k, v in cache_stats.items()})
        
        return results


# ============================================================================
# COMPARISON EXPERIMENTS
# ============================================================================

def run_baseline_comparison(cfg, num_runs: int = None) -> pd.DataFrame:
    """
    Compare different caching policies (baseline + DQN).
    
    Args:
        cfg: Configuration object
        num_runs: Number of Monte Carlo runs (overrides cfg.NUM_RUNS)
    
    Returns:
        DataFrame with results for all policies
    """
    if num_runs is None:
        num_runs = cfg.NUM_RUNS
    
    # Policies to test
    policies = ['topk', 'lru', 'lfu', 'random']
    if HAS_DQN:
        policies.append('dqn')
    
    all_results = []
    
    for policy in policies:
        print(f"\n{'='*70}")
        print(f"Testing {policy.upper()} Policy")
        print(f"{'='*70}")
        
        # Create cache
        if policy == 'dqn':
            cache = create_cache(
                policy,
                capacity=cfg.CACHE_SIZE,
                num_files=cfg.NUM_FILES,
                num_users=cfg.NUM_USERS,
                learning_rate=cfg.RL_LEARNING_RATE,
                gamma=cfg.RL_GAMMA,
                epsilon_start=cfg.RL_EPSILON_START,
                epsilon_end=cfg.RL_EPSILON_END,
                epsilon_decay_steps=cfg.RL_EPSILON_DECAY_STEPS,
                batch_size=cfg.RL_BATCH_SIZE,
                seed=cfg.RANDOM_SEED
            )
        else:
            cache = create_cache(policy, capacity=cfg.CACHE_SIZE)
        
        # Run experiments
        simulator = NOMACachingSimulator(cfg)
        
        for run in range(num_runs):
            seed = cfg.RANDOM_SEED + run
            episode_done = (run == num_runs - 1)
            
            results = simulator.run_single_episode(cache, seed, episode_done)
            results['policy'] = policy
            results['run'] = run
            results['seed'] = seed
            
            all_results.append(results)
            
            # Print progress
            print(f"  Run {run+1}/{num_runs}: "
                  f"Hit={results['hit_rate']:.3f}, "
                  f"Outage={results['outage_probability']:.3f}, "
                  f"CIC={results['cic_benefit_rate']:.3f}")
    
    return pd.DataFrame(all_results)


def run_dqn_training(cfg, episodes: int = None):
    """
    Train DQN cache and evaluate performance over time.
    
    Args:
        cfg: Configuration object
        episodes: Number of training episodes (overrides cfg.RL_TRAINING_EPISODES)
    
    Returns:
        trained_cache: Trained DQN cache
        training_history: DataFrame with training metrics
    """
    if not HAS_DQN:
        raise ImportError("DQN cache not available")
    
    if episodes is None:
        episodes = cfg.RL_TRAINING_EPISODES
    
    print(f"\n{'='*70}")
    print(f"DQN TRAINING: {episodes} episodes")
    print(f"{'='*70}\n")
    
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
    
    for episode in range(episodes):
        seed = cfg.RANDOM_SEED + episode
        episode_done = (episode == episodes - 1)
        
        # Run episode
        results = simulator.run_single_episode(dqn_cache, seed, episode_done)
        
        # Get DQN stats
        dqn_stats = dqn_cache.get_stats()
        results.update(dqn_stats)
        results['episode'] = episode
        
        training_history.append(results)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{episodes}: "
                  f"Hit={results['hit_rate']:.3f}, "
                  f"ε={dqn_stats['epsilon']:.3f}, "
                  f"Loss={dqn_stats['avg_loss']:.4f}")
    
    return dqn_cache, pd.DataFrame(training_history)


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_comparison_results(df: pd.DataFrame, save_path: str = None):
    """
    Create comprehensive visualization of comparison results.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Cache-Aided NOMA Performance Comparison', fontsize=16, y=1.02)
    
    metrics = [
        ('hit_rate', 'Cache Hit Rate', axes[0, 0]),
        ('outage_probability', 'Outage Probability', axes[0, 1]),
        ('cic_benefit_rate', 'CIC Benefit Rate', axes[0, 2]),
        ('spectral_efficiency', 'Spectral Efficiency (bps/Hz)', axes[1, 0]),
        ('sic_success_rate', 'SIC Success Rate', axes[1, 1]),
        ('energy_per_bit', 'Energy per Bit (J/bit)', axes[1, 2]),
    ]
    
    for metric, title, ax in metrics:
        sns.boxplot(data=df, x='policy', y=metric, ax=ax)
        ax.set_title(title)
        ax.set_xlabel('Policy')
        ax.set_ylabel(title)
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_dqn_training(df: pd.DataFrame, save_path: str = None):
    """
    Plot DQN training curves.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('DQN Training Progress', fontsize=16)
    
    # Hit rate over time
    axes[0, 0].plot(df['episode'], df['hit_rate'])
    axes[0, 0].set_title('Cache Hit Rate')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Hit Rate')
    axes[0, 0].grid(True)
    
    # Epsilon decay
    axes[0, 1].plot(df['episode'], df['epsilon'])
    axes[0, 1].set_title('Exploration Rate (Epsilon)')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Epsilon')
    axes[0, 1].grid(True)
    
    # Loss
    axes[1, 0].plot(df['episode'], df['avg_loss'])
    axes[1, 0].set_title('Average Loss')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True)
    
    # CIC benefit rate
    axes[1, 1].plot(df['episode'], df['cic_benefit_rate'])
    axes[1, 1].set_title('CIC Benefit Rate')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('CIC Rate')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Training plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    from src import config as cfg
    
    print("\n" + "#"*70)
    print("#" + " "*15 + "CACHE-AIDED NOMA SIMULATOR" + " "*17 + "#")
    print("#"*70 + "\n")
    
    # Configuration summary
    print("Configuration:")
    print(f"  Cache size: {cfg.CACHE_SIZE}")
    print(f"  Num files: {cfg.NUM_FILES}")
    print(f"  Num users: {cfg.NUM_USERS}")
    print(f"  Pairing: {cfg.PAIRING_METHOD}")
    print(f"  Power allocation: {cfg.POWER_ALLOC_METHOD}")
    print(f"  CIC enabled: {cfg.ENABLE_CIC}")
    print(f"  Runs: {cfg.NUM_RUNS}")
    
    t0 = time.time()
    
    # Run comparison
    print("\n" + "="*70)
    print("RUNNING BASELINE COMPARISON")
    print("="*70)
    
    results_df = run_baseline_comparison(cfg)
    
    # Aggregate statistics
    print("\n" + "="*70)
    print("AGGREGATE RESULTS")
    print("="*70)
    
    summary = results_df.groupby('policy').agg({
        'hit_rate': ['mean', 'std'],
        'outage_probability': ['mean', 'std'],
        'cic_benefit_rate': ['mean', 'std'],
        'spectral_efficiency': ['mean', 'std']
    }).round(4)
    
    print(summary)
    
    # Save results
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/noma_caching_results.csv', index=False)
    print("\n✅ Results saved to results/noma_caching_results.csv")
    
    # Plot
    plot_comparison_results(results_df, 'results/comparison_plots.png')
    
    print(f"\n⏱️  Total time: {time.time() - t0:.2f}s")
    print("\n" + "#"*70)
    print("#" + " "*20 + "SIMULATION COMPLETE" + " "*20 + "#")
    print("#"*70 + "\n")
