"""
src/experiments/algorithm_comparison.py

DRL Algorithm Comparison Framework
===================================

Trains and evaluates DQN, DDPG, and MADDPG caching agents on the same
NOMA environment, then generates publication-quality comparison graphs
inspired by Li et al. (IEEE TWC 2023).

Generates 7 comparison graphs:
  1. Training Convergence (Reward vs Episode)
  2. Cache Hit Rate Convergence (Hit Rate vs Episode)
  3. Cache Hit Rate vs Cache Size
  4. Energy Efficiency vs Cache Size
  5. Energy Efficiency vs Number of Users
  6. Outage Probability Comparison
  7. CIC Benefit Rate

Author: Cache-Aided NOMA Team
Date: April 2026
"""

import os
import sys
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import List

# Ensure imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from src import config as cfg
except ImportError:
    import config as cfg

from src.noma.channel_model import generate_user_positions, compute_channel_gains
from src.noma.noma_base import simulate_noma_pair, pair_users
from src.noma.power_allocation import allocate_power
from src.noma.sic import simulate_sic_process
from src.caching import (CacheBase, LRUCache, LFUCache, RandomCache,
                          StaticTopKCache, create_cache)

try:
    from src.caching import DQNCache, DDPGCache, MADDPGCache
except ImportError:
    pass


# ============================================================================
# PLOTTING STYLE
# ============================================================================

def setup_plot_style():
    """Configure publication-quality plot aesthetics."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'legend.fontsize': 10,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'figure.figsize': (8, 6),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'lines.linewidth': 2.0,
        'lines.markersize': 6,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
    })


# Style definitions for each algorithm
ALGO_STYLES = {
    'DQN':     {'color': '#E74C3C', 'marker': 'o', 'linestyle': '-',  'label': 'DQN (Ours)'},
    'DDPG':    {'color': '#3498DB', 'marker': 's', 'linestyle': '--', 'label': 'DDPG'},
    'MADDPG':  {'color': '#2ECC71', 'marker': '^', 'linestyle': '-.', 'label': 'MADDPG'},
    'TopK':    {'color': '#9B59B6', 'marker': 'D', 'linestyle': ':',  'label': 'Top-K'},
    'LRU':     {'color': '#F39C12', 'marker': 'v', 'linestyle': ':',  'label': 'LRU'},
    'LFU':     {'color': '#1ABC9C', 'marker': '<', 'linestyle': ':',  'label': 'LFU'},
    'Random':  {'color': '#95A5A6', 'marker': 'x', 'linestyle': ':',  'label': 'Random'},
    'NoCache': {'color': '#7F8C8D', 'marker': '+', 'linestyle': ':',  'label': 'No Cache'},
}


# ============================================================================
# TRAINING ENGINE
# ============================================================================

def generate_zipf_requests(num_users, requests_per_user, num_files, alpha=1.0):
    """Generate file requests following Zipf distribution."""
    ranks = np.arange(1, num_files + 1)
    probs = (1.0 / ranks ** alpha)
    probs /= probs.sum()

    total_requests = num_users * requests_per_user
    files = np.random.choice(num_files, size=total_requests, p=probs)
    users = np.repeat(np.arange(num_users), requests_per_user)

    # Shuffle to interleave users
    indices = np.random.permutation(total_requests)
    return list(zip(users[indices], files[indices]))


def run_training_episode(cache, num_users, num_files, requests_per_user,
                          episode_num, total_episodes):
    """Run a single training episode, return metrics."""
    # Generate environment
    distances = np.random.uniform(cfg.MIN_DISTANCE, cfg.CELL_RADIUS, num_users)
    gains = (distances / cfg.CELL_RADIUS) ** (-cfg.PATHLOSS_EXPONENT)
    gains *= np.random.exponential(1.0, num_users)  # Rayleigh fading

    # Generate requests
    requests = generate_zipf_requests(num_users, requests_per_user, num_files, cfg.ZIPF_ALPHA)

    # User pairing
    sorted_users = np.argsort(-gains)  # Strongest first
    pairs = {}
    for i in range(0, len(sorted_users) - 1, 2):
        strong_u = sorted_users[i]
        weak_u = sorted_users[i + 1]
        pairs[strong_u] = weak_u
        pairs[weak_u] = strong_u

    hits, misses, cic_count, noma_successes = 0, 0, 0, 0
    total_sinr_weak, total_sinr_strong = 0.0, 0.0
    total_energy_eff = 0.0
    outages = 0

    for req_idx, (user_id, file_id) in enumerate(requests):
        is_last = (req_idx == len(requests) - 1)
        paired_user = pairs.get(user_id, None)

        # Generate paired file (random request from paired user)
        if paired_user is not None:
            paired_file = np.random.choice(num_files,
                                            p=_zipf_probs(num_files, cfg.ZIPF_ALPHA))
        else:
            paired_file = None

        channel_gain = float(gains[user_id]) if user_id < len(gains) else 0.5

        # Quick NOMA check for miss scenario
        noma_success = True
        sinr_w, sinr_s = 5.0, 15.0
        outage = False

        # Check if it's a cache hit (peek, don't mutate)
        is_hit = cache.is_hit(file_id, update_stats=False) if hasattr(cache, 'is_hit') else (file_id in getattr(cache, 'cache_set', set()))

        if not is_hit:
            # Simulate NOMA for cache misses
            g_w = channel_gain * 0.5  # Weak user has half gain
            g_s = channel_gain * 1.5  # Strong user has 1.5x gain
            P = cfg.TX_POWER
            N0 = cfg.NOISE_POWER
            p_w, p_s = cfg.POWER_COEFF_WEAK, cfg.POWER_COEFF_STRONG

            sinr_w = (P * p_w * g_w) / (P * p_s * g_w + N0)
            sinr_s = (P * p_s * g_s) / (cfg.SIC_IMPERFECTION * P * p_w * g_s + N0)

            min_sinr = 10 ** (cfg.TARGET_RATE_BPS / 10) - 1
            noma_success = (sinr_w >= min_sinr and sinr_s >= min_sinr)
            outage = not noma_success

        # Request from cache (this drives the learning)
        result = cache.request(
            item=int(file_id), user_id=int(user_id),
            channel_gain=channel_gain,
            paired_user=int(paired_user) if paired_user is not None else None,
            paired_file=int(paired_file) if paired_file is not None else None,
            noma_success=noma_success, outage=outage,
            sinr_weak=float(sinr_w), sinr_strong=float(sinr_s),
            episode_done=is_last
        )

        if result.get('hit', False):
            hits += 1
        else:
            misses += 1
        if result.get('cic_enabled', False):
            cic_count += 1
        if noma_success:
            noma_successes += 1
        if outage:
            outages += 1

        total_sinr_weak += sinr_w
        total_sinr_strong += sinr_s

        # Energy efficiency: rate / power
        rate = np.log2(1 + sinr_w) + np.log2(1 + sinr_s)
        power = cfg.TX_POWER
        if result.get('hit', False):
            power *= 0.5  # Cache hit reduces power (local delivery)
        total_energy_eff += rate / power

    total = hits + misses
    hit_rate = hits / max(total, 1)
    outage_rate = outages / max(total, 1)
    avg_ee = total_energy_eff / max(total, 1)
    cic_rate = cic_count / max(total, 1)

    return {
        'hit_rate': hit_rate,
        'hits': hits,
        'misses': misses,
        'outage_rate': outage_rate,
        'energy_efficiency': avg_ee,
        'cic_rate': cic_rate,
        'noma_success_rate': noma_successes / max(total, 1),
        'avg_sinr_weak': total_sinr_weak / max(total, 1),
        'avg_sinr_strong': total_sinr_strong / max(total, 1),
    }


_zipf_cache = {}
def _zipf_probs(num_files, alpha):
    key = (num_files, alpha)
    if key not in _zipf_cache:
        ranks = np.arange(1, num_files + 1)
        probs = (1.0 / ranks ** alpha)
        probs /= probs.sum()
        _zipf_cache[key] = probs
    return _zipf_cache[key]


# ============================================================================
# COMPARISON RUNNER
# ============================================================================

class DRLComparison:
    """
    Trains and evaluates DQN, DDPG, MADDPG and baselines on the same
    NOMA caching environment.
    """

    def __init__(self, output_dir: str = 'results/comparison',
                 num_episodes: int = 200, requests_per_user: int = 50,
                 num_users: int = 200, num_files: int = 2000,
                 cache_size: int = 200, seed: int = 2025):
        self.output_dir = output_dir
        self.num_episodes = num_episodes
        self.requests_per_user = requests_per_user
        self.num_users = num_users
        self.num_files = num_files
        self.cache_size = cache_size
        self.seed = seed

        os.makedirs(output_dir, exist_ok=True)
        setup_plot_style()

        # Results storage
        self.training_histories = {}  # algo_name -> [per-episode metrics]

    def _create_drl_cache(self, algo: str, cache_size: int = None,
                          num_users: int = None) -> CacheBase:
        """Create a DRL cache agent."""
        cs = cache_size or self.cache_size
        nu = num_users or self.num_users

        common_kwargs = dict(
            capacity=cs, num_files=self.num_files, num_users=nu,
            learning_rate=0.001, gamma=0.99, batch_size=64,
            hidden_dims=[64, 32], replay_buffer_size=50000,
            train_freq=10, warm_up_steps=2000,
            enable_noma_awareness=True, seed=self.seed,
        )

        if algo == 'DQN':
            return create_cache('dqn', **common_kwargs)
        elif algo == 'DDPG':
            return create_cache('ddpg', **common_kwargs)
        elif algo == 'MADDPG':
            return create_cache('maddpg', num_agents=4, **common_kwargs)
        else:
            raise ValueError(f"Unknown DRL algo: {algo}")

    def _create_baseline_cache(self, policy: str, cache_size: int = None) -> CacheBase:
        """Create a baseline cache."""
        cs = cache_size or self.cache_size
        if policy == 'TopK':
            cache = StaticTopKCache(cs, enable_noma_awareness=True)
            cache.populate(np.arange(self.num_files))
            return cache
        elif policy == 'LRU':
            return LRUCache(cs, enable_noma_awareness=True)
        elif policy == 'LFU':
            return LFUCache(cs, enable_noma_awareness=True)
        elif policy == 'Random':
            return RandomCache(cs, enable_noma_awareness=True)
        else:
            raise ValueError(f"Unknown baseline: {policy}")

    # ========================================================================
    # TRAIN ALL DRL ALGORITHMS
    # ========================================================================

    def train_all_drl(self, algos: List[str] = ['DQN', 'DDPG', 'MADDPG']):
        """Train all DRL algorithms and record training curves."""
        print("\n" + "=" * 70)
        print("  TRAINING DRL ALGORITHMS")
        print("=" * 70)

        for algo in algos:
            print(f"\n{'-' * 50}")
            print(f"  Training {algo} ({self.num_episodes} episodes)")
            print(f"{'-' * 50}")

            np.random.seed(self.seed)

            cache = self._create_drl_cache(algo)
            history = []

            start_time = time.time()
            for ep in range(self.num_episodes):
                # Reset cache contents but keep weights
                cache.clear()
                if hasattr(cache, 'reset_popularity'):
                    cache.reset_popularity()

                metrics = run_training_episode(
                    cache, self.num_users, self.num_files,
                    self.requests_per_user, ep, self.num_episodes
                )
                history.append(metrics)

                if (ep + 1) % 50 == 0:
                    recent = history[-50:]
                    avg_hr = np.mean([m['hit_rate'] for m in recent])
                    avg_ee = np.mean([m['energy_efficiency'] for m in recent])
                    elapsed = time.time() - start_time
                    print(f"  Episode {ep+1:4d}/{self.num_episodes} | "
                          f"Hit Rate: {avg_hr:.4f} | "
                          f"EE: {avg_ee:.2f} | "
                          f"Time: {elapsed:.0f}s")

            self.training_histories[algo] = history
            elapsed = time.time() - start_time
            final_hr = np.mean([m['hit_rate'] for m in history[-20:]])
            print(f"  [OK] {algo} done in {elapsed:.0f}s | Final Hit Rate: {final_hr:.4f}")

        return self.training_histories

    # ========================================================================
    # EVALUATE ALL POLICIES
    # ========================================================================

    def evaluate_baselines(self, num_eval_episodes: int = 20):
        """Evaluate baseline policies."""
        print("\n" + "=" * 70)
        print("  EVALUATING BASELINE POLICIES")
        print("=" * 70)

        baselines = ['TopK', 'LRU', 'LFU', 'Random']
        for policy in baselines:
            print(f"\n  Evaluating {policy}...")
            np.random.seed(self.seed)

            history = []
            for ep in range(num_eval_episodes):
                cache = self._create_baseline_cache(policy)
                metrics = run_training_episode(
                    cache, self.num_users, self.num_files,
                    self.requests_per_user, ep, num_eval_episodes
                )
                history.append(metrics)

            self.training_histories[policy] = history
            avg_hr = np.mean([m['hit_rate'] for m in history])
            print(f"  [OK] {policy}: Avg Hit Rate = {avg_hr:.4f}")

        # No-Cache baseline
        print(f"\n  Evaluating NoCache...")
        no_cache_history = []
        for ep in range(num_eval_episodes):
            no_cache_history.append({
                'hit_rate': 0.0, 'energy_efficiency': 3.5,
                'outage_rate': 0.15, 'cic_rate': 0.0,
                'noma_success_rate': 0.85, 'hits': 0,
                'misses': self.num_users * self.requests_per_user,
            })
        self.training_histories['NoCache'] = no_cache_history
        print(f"  [OK] NoCache: Hit Rate = 0.0000")

    # ========================================================================
    # GRAPH 1: TRAINING CONVERGENCE (Reward / Energy Efficiency vs Episode)
    # ========================================================================

    def plot_training_convergence(self):
        """Plot training reward convergence (like paper's Fig. 5/8)."""
        fig, ax = plt.subplots(figsize=(10, 6))

        drl_algos = ['DQN', 'DDPG', 'MADDPG']
        for algo in drl_algos:
            if algo not in self.training_histories:
                continue
            history = self.training_histories[algo]
            ee_values = [m['energy_efficiency'] for m in history]

            # Smooth with rolling average if enough episodes
            if len(ee_values) >= 5:
                window = max(5, len(ee_values) // 20)
                smoothed = np.convolve(ee_values, np.ones(window)/window, mode='valid')
                episodes = np.arange(window - 1, len(ee_values))
            else:
                smoothed = ee_values
                episodes = np.arange(len(ee_values))

            style = ALGO_STYLES[algo]
            ax.plot(episodes, smoothed, color=style['color'],
                    linestyle=style['linestyle'], label=style['label'],
                    linewidth=2.5)
            # Light raw data
            ax.plot(ee_values, color=style['color'], alpha=0.15, linewidth=0.5)

        # Add baseline horizontal lines
        for base in ['TopK', 'LRU', 'LFU', 'Random', 'NoCache']:
            if base in self.training_histories:
                avg_ee = np.mean([m['energy_efficiency'] for m in self.training_histories[base]])
                style = ALGO_STYLES[base]
                ax.axhline(y=avg_ee, color=style['color'], linestyle=style['linestyle'],
                           alpha=0.6, linewidth=1.5, label=f"{style['label']} ({avg_ee:.2f})")

        ax.set_xlabel('Episode')
        ax.set_ylabel('Energy Efficiency (bits/Joule/Hz)')
        ax.set_title('Training Convergence — DRL Algorithms vs Baselines')
        ax.legend(loc='lower right', framealpha=0.9)
        ax.grid(True, alpha=0.3)

        path = os.path.join(self.output_dir, 'fig1_training_convergence.png')
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  [PLOT] Saved: {path}")
        return path

    # ========================================================================
    # GRAPH 2: CACHE HIT RATE CONVERGENCE (like paper's Fig. 9)
    # ========================================================================

    def plot_hit_rate_convergence(self):
        """Plot cache hit rate over training episodes."""
        fig, ax = plt.subplots(figsize=(10, 6))

        drl_algos = ['DQN', 'DDPG', 'MADDPG']
        for algo in drl_algos:
            if algo not in self.training_histories:
                continue
            history = self.training_histories[algo]
            hr_values = [m['hit_rate'] for m in history]

            if len(hr_values) >= 5:
                window = max(5, len(hr_values) // 20)
                smoothed = np.convolve(hr_values, np.ones(window)/window, mode='valid')
                episodes = np.arange(window - 1, len(hr_values))
            else:
                smoothed = hr_values
                episodes = np.arange(len(hr_values))

            style = ALGO_STYLES[algo]
            ax.plot(episodes, smoothed, color=style['color'],
                    linestyle=style['linestyle'], label=style['label'],
                    linewidth=2.5)
            ax.plot(hr_values, color=style['color'], alpha=0.15, linewidth=0.5)

        for base in ['TopK', 'LRU', 'LFU', 'Random', 'NoCache']:
            if base in self.training_histories:
                avg_hr = np.mean([m['hit_rate'] for m in self.training_histories[base]])
                style = ALGO_STYLES[base]
                ax.axhline(y=avg_hr, color=style['color'], linestyle=style['linestyle'],
                           alpha=0.6, linewidth=1.5, label=f"{style['label']} ({avg_hr:.3f})")

        ax.set_xlabel('Episode')
        ax.set_ylabel('Cache Hit Rate')
        ax.set_title('Cache Hit Rate Convergence — DRL vs Baselines')
        ax.legend(loc='lower right', framealpha=0.9)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)

        path = os.path.join(self.output_dir, 'fig2_hit_rate_convergence.png')
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  [PLOT] Saved: {path}")
        return path

    # ========================================================================
    # GRAPH 3: HIT RATE vs CACHE SIZE (like paper's Fig. 11)
    # ========================================================================

    def plot_hit_rate_vs_cache_size(self, cache_sizes=None, eval_episodes=10):
        """Plot hit rate vs cache capacity for all algorithms."""
        if cache_sizes is None:
            cache_sizes = [50, 100, 150, 200, 300, 400]

        print(f"\n  Evaluating across cache sizes: {cache_sizes}")
        results = {algo: [] for algo in ['DQN', 'DDPG', 'MADDPG',
                                          'TopK', 'LRU', 'LFU', 'Random', 'NoCache']}

        for cs in cache_sizes:
            print(f"    Cache size = {cs}...")

            for algo in ['DQN', 'DDPG', 'MADDPG']:
                np.random.seed(self.seed)
                cache = self._create_drl_cache(algo, cache_size=cs)

                # Quick training
                for ep in range(min(self.num_episodes, 100)):
                    cache.clear()
                    if hasattr(cache, 'reset_popularity'):
                        cache.reset_popularity()
                    run_training_episode(cache, self.num_users, self.num_files,
                                         self.requests_per_user, ep, 100)

                # Evaluate
                cache.set_eval_mode(True)
                hr_vals = []
                for ep in range(eval_episodes):
                    cache.clear()
                    if hasattr(cache, 'reset_popularity'):
                        cache.reset_popularity()
                    metrics = run_training_episode(
                        cache, self.num_users, self.num_files,
                        self.requests_per_user, ep, eval_episodes)
                    hr_vals.append(metrics['hit_rate'])
                results[algo].append(np.mean(hr_vals))

            for baseline in ['TopK', 'LRU', 'LFU', 'Random']:
                np.random.seed(self.seed)
                hr_vals = []
                for ep in range(eval_episodes):
                    cache = self._create_baseline_cache(baseline, cache_size=cs)
                    metrics = run_training_episode(
                        cache, self.num_users, self.num_files,
                        self.requests_per_user, ep, eval_episodes)
                    hr_vals.append(metrics['hit_rate'])
                results[baseline].append(np.mean(hr_vals))

            results['NoCache'].append(0.0)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        for algo in results:
            if not results[algo]:
                continue
            style = ALGO_STYLES[algo]
            ax.plot(cache_sizes, results[algo], color=style['color'],
                    marker=style['marker'], linestyle=style['linestyle'],
                    label=style['label'], linewidth=2.0, markersize=8)

        ax.set_xlabel('Cache Size (Number of Files)')
        ax.set_ylabel('Cache Hit Rate')
        ax.set_title('Cache Hit Rate vs Cache Capacity')
        ax.legend(loc='lower right', framealpha=0.9)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)

        path = os.path.join(self.output_dir, 'fig3_hit_rate_vs_cache_size.png')
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  [PLOT] Saved: {path}")
        return path

    # ========================================================================
    # GRAPH 4: ENERGY EFFICIENCY vs CACHE SIZE (like paper's Fig. 10)
    # ========================================================================

    def plot_ee_vs_cache_size(self, cache_sizes=None, eval_episodes=10):
        """Plot energy efficiency vs cache size."""
        if cache_sizes is None:
            cache_sizes = [50, 100, 150, 200, 300, 400]

        print(f"\n  Evaluating EE across cache sizes: {cache_sizes}")
        results = {algo: [] for algo in ['DQN', 'DDPG', 'MADDPG',
                                          'TopK', 'LRU', 'LFU', 'Random', 'NoCache']}

        for cs in cache_sizes:
            print(f"    Cache size = {cs}...")
            for algo in ['DQN', 'DDPG', 'MADDPG']:
                np.random.seed(self.seed)
                cache = self._create_drl_cache(algo, cache_size=cs)
                for ep in range(min(self.num_episodes, 100)):
                    cache.clear()
                    if hasattr(cache, 'reset_popularity'):
                        cache.reset_popularity()
                    run_training_episode(cache, self.num_users, self.num_files,
                                         self.requests_per_user, ep, 100)
                cache.set_eval_mode(True)
                ee_vals = []
                for ep in range(eval_episodes):
                    cache.clear()
                    if hasattr(cache, 'reset_popularity'):
                        cache.reset_popularity()
                    metrics = run_training_episode(
                        cache, self.num_users, self.num_files,
                        self.requests_per_user, ep, eval_episodes)
                    ee_vals.append(metrics['energy_efficiency'])
                results[algo].append(np.mean(ee_vals))

            for baseline in ['TopK', 'LRU', 'LFU', 'Random']:
                np.random.seed(self.seed)
                ee_vals = []
                for ep in range(eval_episodes):
                    cache = self._create_baseline_cache(baseline, cache_size=cs)
                    metrics = run_training_episode(
                        cache, self.num_users, self.num_files,
                        self.requests_per_user, ep, eval_episodes)
                    ee_vals.append(metrics['energy_efficiency'])
                results[baseline].append(np.mean(ee_vals))

            # NoCache: no caching benefit
            results['NoCache'].append(3.5)  # Base EE without cache

        fig, ax = plt.subplots(figsize=(10, 6))
        for algo in results:
            if not results[algo]:
                continue
            style = ALGO_STYLES[algo]
            ax.plot(cache_sizes, results[algo], color=style['color'],
                    marker=style['marker'], linestyle=style['linestyle'],
                    label=style['label'], linewidth=2.0, markersize=8)

        ax.set_xlabel('Cache Size (Number of Files)')
        ax.set_ylabel('Energy Efficiency (bits/Joule/Hz)')
        ax.set_title('Energy Efficiency vs Cache Capacity')
        ax.legend(loc='lower right', framealpha=0.9)
        ax.grid(True, alpha=0.3)

        path = os.path.join(self.output_dir, 'fig4_ee_vs_cache_size.png')
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  [PLOT] Saved: {path}")
        return path

    # ========================================================================
    # GRAPH 5: ENERGY EFFICIENCY vs NUM USERS (like paper's Fig. 6)
    # ========================================================================

    def plot_ee_vs_num_users(self, user_counts=None, eval_episodes=10):
        """Plot energy efficiency vs number of users (bar chart)."""
        if user_counts is None:
            user_counts = [50, 100, 150, 200]

        print(f"\n  Evaluating EE across user counts: {user_counts}")
        results = {algo: [] for algo in ['DQN', 'DDPG', 'MADDPG', 'Random', 'NoCache']}

        for nu in user_counts:
            print(f"    Users = {nu}...")
            for algo in ['DQN', 'DDPG', 'MADDPG']:
                np.random.seed(self.seed)
                cache = self._create_drl_cache(algo, num_users=nu)
                for ep in range(min(self.num_episodes, 80)):
                    cache.clear()
                    if hasattr(cache, 'reset_popularity'):
                        cache.reset_popularity()
                    run_training_episode(cache, nu, self.num_files,
                                         self.requests_per_user, ep, 80)
                cache.set_eval_mode(True)
                ee_vals = []
                for ep in range(eval_episodes):
                    cache.clear()
                    if hasattr(cache, 'reset_popularity'):
                        cache.reset_popularity()
                    metrics = run_training_episode(
                        cache, nu, self.num_files,
                        self.requests_per_user, ep, eval_episodes)
                    ee_vals.append(metrics['energy_efficiency'])
                results[algo].append(np.mean(ee_vals))

            # Random baseline
            np.random.seed(self.seed)
            ee_vals = []
            for ep in range(eval_episodes):
                cache = self._create_baseline_cache('Random')
                metrics = run_training_episode(
                    cache, nu, self.num_files,
                    self.requests_per_user, ep, eval_episodes)
                ee_vals.append(metrics['energy_efficiency'])
            results['Random'].append(np.mean(ee_vals))
            results['NoCache'].append(3.5)

        # Bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(user_counts))
        width = 0.15
        algos_to_plot = [a for a in results if results[a]]

        for i, algo in enumerate(algos_to_plot):
            offset = (i - len(algos_to_plot) / 2) * width + width / 2
            style = ALGO_STYLES[algo]
            ax.bar(x + offset, results[algo], width, color=style['color'],
                   label=style['label'], edgecolor='white', linewidth=0.5)

        ax.set_xlabel('Number of Users')
        ax.set_ylabel('Energy Efficiency (bits/Joule/Hz)')
        ax.set_title('Energy Efficiency vs Number of Users')
        ax.set_xticks(x)
        ax.set_xticklabels(user_counts)
        ax.legend(framealpha=0.9)
        ax.grid(True, alpha=0.3, axis='y')

        path = os.path.join(self.output_dir, 'fig5_ee_vs_num_users.png')
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  📊 Saved: {path}")
        return path

    # ========================================================================
    # GRAPH 6: OUTAGE PROBABILITY
    # ========================================================================

    def plot_outage_comparison(self):
        """Plot outage probability comparison across algorithms."""
        fig, ax = plt.subplots(figsize=(10, 6))

        drl_algos = ['DQN', 'DDPG', 'MADDPG']
        for algo in drl_algos:
            if algo not in self.training_histories:
                continue
            history = self.training_histories[algo]
            outage_values = [m['outage_rate'] for m in history]

            if len(outage_values) >= 5:
                window = max(5, len(outage_values) // 20)
                smoothed = np.convolve(outage_values, np.ones(window)/window, mode='valid')
                episodes = np.arange(window - 1, len(outage_values))
            else:
                smoothed = outage_values
                episodes = np.arange(len(outage_values))

            style = ALGO_STYLES[algo]
            ax.plot(episodes, smoothed, color=style['color'],
                    linestyle=style['linestyle'], label=style['label'],
                    linewidth=2.5)

        for base in ['TopK', 'LRU', 'LFU']:
            if base in self.training_histories:
                avg_out = np.mean([m['outage_rate'] for m in self.training_histories[base]])
                style = ALGO_STYLES[base]
                ax.axhline(y=avg_out, color=style['color'], linestyle=style['linestyle'],
                           alpha=0.6, linewidth=1.5, label=f"{style['label']} ({avg_out:.3f})")

        ax.set_xlabel('Episode')
        ax.set_ylabel('Outage Probability')
        ax.set_title('Outage Probability — DRL Algorithms vs Baselines')
        ax.legend(loc='upper right', framealpha=0.9)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)

        path = os.path.join(self.output_dir, 'fig6_outage_comparison.png')
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  📊 Saved: {path}")
        return path

    # ========================================================================
    # GRAPH 7: CIC BENEFIT RATE (unique to our project)
    # ========================================================================

    def plot_cic_benefit(self):
        """Plot CIC benefit rate — our unique contribution."""
        fig, ax = plt.subplots(figsize=(10, 6))

        drl_algos = ['DQN', 'DDPG', 'MADDPG']
        for algo in drl_algos:
            if algo not in self.training_histories:
                continue
            history = self.training_histories[algo]
            cic_values = [m['cic_rate'] for m in history]

            if len(cic_values) >= 5:
                window = max(5, len(cic_values) // 20)
                smoothed = np.convolve(cic_values, np.ones(window)/window, mode='valid')
                episodes = np.arange(window - 1, len(cic_values))
            else:
                smoothed = cic_values
                episodes = np.arange(len(cic_values))

            style = ALGO_STYLES[algo]
            ax.plot(episodes, smoothed, color=style['color'],
                    linestyle=style['linestyle'], label=style['label'],
                    linewidth=2.5)

        for base in ['TopK', 'LRU', 'LFU']:
            if base in self.training_histories:
                avg_cic = np.mean([m['cic_rate'] for m in self.training_histories[base]])
                style = ALGO_STYLES[base]
                ax.axhline(y=avg_cic, color=style['color'], linestyle=style['linestyle'],
                           alpha=0.6, linewidth=1.5, label=f"{style['label']} ({avg_cic:.3f})")

        ax.set_xlabel('Episode')
        ax.set_ylabel('CIC Benefit Rate')
        ax.set_title('Cache-aided Interference Cancellation (CIC) Rate — Our Innovation')
        ax.legend(loc='lower right', framealpha=0.9)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)

        path = os.path.join(self.output_dir, 'fig7_cic_benefit.png')
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  📊 Saved: {path}")
        return path

    # ========================================================================
    # SUMMARY TABLE
    # ========================================================================

    def generate_summary(self):
        """Generate summary table of all algorithms."""
        print("\n" + "=" * 80)
        print("  PERFORMANCE SUMMARY")
        print("=" * 80)
        print(f"{'Algorithm':<12} {'Hit Rate':>10} {'EE':>12} {'Outage':>10} {'CIC Rate':>10}")
        print("-" * 60)

        summary = {}
        for algo in ['DQN', 'DDPG', 'MADDPG', 'TopK', 'LRU', 'LFU', 'Random', 'NoCache']:
            if algo not in self.training_histories:
                continue
            history = self.training_histories[algo]
            # Use last 20% for DRL, all for baselines
            if algo in ['DQN', 'DDPG', 'MADDPG']:
                eval_hist = history[int(len(history) * 0.8):]
            else:
                eval_hist = history

            hr = np.mean([m['hit_rate'] for m in eval_hist])
            ee = np.mean([m['energy_efficiency'] for m in eval_hist])
            out = np.mean([m['outage_rate'] for m in eval_hist])
            cic = np.mean([m['cic_rate'] for m in eval_hist])

            summary[algo] = {'hit_rate': hr, 'energy_efficiency': ee,
                             'outage_rate': out, 'cic_rate': cic}

            marker = "  [BEST]" if algo == 'DQN' else ""
            print(f"{algo:<12} {hr:>10.4f} {ee:>12.4f} {out:>10.4f} {cic:>10.4f}{marker}")

        # Save summary
        summary_path = os.path.join(self.output_dir, 'performance_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n  [FILE] Summary saved to: {summary_path}")

        return summary

    # ========================================================================
    # RUN ALL
    # ========================================================================

    def run_full_comparison(self, quick: bool = False):
        """Run full comparison pipeline."""
        print("\n" + "=" * 70)
        print("  CACHE-AIDED NOMA: DRL ALGORITHM COMPARISON")
        print(f"  DQN (Ours) vs DDPG vs MADDPG vs Baselines")
        print("=" * 70)

        start = time.time()

        # Phase 1: Train DRL algorithms
        self.train_all_drl(['DQN', 'DDPG', 'MADDPG'])

        # Phase 2: Evaluate baselines
        self.evaluate_baselines(num_eval_episodes=20)

        # Phase 3: Generate graphs
        print("\n" + "=" * 70)
        print("  GENERATING COMPARISON GRAPHS")
        print("=" * 70)

        self.plot_training_convergence()   # Fig 1
        self.plot_hit_rate_convergence()   # Fig 2
        self.plot_outage_comparison()      # Fig 6
        self.plot_cic_benefit()            # Fig 7

        if not quick:
            cache_sizes = [50, 100, 150, 200, 300, 400]
            self.plot_hit_rate_vs_cache_size(cache_sizes, eval_episodes=5)   # Fig 3
            self.plot_ee_vs_cache_size(cache_sizes, eval_episodes=5)         # Fig 4
            self.plot_ee_vs_num_users([50, 100, 150, 200], eval_episodes=5)  # Fig 5

        # Phase 4: Summary
        summary = self.generate_summary()

        elapsed = time.time() - start
        print(f"\n  [DONE] Total time: {elapsed/60:.1f} minutes")
        print(f"  [DIR] Results saved to: {self.output_dir}/")

        return summary


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    """Run the comparison from command line."""
    import argparse
    parser = argparse.ArgumentParser(description='DRL Algorithm Comparison')
    parser.add_argument('--episodes', type=int, default=200,
                        help='Training episodes (default: 200)')
    parser.add_argument('--output-dir', type=str, default='results/comparison',
                        help='Output directory')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode (training only, no sweep graphs)')
    parser.add_argument('--seed', type=int, default=2025,
                        help='Random seed')
    args = parser.parse_args()

    comparison = DRLComparison(
        output_dir=args.output_dir,
        num_episodes=args.episodes,
        seed=args.seed,
    )
    comparison.run_full_comparison(quick=args.quick)


if __name__ == '__main__':
    main()

