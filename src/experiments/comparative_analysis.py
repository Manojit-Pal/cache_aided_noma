# src/experiments/comparative_analysis.py
"""
Comprehensive Comparative Analysis: Cache-Aided NOMA vs Traditional Systems

✅ AUTO-TRAINS DQN IF NEEDED (Dec 12, 2025)

This module implements research-validated comparative studies with automatic
DQN training integration. If DQN checkpoint doesn't exist, it will train
automatically before running comparisons.

Research References:
- arXiv:1712.09557 (2018): "Cache-Aided Non-Orthogonal Multiple Access"
- arXiv:1909.11074 (2019): "Power Allocation in Cache-Aided NOMA"
- IEEE Survey (2022): "A Survey on Applications of Cache-Aided NOMA"
- arXiv:2209.07809 (2022): "M2DQN - DQN training requirements"

Key Features:
1. ✅ Automatic DQN checkpoint detection
2. ✅ Auto-training if checkpoint missing (with user approval)
3. ✅ Fair comparison: all policies optimized (including DQN)
4. ✅ Complete 6-policy analysis: TopK, LRU, LFU, Random, NO-CACHE, DQN
5. ✅ 9 comprehensive metrics with BER for both users

Workflow:
- First run: Auto-trains DQN (1000 episodes, ~10-30 min)
- Subsequent runs: Loads trained DQN checkpoint
- Results: Publication-ready comparison plots

Outputs:
- cache_aided_vs_traditional_noma.png: Main comparison (9 subplots)
- results_comparative_analysis.csv: Complete dataset
- performance_summary.txt: Statistical summary
- models/dqn_cache/dqn_cache_final.pth: Trained DQN checkpoint

Author: Cache-Aided NOMA Team
Date: December 12, 2025
Version: 4.0 (AUTO-TRAIN DQN Integration)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import erfc
from scipy import stats
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional
import os
import warnings
import sys
warnings.filterwarnings('ignore')

# Project imports
from src import config as cfg
from src.utils import set_seed, sample_zipf_catalog

# NOMA imports
from src.noma import (
    generate_user_positions,
    compute_channel_gains,
    pair_users,
    allocate_power,
    simulate_sic_process,
    sinr_threshold_from_rate,
    rate_from_sinr
)

# Caching imports
from src.caching import (
    create_cache,
    StaticTopKCache,
    LRUCache,
    LFUCache,
    RandomCache
)

# Try DQN import
try:
    from src.caching import DQNCache
    HAS_DQN = True
except ImportError:
    HAS_DQN = False
    print("⚠️  DQN not available - will skip DQN comparison")


# ============================================================================
# DQN TRAINING INTEGRATION
# ============================================================================

def check_dqn_checkpoint() -> Optional[str]:
    """
    Check if trained DQN checkpoint exists.
    
    Returns:
        Path to checkpoint if exists, None otherwise
    """
    checkpoint_path = 'models/dqn_cache/dqn_cache_final.pth'
    
    if os.path.exists(checkpoint_path):
        return checkpoint_path
    
    # Try alternative location
    alt_path = 'models/dqn_cache/dqn_cache_best_ep999.pth'
    if os.path.exists(alt_path):
        return alt_path
    
    return None


def train_dqn_automatically(cfg, num_episodes: int = 1000) -> Optional[str]:
    """
    Automatically train DQN cache.
    
    Args:
        cfg: Configuration object
        num_episodes: Number of training episodes
    
    Returns:
        Path to saved checkpoint or None if training failed
    """
    if not HAS_DQN:
        print("❌ Error: DQN not available for training")
        return None
    
    try:
        # Import trainer
        from src.simulation.stable_dqn_sim import NOMADQNTrainer
        
        print("\n" + "#"*80)
        print("#" + " "*20 + "AUTO-TRAINING DQN CACHE" + " "*24 + "#")
        print("#"*80)
        print(f"\nNo trained DQN checkpoint found.")
        print(f"Training will take approximately 10-30 minutes.")
        print(f"Episodes: {num_episodes}")
        print(f"\nResults will be saved to: models/dqn_cache/dqn_cache_final.pth")
        print(f"\nPress Ctrl+C to cancel and run without DQN.\n")
        
        # User confirmation
        response = input("Proceed with DQN training? [Y/n]: ").strip().lower()
        if response and response not in ['y', 'yes']:
            print("\n⚠️  Skipping DQN training. Will run comparison without DQN.")
            return None
        
        print("\n✅ Starting DQN training...\n")
        
        # Create trainer
        trainer = NOMADQNTrainer(cfg)
        
        # Train
        trained_cache, train_history = trainer.train(
            num_episodes=num_episodes,
            test_interval=50,
            save_best=True
        )
        
        # Check if checkpoint was created
        checkpoint_path = check_dqn_checkpoint()
        
        if checkpoint_path:
            print(f"\n✅ DQN training complete!")
            print(f"   Checkpoint saved: {checkpoint_path}")
            return checkpoint_path
        else:
            print(f"\n⚠️  Training completed but checkpoint not found.")
            return None
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Training cancelled by user.")
        return None
    
    except Exception as e:
        print(f"\n❌ Error during DQN training: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_trained_dqn(checkpoint_path: str, cfg) -> Optional[object]:
    """
    Load trained DQN cache from checkpoint.
    
    Args:
        checkpoint_path: Path to .pth checkpoint
        cfg: Configuration object
    
    Returns:
        Loaded DQN cache or None if loading failed
    """
    if not HAS_DQN:
        return None
    
    try:
        # Create DQN cache instance
        cache = DQNCache(
            capacity=cfg.CACHE_SIZE,
            num_files=cfg.NUM_FILES,
            num_users=cfg.NUM_USERS,
            learning_rate=cfg.RL_LEARNING_RATE,
            gamma=cfg.RL_GAMMA,
            seed=cfg.RANDOM_SEED
        )
        
        # Load weights
        cache.load_model(checkpoint_path)
        
        # Set to evaluation mode (no exploration, no learning)
        cache.set_eval_mode(True)
        
        print(f"✅ Loaded trained DQN from: {checkpoint_path}")
        return cache
    
    except Exception as e:
        print(f"❌ Error loading DQN checkpoint: {e}")
        return None


# ============================================================================
# COMPREHENSIVE COMPARATIVE ANALYSIS ENGINE
# ============================================================================

class CacheAidedNOMAAnalysis:
    """
    Research-validated comparative analysis engine.
    
    ✅ NOW INCLUDES TRAINED DQN (not random!)
    
    Compares:
    - Cache-Aided NOMA (TopK, LRU, LFU, Random, DQN)
    - Traditional NOMA (no cache)
    """
    
    def __init__(self, cfg, snr_range_db: np.ndarray = None, 
                 num_realizations: int = 1000,
                 trained_dqn_cache: Optional[object] = None):
        """
        Initialize analysis engine.
        
        Args:
            cfg: Configuration object
            snr_range_db: SNR values to test (dB)
            num_realizations: Monte Carlo runs per SNR
            trained_dqn_cache: Pre-trained DQN cache instance
        """
        self.cfg = cfg
        self.snr_db_range = snr_range_db if snr_range_db is not None else np.arange(-10, 31, 2)
        self.num_realizations = num_realizations
        self.trained_dqn_cache = trained_dqn_cache
        
        # Results storage
        self.results = defaultdict(list)
        
        print(f"\n✅ CacheAidedNOMAAnalysis initialized")
        print(f"   SNR range: {self.snr_db_range[0]} to {self.snr_db_range[-1]} dB")
        print(f"   Realizations: {self.num_realizations}")
        
        policies_list = ['TopK', 'LRU', 'LFU', 'Random', 'NO-CACHE']
        if trained_dqn_cache is not None:
            policies_list.append('DQN (trained)')
        print(f"   Policies: {', '.join(policies_list)}\n")
    
    def db_to_linear(self, db_value):
        """Convert dB to linear scale."""
        return 10 ** (db_value / 10.0)
    
    def linear_to_db(self, linear_value):
        """Convert linear to dB scale."""
        return 10 * np.log10(np.maximum(linear_value, 1e-12))
    
    def compute_jains_fairness(self, rates: List[float]) -> float:
        """
        Compute Jain's Fairness Index.
        
        J = (sum(r_i))^2 / (n * sum(r_i^2))
        Range: [1/n, 1], where 1 = perfect fairness
        """
        rates = np.array(rates)
        n = len(rates)
        if n == 0 or rates.sum() == 0:
            return 0.0
        return (rates.sum() ** 2) / (n * (rates ** 2).sum())
    
    def setup_cache(self, policy: str, requests: np.ndarray = None) -> Optional[object]:
        """
        Create cache instance for given policy.
        
        ✅ FIXED: Uses trained DQN if available
        """
        if policy is None or policy == 'none':
            return None
        
        if policy == 'dqn':
            if self.trained_dqn_cache is None:
                print("⚠️  Warning: DQN requested but no trained cache available")
                return None
            # Return the pre-trained DQN cache (already in eval mode)
            return self.trained_dqn_cache
        
        # Non-DQN caches
        cache = create_cache(policy, capacity=self.cfg.CACHE_SIZE)
        
        # Populate static caches
        if isinstance(cache, StaticTopKCache) and requests is not None:
            cnt = Counter(requests)
            ranking = [item for item, _ in cnt.most_common()]
            cache.populate(ranking)
        
        return cache
    
    def generate_user_pair_channels(self, snr_db: float, seed: int = None) -> Tuple:
        """Generate channel gains for weak-strong user pair."""
        if seed is not None:
            set_seed(seed)
        
        positions = generate_user_positions(
            num_users=2,
            cell_radius=self.cfg.CELL_RADIUS,
            seed=seed
        )
        
        channel_gains = compute_channel_gains(
            positions,
            exponent=self.cfg.PATHLOSS_EXPONENT,
            fading_type=self.cfg.FADING_TYPE,
            K_factor_db=self.cfg.RICIAN_K_FACTOR_DB,
            los_probability=self.cfg.LOS_PROBABILITY
        )
        
        gain_weak = np.min(channel_gains)
        gain_strong = np.max(channel_gains)
        
        gain_avg = np.mean(channel_gains)
        noise_power = self.cfg.TX_POWER * gain_avg / self.db_to_linear(snr_db)
        
        return gain_weak, gain_strong, noise_power
    
    def simulate_noma_transmission(self, gain_weak: float, gain_strong: float,
                                   noise_power: float, cache = None,
                                   file_weak: int = 0, file_strong: int = 1) -> Dict:
        """Simulate NOMA transmission with optional cache assistance."""
        # Check cache status
        if cache is not None:
            cache_hit_weak = cache.is_hit(file_weak, update_stats=False)
            cache_hit_strong = cache.is_hit(file_strong, update_stats=False)
        else:
            cache_hit_weak = False
            cache_hit_strong = False
        
        # If both cached, no transmission needed
        if cache_hit_weak and cache_hit_strong:
            cache_rate = getattr(self.cfg, 'CACHE_DELIVERY_RATE', self.cfg.TARGET_RATE_BPS)
            return {
                'sinr_weak': np.inf,
                'sinr_strong': np.inf,
                'rate_weak': cache_rate,
                'rate_strong': cache_rate,
                'sum_rate': 2 * cache_rate,
                'outage_weak': 0,
                'outage_strong': 0,
                'ber_weak': 0.0,
                'ber_strong': 0.0,
                'cache_hit_weak': 1,
                'cache_hit_strong': 1,
                'transmission_needed': 0,
                'sic_success': 1,
                'cic_opportunity': 0,
                'cic_benefit': 0,
                'energy': 0.0
            }
        
        # Power Allocation
        p_weak, p_strong, feasible, alloc_info = allocate_power(
            gain_w=gain_weak,
            gain_s=gain_strong,
            cfg=self.cfg,
            method=self.cfg.POWER_ALLOC_METHOD,
            weak_cached=cache_hit_weak,
            strong_cached=cache_hit_strong,
            grid_points=self.cfg.POWER_ALLOC_GRID
        )
        
        # SIC/CIC Simulation
        sinr_threshold = sinr_threshold_from_rate(self.cfg.TARGET_RATE_BPS)
        
        sic_results = simulate_sic_process(
            P_tx=self.cfg.TX_POWER,
            p_weak=p_weak,
            p_strong=p_strong,
            gain_w=gain_weak,
            gain_s=gain_strong,
            noise=noise_power,
            target_sinr=sinr_threshold,
            imperfection_factor=self.cfg.SIC_IMPERFECTION,
            weak_cached=cache_hit_weak,
            strong_cached=cache_hit_strong
        )
        
        weak_success = sic_results['weak_success']
        strong_success = sic_results['strong_success']
        
        # Compute Rates
        cache_rate = getattr(self.cfg, 'CACHE_DELIVERY_RATE', self.cfg.TARGET_RATE_BPS)
        
        if cache_hit_weak:
            rate_weak = cache_rate
        else:
            rate_weak = sic_results['rate_w'] if weak_success else 0.0
        
        if cache_hit_strong:
            rate_strong = cache_rate
        else:
            rate_strong = sic_results['rate_s'] if strong_success else 0.0
        
        # Outage Detection
        outage_weak = 0 if (cache_hit_weak or weak_success) else 1
        outage_strong = 0 if (cache_hit_strong or strong_success) else 1
        
        # BER Computation (BPSK)
        def compute_ber(sinr):
            sinr_val = np.maximum(sinr, 0.0)
            return 0.5 * erfc(np.sqrt(sinr_val))
        
        ber_weak = 0.0 if cache_hit_weak else compute_ber(sic_results['sinr_w'])
        ber_strong = 0.0 if cache_hit_strong else compute_ber(sic_results['sinr_s_after'])
        
        # CIC Tracking
        cic_opportunity = 0
        cic_benefit = 0
        
        if cache_hit_strong and not cache_hit_weak:
            cic_opportunity += 1
            if weak_success:
                cic_benefit += 1
        
        if cache_hit_weak and not cache_hit_strong:
            cic_opportunity += 1
            if strong_success:
                cic_benefit += 1
        
        # Energy consumption
        energy = self.cfg.TX_POWER * (p_weak + p_strong)
        
        return {
            'sinr_weak': sic_results['sinr_w'],
            'sinr_strong': sic_results['sinr_s_after'],
            'rate_weak': rate_weak,
            'rate_strong': rate_strong,
            'sum_rate': sic_results['sum_rate'],
            'outage_weak': outage_weak,
            'outage_strong': outage_strong,
            'ber_weak': ber_weak,
            'ber_strong': ber_strong,
            'cache_hit_weak': int(cache_hit_weak),
            'cache_hit_strong': int(cache_hit_strong),
            'transmission_needed': int(not (cache_hit_weak and cache_hit_strong)),
            'sic_success': int(sic_results['can_decode_weak']),
            'cic_opportunity': cic_opportunity,
            'cic_benefit': cic_benefit,
            'energy': energy
        }
    
    def run_single_snr(self, snr_db: float, policy: str = 'topk',
                       seed_offset: int = 0) -> Dict:
        """Run Monte Carlo simulations for a single SNR point."""
        # Generate file requests for cache setup
        requests = sample_zipf_catalog(
            self.cfg.NUM_FILES,
            self.cfg.ZIPF_ALPHA,
            size=self.cfg.NUM_USERS * self.cfg.REQUESTS_PER_USER
        )
        
        # Setup cache
        cache = self.setup_cache(policy, requests)
        
        # Storage for per-realization metrics
        metrics = defaultdict(list)
        
        # Zipf probability distribution
        ranks = np.arange(1, self.cfg.NUM_FILES + 1)
        zipf_weights = 1.0 / np.power(ranks, self.cfg.ZIPF_ALPHA)
        zipf_probs = zipf_weights / zipf_weights.sum()
        
        # Monte Carlo simulations
        for i in range(self.num_realizations):
            seed = self.cfg.RANDOM_SEED + seed_offset + i
            
            # Generate channels
            gain_weak, gain_strong, noise_power = self.generate_user_pair_channels(snr_db, seed)
            
            # Sample file requests
            file_weak = np.random.choice(self.cfg.NUM_FILES, p=zipf_probs)
            file_strong = np.random.choice(self.cfg.NUM_FILES, p=zipf_probs)
            
            # Simulate transmission
            result = self.simulate_noma_transmission(
                gain_weak, gain_strong, noise_power,
                cache, file_weak, file_strong
            )
            
            # Store metrics
            for key, value in result.items():
                metrics[key].append(value)
        
        # Aggregate statistics
        def compute_stats(arr):
            arr = np.array(arr)
            mean = arr.mean()
            std = arr.std(ddof=1)
            sem = std / np.sqrt(len(arr))
            ci_95 = 1.96 * sem
            return mean, std, sem, ci_95
        
        aggregated = {'snr_db': snr_db, 'policy': policy}
        
        for key in metrics.keys():
            mean, std, sem, ci = compute_stats(metrics[key])
            aggregated[f'{key}_mean'] = mean
            aggregated[f'{key}_std'] = std
            aggregated[f'{key}_ci95'] = ci
        
        # Compute derived metrics
        aggregated['outage_probability'] = (
            (aggregated['outage_weak_mean'] + aggregated['outage_strong_mean']) / 2
        )
        aggregated['cache_hit_rate'] = (
            (aggregated['cache_hit_weak_mean'] + aggregated['cache_hit_strong_mean']) / 2
        )
        aggregated['sic_success_rate'] = aggregated['sic_success_mean']
        aggregated['cic_benefit_rate'] = (
            aggregated['cic_benefit_mean'] / max(aggregated['cic_opportunity_mean'], 1)
        )
        
        # Fairness
        fairness_values = []
        for i in range(self.num_realizations):
            rates = [metrics['rate_weak'][i], metrics['rate_strong'][i]]
            fairness_values.append(self.compute_jains_fairness(rates))
        aggregated['fairness_mean'] = np.mean(fairness_values)
        
        # Energy efficiency
        total_energy = aggregated['energy_mean'] * self.num_realizations
        total_bits = aggregated['sum_rate_mean'] * self.num_realizations
        aggregated['energy_efficiency'] = total_bits / max(total_energy, 1e-12)
        
        return aggregated
    
    def run_full_comparison(self, policies: List[str] = None) -> pd.DataFrame:
        """Run comprehensive comparison."""
        if policies is None:
            policies = ['topk', 'lru', 'lfu', 'random', 'none']
            if self.trained_dqn_cache is not None:
                policies.append('dqn')
        
        print(f"\n{'='*70}")
        print("RUNNING COMPREHENSIVE COMPARISON")
        print(f"{'='*70}")
        
        policy_names = []
        for p in policies:
            if p == 'none':
                policy_names.append('NO-CACHE')
            elif p == 'dqn':
                policy_names.append('DQN (trained)')
            else:
                policy_names.append(p.upper())
        
        print(f"Policies: {', '.join(policy_names)}")
        print(f"SNR range: {self.snr_db_range[0]} to {self.snr_db_range[-1]} dB")
        print(f"Monte Carlo runs per point: {self.num_realizations}\n")
        
        all_results = []
        
        for policy in policies:
            policy_name = policy.upper() if policy != 'none' else 'NO-CACHE'
            if policy == 'dqn':
                policy_name = 'DQN (trained)'
            
            print(f"\nProcessing {policy_name} policy...")
            
            for idx, snr_db in enumerate(self.snr_db_range):
                print(f"  SNR = {snr_db:+3d} dB ({idx+1}/{len(self.snr_db_range)})", end='\r')
                
                result = self.run_single_snr(
                    snr_db, 
                    policy=policy,
                    seed_offset=idx * 10000
                )
                all_results.append(result)
            
            print(f"  ✅ {policy_name} completed" + " "*30)
        
        df = pd.DataFrame(all_results)
        
        print(f"\n{'='*70}")
        print(f"✅ COMPARISON COMPLETE")
        print(f"   Total data points: {len(df)}")
        print(f"{'='*70}\n")
        
        return df
    
    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    
    def plot_main_comparison(self, df: pd.DataFrame, save_path: str = None):
        """
        Create comprehensive 9-subplot comparison.
        
        ✅ INCLUDES TRAINED DQN (distinct from NO-CACHE)
        """
        fig = plt.figure(figsize=(20, 14))
        fig.suptitle('Cache-Aided NOMA vs Traditional NOMA: Comprehensive Analysis',
                    fontsize=18, fontweight='bold', y=0.995)
        
        policies = df['policy'].unique()
        
        # ✅ FIXED: Distinct colors for DQN and NO-CACHE
        policy_colors = {
            'topk': '#1f77b4',      # Blue
            'lru': '#2ca02c',       # Green
            'lfu': '#9467bd',       # Purple
            'random': '#ff7f0e',    # Orange
            'none': '#FFD700',      # YELLOW (NO-CACHE baseline)
            'dqn': '#00CED1'        # CYAN (trained DQN)
        }
        
        policy_markers = {
            'topk': 'o',
            'lru': 's',
            'lfu': '^',
            'random': 'D',
            'none': 'P',   # Plus for NO-CACHE
            'dqn': '*'     # Star for DQN
        }
        
        def get_linewidth(policy):
            return 3.0 if policy == 'none' else 2.5 if policy == 'dqn' else 2.0
        
        def get_label(policy):
            if policy == 'none':
                return 'NO-CACHE'
            elif policy == 'dqn':
                return 'DQN (trained)'
            else:
                return policy.upper()
        
        # Helper function for plotting
        def plot_metric(ax, metric, title, ylabel, log_scale=False):
            for policy in policies:
                data = df[df['policy'] == policy]
                label = get_label(policy)
                
                if log_scale:
                    ax.semilogy(data['snr_db'], data[metric],
                               marker=policy_markers.get(policy, 'o'),
                               label=label,
                               color=policy_colors.get(policy, 'gray'),
                               linewidth=get_linewidth(policy),
                               markersize=6)
                else:
                    ax.plot(data['snr_db'], data[metric],
                           marker=policy_markers.get(policy, 'o'),
                           label=label,
                           color=policy_colors.get(policy, 'gray'),
                           linewidth=get_linewidth(policy),
                           markersize=6)
                    
                    # Add CI if available
                    ci_key = metric.replace('_mean', '_ci95')
                    if ci_key in data.columns:
                        ax.fill_between(data['snr_db'],
                                       data[metric] - data[ci_key],
                                       data[metric] + data[ci_key],
                                       alpha=0.15,
                                       color=policy_colors.get(policy, 'gray'))
            
            ax.set_xlabel('SNR (dB)', fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, which='both' if log_scale else 'major')
            ax.legend(fontsize=9, loc='best')
        
        # 1. Sum-Rate
        ax1 = plt.subplot(3, 3, 1)
        plot_metric(ax1, 'sum_rate_mean', 'Sum-Rate vs SNR', 'Sum-Rate (bps/Hz)')
        
        # 2. Outage Probability
        ax2 = plt.subplot(3, 3, 2)
        plot_metric(ax2, 'outage_probability', 'Outage Probability vs SNR', 'Outage Probability', log_scale=True)
        
        # 3. Cache Hit Rate
        ax3 = plt.subplot(3, 3, 3)
        for policy in policies:
            if policy == 'none':
                continue
            data = df[df['policy'] == policy]
            label = get_label(policy)
            ax3.plot(data['snr_db'], data['cache_hit_rate'],
                    marker=policy_markers.get(policy, '^'),
                    label=label,
                    color=policy_colors.get(policy, 'gray'),
                    linewidth=2.0,
                    markersize=6)
        ax3.set_xlabel('SNR (dB)', fontsize=11)
        ax3.set_ylabel('Cache Hit Rate', fontsize=11)
        ax3.set_title('Cache Hit Rate vs SNR', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=9, loc='best')
        
        # 4. Weak User Rate
        ax4 = plt.subplot(3, 3, 4)
        plot_metric(ax4, 'rate_weak_mean', 'Weak User Rate vs SNR', 'Rate (bps/Hz)')
        
        # 5. Strong User Rate
        ax5 = plt.subplot(3, 3, 5)
        plot_metric(ax5, 'rate_strong_mean', 'Strong User Rate vs SNR', 'Rate (bps/Hz)')
        
        # 6. BER Weak
        ax6 = plt.subplot(3, 3, 6)
        plot_metric(ax6, 'ber_weak_mean', 'BER vs SNR (Weak User)', 'BER', log_scale=True)
        
        # 7. BER Strong
        ax7 = plt.subplot(3, 3, 7)
        plot_metric(ax7, 'ber_strong_mean', 'BER vs SNR (Strong User)', 'BER', log_scale=True)
        
        # 8. Fairness
        ax8 = plt.subplot(3, 3, 8)
        plot_metric(ax8, 'fairness_mean', 'Fairness vs SNR', "Jain's Fairness Index")
        
        # 9. Energy Efficiency
        ax9 = plt.subplot(3, 3, 9)
        plot_metric(ax9, 'energy_efficiency', 'Energy Efficiency vs SNR', 'Energy Efficiency (bits/J)')
        
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save_results(self, df: pd.DataFrame, save_dir: str = 'results'):
        """Save results to CSV and generate summary."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save full dataset
        csv_path = os.path.join(save_dir, 'comparative_analysis_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"✅ Saved: {csv_path}")
        
        # Generate summary
        summary_path = os.path.join(save_dir, 'performance_summary.txt')
        
        try:
            with open(summary_path, 'w') as f:
                f.write("="*70 + "\n")
                f.write("CACHE-AIDED NOMA COMPARATIVE ANALYSIS SUMMARY\n")
                f.write("="*70 + "\n\n")
                
                high_snr = df['snr_db'].max()
                f.write(f"Performance at SNR = {high_snr} dB:\n")
                f.write("-"*70 + "\n\n")
                
                high_snr_data = df[df['snr_db'] == high_snr]
                
                for policy in high_snr_data['policy'].unique():
                    policy_data = high_snr_data[high_snr_data['policy'] == policy]
                    
                    if len(policy_data) == 0:
                        continue
                    
                    policy_row = policy_data.iloc[0]
                    
                    if policy == 'none':
                        policy_name = 'NO-CACHE'
                    elif policy == 'dqn':
                        policy_name = 'DQN (trained)'
                    else:
                        policy_name = policy.upper()
                    
                    f.write(f"{policy_name}:\n")
                    f.write(f"  Sum-Rate: {policy_row['sum_rate_mean']:.4f} bps/Hz\n")
                    f.write(f"  Outage Prob: {policy_row['outage_probability']:.6f}\n")
                    f.write(f"  Cache Hit Rate: {policy_row.get('cache_hit_rate', 0.0):.4f}\n")
                    f.write(f"  Fairness: {policy_row['fairness_mean']:.4f}\n")
                    f.write(f"  Energy Efficiency: {policy_row['energy_efficiency']:.2f} bits/J\n\n")
                
                f.write("="*70 + "\n")
            
            print(f"✅ Saved: {summary_path}")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not generate summary: {e}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Run comprehensive comparative analysis with automatic DQN training.
    """
    print("\n" + "#"*80)
    print("#" + " "*10 + "CACHE-AIDED NOMA COMPARATIVE ANALYSIS" + " "*12 + "#")
    print("#" + " "*15 + "(with Auto-Training DQN)" + " "*20 + "#")
    print("#"*80 + "\n")
    
    # ========================================================================
    # STEP 1: Check for trained DQN
    # ========================================================================
    
    trained_dqn_cache = None
    
    if HAS_DQN:
        checkpoint_path = check_dqn_checkpoint()
        
        if checkpoint_path:
            print(f"✅ Found trained DQN checkpoint: {checkpoint_path}")
            trained_dqn_cache = load_trained_dqn(checkpoint_path, cfg)
        else:
            print("⚠️  No trained DQN checkpoint found.")
            
            # Auto-train DQN
            checkpoint_path = train_dqn_automatically(
                cfg,
                num_episodes=cfg.RL_TRAINING_EPISODES
            )
            
            if checkpoint_path:
                trained_dqn_cache = load_trained_dqn(checkpoint_path, cfg)
            else:
                print("\n⚠️  Proceeding without DQN policy.\n")
    
    # ========================================================================
    # STEP 2: Run comprehensive comparison
    # ========================================================================
    
    analyzer = CacheAidedNOMAAnalysis(
        cfg,
        snr_range_db=np.arange(-10, 31, 2),
        num_realizations=1000,
        trained_dqn_cache=trained_dqn_cache
    )
    
    # Build policy list
    policies = ['topk', 'lru', 'lfu', 'random', 'none']
    if trained_dqn_cache is not None:
        policies.append('dqn')
    
    df = analyzer.run_full_comparison(policies=policies)
    
    # ========================================================================
    # STEP 3: Save results and plots
    # ========================================================================
    
    os.makedirs('results', exist_ok=True)
    analyzer.save_results(df, save_dir='results')
    
    analyzer.plot_main_comparison(
        df,
        save_path='results/cache_aided_vs_traditional_noma.png'
    )
    
    print("\n" + "#"*80)
    print("#" + " "*25 + "ANALYSIS COMPLETE" + " "*24 + "#")
    print("#"*80 + "\n")
    
    # Performance summary
    if trained_dqn_cache is not None:
        print("✅ DQN included in comparison (trained model)")
    else:
        print("⚠️  DQN not included (training skipped or failed)")
    
    print("\nGenerated files:")
    print("  • results/cache_aided_vs_traditional_noma.png")
    print("  • results/comparative_analysis_results.csv")
    print("  • results/performance_summary.txt")
    
    if trained_dqn_cache is not None:
        print("  • models/dqn_cache/dqn_cache_final.pth (DQN checkpoint)")
    
    print("\n")


if __name__ == "__main__":
    main()