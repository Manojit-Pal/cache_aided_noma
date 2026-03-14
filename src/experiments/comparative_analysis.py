# src/experiments/comparative_analysis.py
"""
Comprehensive Comparative Analysis: Cache-Aided NOMA vs Traditional Systems

✅ AUTO-TRAINS DQN IF NEEDED (Dec 12, 2025)

Bug Fix History (2026 Audit):
- BUG-CA-1 (CRITICAL): LRU/LFU/Random caches never warmed up → 0.0 hit rate
- BUG-CA-2 (CRITICAL): CIC flags used own-file instead of partner-file
- BUG-CA-3 (HIGH):     sum_rate ignored cache-rate overrides for partial hits
- BUG-CA-4 (MEDIUM):   DQN cache state bled across SNR evaluation points

Research References:
- arXiv:1712.09557 (2018): "Cache-Aided Non-Orthogonal Multiple Access"
- arXiv:1909.11074 (2019): "Power Allocation in Cache-Aided NOMA"
- IEEE Survey (2022): "A Survey on Applications of Cache-Aided NOMA"
- arXiv:2209.07809 (2022): "M2DQN - DQN training requirements"

Author: Cache-Aided NOMA Team
Date: December 12, 2025
Version: 4.1 (2026 Bug-Fix Revision)
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
        from src.simulation.stable_dqn_sim import NOMADQNTrainer

        print("\n" + "#"*80)
        print("#" + " "*20 + "AUTO-TRAINING DQN CACHE" + " "*24 + "#")
        print("#"*80)
        print(f"\nNo trained DQN checkpoint found.")
        print(f"Training will take approximately 10-30 minutes.")
        print(f"Episodes: {num_episodes}")
        print(f"\nResults will be saved to: models/dqn_cache/dqn_cache_final.pth")
        print(f"\nPress Ctrl+C to cancel and run without DQN.\n")

        response = input("Proceed with DQN training? [Y/n]: ").strip().lower()
        if response and response not in ['y', 'yes']:
            print("\n⚠️  Skipping DQN training. Will run comparison without DQN.")
            return None

        print("\n✅ Starting DQN training...\n")

        trainer = NOMADQNTrainer(cfg)
        trained_cache, train_history = trainer.train(
            num_episodes=num_episodes,
            test_interval=50,
            save_best=True
        )

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
        cache = DQNCache(
            capacity=cfg.CACHE_SIZE,
            num_files=cfg.NUM_FILES,
            num_users=cfg.NUM_USERS,
            learning_rate=cfg.RL_LEARNING_RATE,
            gamma=cfg.RL_GAMMA,
            seed=cfg.RANDOM_SEED
        )
        cache.load_model(checkpoint_path)
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

    Compares:
    - Cache-Aided NOMA (TopK, LRU, LFU, Random, DQN)
    - Traditional NOMA (no cache)
    """

    def __init__(self, cfg, snr_range_db: np.ndarray = None,
                 num_realizations: int = 1000,
                 trained_dqn_cache: Optional[object] = None):
        self.cfg = cfg
        self.snr_db_range = snr_range_db if snr_range_db is not None else np.arange(-10, 31, 2)
        self.num_realizations = num_realizations
        self.trained_dqn_cache = trained_dqn_cache
        self.results = defaultdict(list)

        print(f"\n✅ CacheAidedNOMAAnalysis initialized")
        print(f"   SNR range: {self.snr_db_range[0]} to {self.snr_db_range[-1]} dB")
        print(f"   Realizations: {self.num_realizations}")

        policies_list = ['TopK', 'LRU', 'LFU', 'Random', 'NO-CACHE']
        if trained_dqn_cache is not None:
            policies_list.append('DQN (trained)')
        print(f"   Policies: {', '.join(policies_list)}\n")

    def db_to_linear(self, db_value):
        return 10 ** (db_value / 10.0)

    def linear_to_db(self, linear_value):
        return 10 * np.log10(np.maximum(linear_value, 1e-12))

    def compute_jains_fairness(self, rates: List[float]) -> float:
        """
        Compute Jain's Fairness Index.
        J = (sum(r_i))^2 / (n * sum(r_i^2)),  range [1/n, 1]
        """
        rates = np.array(rates)
        n = len(rates)
        if n == 0 or rates.sum() == 0:
            return 0.0
        return (rates.sum() ** 2) / (n * (rates ** 2).sum())

    # ========================================================================
    # BUG-CA-1 FIX: Cache Warmup Helper
    # ========================================================================

    def _warmup_cache(self, cache, zipf_probs: np.ndarray, seed: int):
        """
        Warm up a dynamic cache (LRU / LFU / Random) by replaying a
        representative stream of Zipf-distributed requests.

        BUG-CA-1 fix:
            Previously LRU/LFU/Random caches were queried with
            update_stats=False, so nothing ever entered them and
            is_hit() always returned False (0.0 hit rate).

            We feed CACHE_SIZE * 10 requests (enough to fill the cache
            and establish a stable access-frequency distribution) using
            is_hit(update_stats=True), which lets LRU/LFU track recency
            and frequency and evict/admit as designed.

        StaticTopKCache and NO-CACHE (None) are skipped.
        DQN cache is also skipped (its warmup is handled by training).
        """
        if cache is None:
            return
        if isinstance(cache, (StaticTopKCache,)):
            return
        if HAS_DQN and isinstance(cache, DQNCache):
            return

        rng = np.random.default_rng(seed)
        warmup_size = self.cfg.CACHE_SIZE * 10
        warmup_requests = rng.choice(
            self.cfg.NUM_FILES,
            size=warmup_size,
            p=zipf_probs
        )
        for file_id in warmup_requests:
            cache.is_hit(int(file_id), update_stats=True)

    def setup_cache(self, policy: str, requests: np.ndarray = None,
                    warmup_seed: int = 0, zipf_probs: np.ndarray = None):
        """
        Create and warm up a cache instance for the given policy.

        BUG-CA-1 fix:
            _warmup_cache() is now called for every dynamic policy
            (LRU, LFU, Random) so the cache is populated before
            the Monte Carlo evaluation loop begins.

        BUG-CA-4 fix:
            DQN cache is flushed (clear()) between SNR evaluations so
            eviction history from one SNR point does not contaminate
            the next. Model weights are NOT reset.
        """
        if policy is None or policy == 'none':
            return None

        if policy == 'dqn':
            if self.trained_dqn_cache is None:
                print("⚠️  Warning: DQN requested but no trained cache available")
                return None
            # BUG-CA-4 FIX: flush contents so each SNR starts clean.
            # clear() resets stored files only; model weights are intact.
            self.trained_dqn_cache.clear()
            return self.trained_dqn_cache

        # Non-DQN caches
        cache = create_cache(policy, capacity=self.cfg.CACHE_SIZE)

        # Populate static caches
        if isinstance(cache, StaticTopKCache) and requests is not None:
            cnt = Counter(requests)
            ranking = [item for item, _ in cnt.most_common()]
            cache.populate(ranking)
        elif zipf_probs is not None:
            # BUG-CA-1 FIX: warm up dynamic caches (LRU, LFU, Random)
            self._warmup_cache(cache, zipf_probs, seed=warmup_seed)

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

        gain_weak   = np.min(channel_gains)
        gain_strong = np.max(channel_gains)

        gain_avg    = np.mean(channel_gains)
        noise_power = self.cfg.TX_POWER * gain_avg / self.db_to_linear(snr_db)

        return gain_weak, gain_strong, noise_power

    def simulate_noma_transmission(
        self,
        gain_weak: float, gain_strong: float,
        noise_power: float,
        cache=None,
        file_weak: int = 0, file_strong: int = 1
    ) -> Dict:
        """
        Simulate one NOMA transmission with optional cache assistance.

        BUG-CA-2 FIX:
            Previously simulate_sic_process() received
                weak_cached   = cache_hit_weak   (own file)
                strong_cached = cache_hit_strong (own file)
            CIC works by cancelling interference using the PARTNER's
            cached file, not one's own. Corrected to:
                weak_cached   = cache_hit_strong  (partner file)
                strong_cached = cache_hit_weak    (partner file)

        BUG-CA-3 FIX:
            sum_rate is now computed as rate_weak + rate_strong AFTER
            the cache-rate overrides, so partial cache hits are
            correctly reflected in sum_rate.
        """
        # ------------------------------------------------------------------
        # 1. Check cache status (own-file hits for delivery decisions)
        # ------------------------------------------------------------------
        if cache is not None:
            cache_hit_weak   = cache.is_hit(file_weak,   update_stats=False)
            cache_hit_strong = cache.is_hit(file_strong, update_stats=False)
        else:
            cache_hit_weak   = False
            cache_hit_strong = False

        cache_rate = getattr(self.cfg, 'CACHE_DELIVERY_RATE', self.cfg.TARGET_RATE_BPS)

        # ------------------------------------------------------------------
        # 2. Both users cached — no NOMA transmission needed
        # ------------------------------------------------------------------
        if cache_hit_weak and cache_hit_strong:
            return {
                'sinr_weak':            np.inf,
                'sinr_strong':          np.inf,
                'rate_weak':            cache_rate,
                'rate_strong':          cache_rate,
                'sum_rate':             2 * cache_rate,
                'outage_weak':          0,
                'outage_strong':        0,
                'ber_weak':             0.0,
                'ber_strong':           0.0,
                'cache_hit_weak':       1,
                'cache_hit_strong':     1,
                'transmission_needed':  0,
                'sic_success':          1,
                'cic_opportunity':      0,
                'cic_benefit':          0,
                'energy':               0.0,
            }

        # ------------------------------------------------------------------
        # 3. Power Allocation
        # ------------------------------------------------------------------
        p_weak, p_strong, feasible, alloc_info = allocate_power(
            gain_w=gain_weak,
            gain_s=gain_strong,
            cfg=self.cfg,
            method=self.cfg.POWER_ALLOC_METHOD,
            weak_cached=cache_hit_weak,
            strong_cached=cache_hit_strong,
            grid_points=self.cfg.POWER_ALLOC_GRID
        )

        sinr_threshold = sinr_threshold_from_rate(self.cfg.TARGET_RATE_BPS)

        # ------------------------------------------------------------------
        # 4. SIC / CIC Simulation
        #
        # BUG-CA-2 FIX: pass PARTNER's cache status, not own.
        #   weak user benefits from CIC when STRONG user's file is cached.
        #   strong user benefits when WEAK user's file is cached.
        # ------------------------------------------------------------------
        sic_results = simulate_sic_process(
            P_tx=self.cfg.TX_POWER,
            p_weak=p_weak,
            p_strong=p_strong,
            gain_w=gain_weak,
            gain_s=gain_strong,
            noise=noise_power,
            target_sinr=sinr_threshold,
            imperfection_factor=self.cfg.SIC_IMPERFECTION,
            weak_cached=cache_hit_strong,   # BUG-CA-2 FIX: partner file
            strong_cached=cache_hit_weak,   # BUG-CA-2 FIX: partner file
        )

        weak_success   = sic_results['weak_success']
        strong_success = sic_results['strong_success']

        # ------------------------------------------------------------------
        # 5. Rate computation (override for cache hits)
        # ------------------------------------------------------------------
        rate_weak   = cache_rate if cache_hit_weak   else (sic_results['rate_w'] if weak_success   else 0.0)
        rate_strong = cache_rate if cache_hit_strong else (sic_results['rate_s'] if strong_success else 0.0)

        # BUG-CA-3 FIX: derive sum_rate from the overridden rates,
        # not from sic_results['sum_rate'] which ignores cache_rate.
        sum_rate = rate_weak + rate_strong

        # ------------------------------------------------------------------
        # 6. Outage detection
        # ------------------------------------------------------------------
        outage_weak   = 0 if (cache_hit_weak   or weak_success)   else 1
        outage_strong = 0 if (cache_hit_strong or strong_success) else 1

        # ------------------------------------------------------------------
        # 7. BER (BPSK)
        # ------------------------------------------------------------------
        def compute_ber(sinr):
            return 0.5 * erfc(np.sqrt(np.maximum(sinr, 0.0)))

        ber_weak   = 0.0 if cache_hit_weak   else compute_ber(sic_results['sinr_w'])
        ber_strong = 0.0 if cache_hit_strong else compute_ber(sic_results['sinr_s_after'])

        # ------------------------------------------------------------------
        # 8. CIC tracking (cross-user: partner's file cached)
        # ------------------------------------------------------------------
        cic_opportunity = 0
        cic_benefit     = 0

        if cache_hit_strong and not cache_hit_weak:
            # Weak user can cancel strong's interference via CIC
            cic_opportunity += 1
            if weak_success:
                cic_benefit += 1

        if cache_hit_weak and not cache_hit_strong:
            # Strong user can cancel weak's interference via CIC
            cic_opportunity += 1
            if strong_success:
                cic_benefit += 1

        # ------------------------------------------------------------------
        # 9. Energy
        # ------------------------------------------------------------------
        energy = self.cfg.TX_POWER * (p_weak + p_strong)

        return {
            'sinr_weak':           sic_results['sinr_w'],
            'sinr_strong':         sic_results['sinr_s_after'],
            'rate_weak':           rate_weak,
            'rate_strong':         rate_strong,
            'sum_rate':            sum_rate,            # BUG-CA-3 FIX
            'outage_weak':         outage_weak,
            'outage_strong':       outage_strong,
            'ber_weak':            ber_weak,
            'ber_strong':          ber_strong,
            'cache_hit_weak':      int(cache_hit_weak),
            'cache_hit_strong':    int(cache_hit_strong),
            'transmission_needed': int(not (cache_hit_weak and cache_hit_strong)),
            'sic_success':         int(sic_results['can_decode_weak']),
            'cic_opportunity':     cic_opportunity,
            'cic_benefit':         cic_benefit,
            'energy':              energy,
        }

    def run_single_snr(self, snr_db: float, policy: str = 'topk',
                       seed_offset: int = 0) -> Dict:
        """
        Run Monte Carlo simulations for a single SNR point.

        BUG-CA-1 / BUG-CA-4 fix:
            setup_cache() now receives zipf_probs and a warmup seed so
            dynamic caches are pre-populated before the eval loop.
            DQN cache is flushed (clear()) to prevent cross-SNR contamination.
        """
        # Zipf probability distribution (shared by warmup + evaluation)
        ranks       = np.arange(1, self.cfg.NUM_FILES + 1)
        zipf_weights = 1.0 / np.power(ranks, self.cfg.ZIPF_ALPHA)
        zipf_probs  = zipf_weights / zipf_weights.sum()

        # Generate file requests for static-cache (TopK) population
        requests = sample_zipf_catalog(
            self.cfg.NUM_FILES,
            self.cfg.ZIPF_ALPHA,
            size=self.cfg.NUM_USERS * self.cfg.REQUESTS_PER_USER
        )

        # Setup + warm up cache (BUG-CA-1 + BUG-CA-4 fixes inside)
        warmup_seed = self.cfg.RANDOM_SEED + seed_offset
        cache = self.setup_cache(
            policy, requests,
            warmup_seed=warmup_seed,
            zipf_probs=zipf_probs
        )

        # Storage for per-realization metrics
        metrics = defaultdict(list)

        # Monte Carlo simulations
        for i in range(self.num_realizations):
            seed = self.cfg.RANDOM_SEED + seed_offset + i

            gain_weak, gain_strong, noise_power = self.generate_user_pair_channels(snr_db, seed)

            rng = np.random.default_rng(seed)
            file_weak   = int(rng.choice(self.cfg.NUM_FILES, p=zipf_probs))
            file_strong = int(rng.choice(self.cfg.NUM_FILES, p=zipf_probs))

            result = self.simulate_noma_transmission(
                gain_weak, gain_strong, noise_power,
                cache, file_weak, file_strong
            )

            for key, value in result.items():
                metrics[key].append(value)

        # Aggregate statistics
        def compute_stats(arr):
            arr = np.array(arr, dtype=float)
            mean = arr.mean()
            std  = arr.std(ddof=1)
            sem  = std / np.sqrt(len(arr))
            ci95 = 1.96 * sem
            return mean, std, sem, ci95

        aggregated = {'snr_db': snr_db, 'policy': policy}

        for key in metrics.keys():
            mean, std, sem, ci = compute_stats(metrics[key])
            aggregated[f'{key}_mean']  = mean
            aggregated[f'{key}_std']   = std
            aggregated[f'{key}_ci95']  = ci

        # Derived metrics
        aggregated['outage_probability'] = (
            (aggregated['outage_weak_mean'] + aggregated['outage_strong_mean']) / 2
        )
        aggregated['cache_hit_rate'] = (
            (aggregated['cache_hit_weak_mean'] + aggregated['cache_hit_strong_mean']) / 2
        )
        aggregated['sic_success_rate'] = aggregated['sic_success_mean']
        aggregated['cic_benefit_rate'] = (
            aggregated['cic_benefit_mean']
            / max(aggregated['cic_opportunity_mean'], 1e-9)
        )

        # Fairness (Jain's index per realization)
        fairness_values = [
            self.compute_jains_fairness([metrics['rate_weak'][i], metrics['rate_strong'][i]])
            for i in range(self.num_realizations)
        ]
        aggregated['fairness_mean'] = np.mean(fairness_values)

        # Energy efficiency
        total_energy = aggregated['energy_mean'] * self.num_realizations
        total_bits   = aggregated['sum_rate_mean'] * self.num_realizations
        aggregated['energy_efficiency'] = total_bits / max(total_energy, 1e-12)

        return aggregated

    def run_full_comparison(self, policies: List[str] = None) -> pd.DataFrame:
        """Run comprehensive comparison across all SNR points."""
        if policies is None:
            policies = ['topk', 'lru', 'lfu', 'random', 'none']
            if self.trained_dqn_cache is not None:
                policies.append('dqn')

        print(f"\n{'='*70}")
        print("RUNNING COMPREHENSIVE COMPARISON")
        print(f"{'='*70}")

        policy_names = []
        for p in policies:
            if p == 'none':  policy_names.append('NO-CACHE')
            elif p == 'dqn': policy_names.append('DQN (trained)')
            else:            policy_names.append(p.upper())

        print(f"Policies: {', '.join(policy_names)}")
        print(f"SNR range: {self.snr_db_range[0]} to {self.snr_db_range[-1]} dB")
        print(f"Monte Carlo runs per point: {self.num_realizations}\n")

        all_results = []

        for policy in policies:
            policy_name = (
                'DQN (trained)' if policy == 'dqn'
                else 'NO-CACHE' if policy == 'none'
                else policy.upper()
            )
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
        """Create comprehensive 9-subplot comparison."""
        fig = plt.figure(figsize=(20, 14))
        fig.suptitle(
            'Cache-Aided NOMA vs Traditional NOMA: Comprehensive Analysis',
            fontsize=18, fontweight='bold', y=0.995
        )

        policies = df['policy'].unique()

        policy_colors = {
            'topk':   '#1f77b4',
            'lru':    '#2ca02c',
            'lfu':    '#9467bd',
            'random': '#ff7f0e',
            'none':   '#FFD700',
            'dqn':    '#00CED1',
        }
        policy_markers = {
            'topk': 'o', 'lru': 's', 'lfu': '^',
            'random': 'D', 'none': 'P', 'dqn': '*',
        }

        def get_linewidth(policy): return 3.0 if policy == 'none' else 2.5 if policy == 'dqn' else 2.0
        def get_label(policy):
            if policy == 'none': return 'NO-CACHE'
            if policy == 'dqn':  return 'DQN (trained)'
            return policy.upper()

        def plot_metric(ax, metric, title, ylabel, log_scale=False):
            for policy in policies:
                data  = df[df['policy'] == policy]
                label = get_label(policy)
                if log_scale:
                    ax.semilogy(
                        data['snr_db'], data[metric],
                        marker=policy_markers.get(policy, 'o'),
                        label=label,
                        color=policy_colors.get(policy, 'gray'),
                        linewidth=get_linewidth(policy), markersize=6
                    )
                else:
                    ax.plot(
                        data['snr_db'], data[metric],
                        marker=policy_markers.get(policy, 'o'),
                        label=label,
                        color=policy_colors.get(policy, 'gray'),
                        linewidth=get_linewidth(policy), markersize=6
                    )
                    ci_key = metric.replace('_mean', '_ci95')
                    if ci_key in data.columns:
                        ax.fill_between(
                            data['snr_db'],
                            data[metric] - data[ci_key],
                            data[metric] + data[ci_key],
                            alpha=0.15, color=policy_colors.get(policy, 'gray')
                        )
            ax.set_xlabel('SNR (dB)', fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, which='both' if log_scale else 'major')
            ax.legend(fontsize=9, loc='best')

        ax1 = plt.subplot(3, 3, 1)
        plot_metric(ax1, 'sum_rate_mean', 'Sum-Rate vs SNR', 'Sum-Rate (bps/Hz)')

        ax2 = plt.subplot(3, 3, 2)
        plot_metric(ax2, 'outage_probability', 'Outage Probability vs SNR',
                    'Outage Probability', log_scale=True)

        ax3 = plt.subplot(3, 3, 3)
        for policy in policies:
            if policy == 'none':
                continue
            data  = df[df['policy'] == policy]
            label = get_label(policy)
            ax3.plot(
                data['snr_db'], data['cache_hit_rate'],
                marker=policy_markers.get(policy, '^'),
                label=label,
                color=policy_colors.get(policy, 'gray'),
                linewidth=2.0, markersize=6
            )
        ax3.set_xlabel('SNR (dB)', fontsize=11)
        ax3.set_ylabel('Cache Hit Rate', fontsize=11)
        ax3.set_title('Cache Hit Rate vs SNR', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=9, loc='best')

        ax4 = plt.subplot(3, 3, 4)
        plot_metric(ax4, 'rate_weak_mean', 'Weak User Rate vs SNR', 'Rate (bps/Hz)')

        ax5 = plt.subplot(3, 3, 5)
        plot_metric(ax5, 'rate_strong_mean', 'Strong User Rate vs SNR', 'Rate (bps/Hz)')

        ax6 = plt.subplot(3, 3, 6)
        plot_metric(ax6, 'ber_weak_mean', 'BER vs SNR (Weak User)', 'BER', log_scale=True)

        ax7 = plt.subplot(3, 3, 7)
        plot_metric(ax7, 'ber_strong_mean', 'BER vs SNR (Strong User)', 'BER', log_scale=True)

        ax8 = plt.subplot(3, 3, 8)
        plot_metric(ax8, 'fairness_mean', 'Fairness vs SNR', "Jain's Fairness Index")

        ax9 = plt.subplot(3, 3, 9)
        plot_metric(ax9, 'energy_efficiency', 'Energy Efficiency vs SNR',
                    'Energy Efficiency (bits/J)')

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

        csv_path = os.path.join(save_dir, 'comparative_analysis_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"✅ Saved: {csv_path}")

        summary_path = os.path.join(save_dir, 'performance_summary.txt')
        try:
            with open(summary_path, 'w') as f:
                f.write("="*70 + "\n")
                f.write("CACHE-AIDED NOMA COMPARATIVE ANALYSIS SUMMARY\n")
                f.write("="*70 + "\n\n")

                high_snr      = df['snr_db'].max()
                high_snr_data = df[df['snr_db'] == high_snr]

                f.write(f"Performance at SNR = {high_snr} dB:\n")
                f.write("-"*70 + "\n\n")

                for policy in high_snr_data['policy'].unique():
                    policy_data = high_snr_data[high_snr_data['policy'] == policy]
                    if len(policy_data) == 0:
                        continue
                    row = policy_data.iloc[0]
                    if policy == 'none':  policy_name = 'NO-CACHE'
                    elif policy == 'dqn': policy_name = 'DQN (trained)'
                    else:                 policy_name = policy.upper()

                    f.write(f"{policy_name}:\n")
                    f.write(f"  Sum-Rate: {row['sum_rate_mean']:.4f} bps/Hz\n")
                    f.write(f"  Outage Prob: {row['outage_probability']:.6f}\n")
                    f.write(f"  Cache Hit Rate: {row.get('cache_hit_rate', 0.0):.4f}\n")
                    f.write(f"  Fairness: {row['fairness_mean']:.4f}\n")
                    f.write(f"  Energy Efficiency: {row['energy_efficiency']:.2f} bits/J\n\n")

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
            checkpoint_path = train_dqn_automatically(
                cfg, num_episodes=cfg.RL_TRAINING_EPISODES
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
