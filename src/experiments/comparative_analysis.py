# src/experiments/comparative_analysis.py
"""
Comprehensive Comparative Analysis: Cache-Aided NOMA vs Traditional Systems

✅ AUTO-TRAINS DQN IF NEEDED (Dec 12, 2025)

Bug Fix History (2026 Audit):
- BUG-CA-1 (CRITICAL): LRU/LFU/Random caches never warmed up → 0.0 hit rate
- BUG-CA-2 (CRITICAL): CIC flags used own-file instead of partner-file
- BUG-CA-3 (HIGH):     sum_rate ignored cache-rate overrides for partial hits
- BUG-CA-4 (MEDIUM):   DQN cache not properly reset+refilled between SNR points
                        Revised: clear() followed by request()-based warmup
                        so the trained policy actually fills the cache.

Research References:
- arXiv:1712.09557 (2018): "Cache-Aided Non-Orthogonal Multiple Access"
- arXiv:1909.11074 (2019): "Power Allocation in Cache-Aided NOMA"
- IEEE Survey (2022): "A Survey on Applications of Cache-Aided NOMA"
- arXiv:2209.07809 (2022): "M2DQN - DQN training requirements"

Author: Cache-Aided NOMA Team
Date: December 12, 2025
Version: 4.2 (2026 Bug-Fix Revision 2)
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
    checkpoint_path = 'models/dqn_cache/dqn_cache_final.pth'
    if os.path.exists(checkpoint_path):
        return checkpoint_path
    alt_path = 'models/dqn_cache/dqn_cache_best_ep999.pth'
    if os.path.exists(alt_path):
        return alt_path
    return None


def train_dqn_automatically(cfg, num_episodes: int = 1000) -> Optional[str]:
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
            num_episodes=num_episodes, test_interval=50, save_best=True
        )
        checkpoint_path = check_dqn_checkpoint()
        if checkpoint_path:
            print(f"\n✅ DQN training complete! Checkpoint: {checkpoint_path}")
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
        Jain's Fairness Index: J = (sum r_i)^2 / (n * sum r_i^2), range [1/n, 1]
        """
        rates = np.array(rates)
        n = len(rates)
        if n == 0 or rates.sum() == 0:
            return 0.0
        return (rates.sum() ** 2) / (n * (rates ** 2).sum())

    # =========================================================================
    # BUG-CA-1 FIX: Dynamic cache warmup (LRU / LFU / Random)
    # =========================================================================

    def _warmup_cache(self, cache, zipf_probs: np.ndarray, seed: int):
        """
        Warm up a dynamic cache (LRU / LFU / Random) before evaluation.

        BUG-CA-1 fix:
            Fresh empty LRU/LFU/Random caches were checked via
            is_hit(update_stats=False) so nothing ever entered them
            and hit rate was always 0.0.

            We feed CACHE_SIZE * 10 Zipf-sampled requests through
            is_hit(update_stats=True) so LRU/LFU track recency and
            frequency and the cache fills up to capacity before eval.

        StaticTopKCache and DQNCache are handled separately and skipped here.
        """
        if cache is None:
            return
        if isinstance(cache, StaticTopKCache):
            return
        if HAS_DQN and isinstance(cache, DQNCache):
            return

        rng = np.random.default_rng(seed)
        warmup_requests = rng.choice(
            self.cfg.NUM_FILES,
            size=self.cfg.CACHE_SIZE * 10,
            p=zipf_probs
        )
        for file_id in warmup_requests:
            cache.is_hit(int(file_id), update_stats=True)

    # =========================================================================
    # BUG-CA-4 FIX (REVISED): DQN reset + refill via request()
    # =========================================================================

    def _reset_dqn_for_eval(self, zipf_probs: np.ndarray, seed: int):
        """
        Reset and re-warm the DQN cache for a new SNR evaluation point.

        BUG-CA-4 fix (revised):
            The earlier fix called clear() alone, which flushed all cached
            files. In eval mode the DQN never calls _learn_from_request(),
            so _execute_action() is never triggered and the cache stayed
            empty for all subsequent SNR points -> 0.0 hit rate.

            Correct approach:
              1. clear() flushes stale contents from the previous SNR run.
              2. Re-warm via request() (NOT is_hit()) for CACHE_SIZE * 10
                 Zipf-sampled file IDs. In eval mode, request() skips
                 learning but still calls _select_action() which calls
                 _execute_action(), so the trained DQN policy makes real
                 eviction/admission decisions and fills the cache.
              3. reset_stats() clears hit/miss counters accumulated during
                 warmup so the evaluation starts with clean statistics.

        Why request() and not is_hit():
            is_hit() only checks membership; it never inserts files.
            request() triggers the full DQN inference pipeline including
            action selection and cache insertion (_execute_action).
        """
        cache = self.trained_dqn_cache
        if cache is None:
            return

        # Step 1: flush stale cached files from the previous SNR point
        cache.clear()

        # Step 2: re-warm using the trained policy's inference path
        rng = np.random.default_rng(seed)
        warmup_requests = rng.choice(
            self.cfg.NUM_FILES,
            size=self.cfg.CACHE_SIZE * 10,
            p=zipf_probs
        )
        for file_id in warmup_requests:
            # request() in eval mode: no gradient update, but
            # _select_action() + _execute_action() DO run -> cache fills up
            cache.request(int(file_id))

        # Step 3: clear warmup statistics so eval counters start at 0
        cache.reset_stats()

    # =========================================================================
    # CACHE SETUP
    # =========================================================================

    def setup_cache(self, policy: str, requests: np.ndarray = None,
                    warmup_seed: int = 0, zipf_probs: np.ndarray = None):
        """
        Create and warm up a cache instance for the given policy.

        BUG-CA-1 fix: _warmup_cache() populates LRU/LFU/Random.
        BUG-CA-4 fix: _reset_dqn_for_eval() flushes and re-warms DQN.
        """
        if policy is None or policy == 'none':
            return None

        if policy == 'dqn':
            if self.trained_dqn_cache is None:
                print("⚠️  Warning: DQN requested but no trained cache available")
                return None
            # BUG-CA-4 REVISED FIX: flush + request()-based warmup
            if zipf_probs is not None:
                self._reset_dqn_for_eval(zipf_probs, seed=warmup_seed)
            return self.trained_dqn_cache

        # Non-DQN caches
        cache = create_cache(policy, capacity=self.cfg.CACHE_SIZE)

        if isinstance(cache, StaticTopKCache) and requests is not None:
            cnt     = Counter(requests)
            ranking = [item for item, _ in cnt.most_common()]
            cache.populate(ranking)
        elif zipf_probs is not None:
            # BUG-CA-1 FIX: warm up dynamic caches (LRU, LFU, Random)
            self._warmup_cache(cache, zipf_probs, seed=warmup_seed)

        return cache

    # =========================================================================
    # CHANNEL GENERATION
    # =========================================================================

    def generate_user_pair_channels(self, snr_db: float,
                                    seed: int = None) -> Tuple:
        """Generate channel gains for a weak-strong user pair."""
        if seed is not None:
            set_seed(seed)

        positions = generate_user_positions(
            num_users=2, cell_radius=self.cfg.CELL_RADIUS, seed=seed
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

    # =========================================================================
    # NOMA TRANSMISSION SIMULATION
    # =========================================================================

    def simulate_noma_transmission(
        self,
        gain_weak: float, gain_strong: float, noise_power: float,
        cache=None, file_weak: int = 0, file_strong: int = 1
    ) -> Dict:
        """
        Simulate one NOMA transmission with optional cache assistance.

        BUG-CA-2 FIX:
            simulate_sic_process() now receives PARTNER file cache status:
                weak_cached   = cache_hit_strong  (strong's file cached)
                strong_cached = cache_hit_weak    (weak's file cached)
            Previously used own-file flags, which is physically wrong.

        BUG-CA-3 FIX:
            sum_rate = rate_weak + rate_strong (after cache-rate overrides).
            Previously used sic_results['sum_rate'] which ignored overrides.
        """
        # ------------------------------------------------------------------
        # 1. Cache status (own-file delivery checks)
        # ------------------------------------------------------------------
        if cache is not None:
            cache_hit_weak   = cache.is_hit(file_weak,   update_stats=False)
            cache_hit_strong = cache.is_hit(file_strong, update_stats=False)
        else:
            cache_hit_weak   = False
            cache_hit_strong = False

        cache_rate = getattr(self.cfg, 'CACHE_DELIVERY_RATE', self.cfg.TARGET_RATE_BPS)

        # ------------------------------------------------------------------
        # 2. Both cached: no NOMA transmission
        # ------------------------------------------------------------------
        if cache_hit_weak and cache_hit_strong:
            return {
                'sinr_weak':           np.inf,
                'sinr_strong':         np.inf,
                'rate_weak':           cache_rate,
                'rate_strong':         cache_rate,
                'sum_rate':            2 * cache_rate,
                'outage_weak':         0,
                'outage_strong':       0,
                'ber_weak':            0.0,
                'ber_strong':          0.0,
                'cache_hit_weak':      1,
                'cache_hit_strong':    1,
                'transmission_needed': 0,
                'sic_success':         1,
                'cic_opportunity':     0,
                'cic_benefit':         0,
                'energy':              0.0,
            }

        # ------------------------------------------------------------------
        # 3. Power allocation
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
        # 4. SIC/CIC simulation
        #    BUG-CA-2 FIX: pass PARTNER cache status, not own.
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
        # 5. Rates (override for cache hits)
        # ------------------------------------------------------------------
        rate_weak   = cache_rate if cache_hit_weak   else (sic_results['rate_w'] if weak_success   else 0.0)
        rate_strong = cache_rate if cache_hit_strong else (sic_results['rate_s'] if strong_success else 0.0)

        # BUG-CA-3 FIX: recompute sum_rate from overridden rates
        sum_rate = rate_weak + rate_strong

        # ------------------------------------------------------------------
        # 6. Outage
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
            cic_opportunity += 1
            if weak_success:
                cic_benefit += 1
        if cache_hit_weak and not cache_hit_strong:
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

    # =========================================================================
    # SINGLE SNR POINT
    # =========================================================================

    def run_single_snr(self, snr_db: float, policy: str = 'topk',
                       seed_offset: int = 0) -> Dict:
        """
        Run Monte Carlo simulations for a single SNR point.

        BUG-CA-1 / BUG-CA-4 fix:
            setup_cache() receives zipf_probs and warmup_seed.
            - Dynamic caches (LRU/LFU/Random): filled via is_hit() warmup.
            - DQN cache: flushed then re-filled via request() warmup.
        """
        # Zipf distribution (shared by warmup and evaluation)
        ranks        = np.arange(1, self.cfg.NUM_FILES + 1)
        zipf_weights = 1.0 / np.power(ranks, self.cfg.ZIPF_ALPHA)
        zipf_probs   = zipf_weights / zipf_weights.sum()

        # Static-cache population requests
        requests = sample_zipf_catalog(
            self.cfg.NUM_FILES, self.cfg.ZIPF_ALPHA,
            size=self.cfg.NUM_USERS * self.cfg.REQUESTS_PER_USER
        )

        warmup_seed = self.cfg.RANDOM_SEED + seed_offset
        cache = self.setup_cache(
            policy, requests,
            warmup_seed=warmup_seed,
            zipf_probs=zipf_probs
        )

        metrics = defaultdict(list)

        for i in range(self.num_realizations):
            seed = self.cfg.RANDOM_SEED + seed_offset + i
            gain_weak, gain_strong, noise_power = \
                self.generate_user_pair_channels(snr_db, seed)

            rng         = np.random.default_rng(seed)
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
            arr  = np.array(arr, dtype=float)
            mean = arr.mean()
            std  = arr.std(ddof=1)
            sem  = std / np.sqrt(len(arr))
            ci95 = 1.96 * sem
            return mean, std, sem, ci95

        aggregated = {'snr_db': snr_db, 'policy': policy}
        for key in metrics:
            mean, std, sem, ci = compute_stats(metrics[key])
            aggregated[f'{key}_mean'] = mean
            aggregated[f'{key}_std']  = std
            aggregated[f'{key}_ci95'] = ci

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

        fairness_values = [
            self.compute_jains_fairness([metrics['rate_weak'][i],
                                         metrics['rate_strong'][i]])
            for i in range(self.num_realizations)
        ]
        aggregated['fairness_mean'] = np.mean(fairness_values)

        total_energy = aggregated['energy_mean'] * self.num_realizations
        total_bits   = aggregated['sum_rate_mean'] * self.num_realizations
        aggregated['energy_efficiency'] = total_bits / max(total_energy, 1e-12)

        return aggregated

    # =========================================================================
    # FULL COMPARISON
    # =========================================================================

    def run_full_comparison(self, policies: List[str] = None) -> pd.DataFrame:
        """Run comprehensive comparison across all SNR points."""
        if policies is None:
            policies = ['topk', 'lru', 'lfu', 'random', 'none']
            if self.trained_dqn_cache is not None:
                policies.append('dqn')

        print(f"\n{'='*70}")
        print("RUNNING COMPREHENSIVE COMPARISON")
        print(f"{'='*70}")

        def pname(p):
            if p == 'none': return 'NO-CACHE'
            if p == 'dqn':  return 'DQN (trained)'
            return p.upper()

        print(f"Policies: {', '.join(pname(p) for p in policies)}")
        print(f"SNR range: {self.snr_db_range[0]} to {self.snr_db_range[-1]} dB")
        print(f"Monte Carlo runs per point: {self.num_realizations}\n")

        all_results = []
        for policy in policies:
            print(f"\nProcessing {pname(policy)} policy...")
            for idx, snr_db in enumerate(self.snr_db_range):
                print(f"  SNR = {snr_db:+3d} dB ({idx+1}/{len(self.snr_db_range)})",
                      end='\r')
                result = self.run_single_snr(
                    snr_db, policy=policy, seed_offset=idx * 10000
                )
                all_results.append(result)
            print(f"  ✅ {pname(policy)} completed" + " "*30)

        df = pd.DataFrame(all_results)
        print(f"\n{'='*70}")
        print(f"✅ COMPARISON COMPLETE — {len(df)} data points")
        print(f"{'='*70}\n")
        return df

    # =========================================================================
    # VISUALIZATION
    # =========================================================================

    def plot_main_comparison(self, df: pd.DataFrame, save_path: str = None):
        """Create comprehensive 9-subplot comparison."""
        fig = plt.figure(figsize=(20, 14))
        fig.suptitle(
            'Cache-Aided NOMA vs Traditional NOMA: Comprehensive Analysis',
            fontsize=18, fontweight='bold', y=0.995
        )
        policies = df['policy'].unique()

        policy_colors  = {'topk': '#1f77b4', 'lru': '#2ca02c', 'lfu': '#9467bd',
                          'random': '#ff7f0e', 'none': '#FFD700', 'dqn': '#00CED1'}
        policy_markers = {'topk': 'o', 'lru': 's', 'lfu': '^',
                          'random': 'D', 'none': 'P', 'dqn': '*'}

        def lw(p):  return 3.0 if p == 'none' else 2.5 if p == 'dqn' else 2.0
        def lbl(p): return 'NO-CACHE' if p == 'none' else 'DQN (trained)' if p == 'dqn' else p.upper()

        def plot_metric(ax, metric, title, ylabel, log_scale=False):
            for p in policies:
                d = df[df['policy'] == p]
                kw = dict(marker=policy_markers.get(p, 'o'), label=lbl(p),
                          color=policy_colors.get(p, 'gray'),
                          linewidth=lw(p), markersize=6)
                if log_scale:
                    ax.semilogy(d['snr_db'], d[metric], **kw)
                else:
                    ax.plot(d['snr_db'], d[metric], **kw)
                    ci_key = metric.replace('_mean', '_ci95')
                    if ci_key in d.columns:
                        ax.fill_between(d['snr_db'],
                                        d[metric] - d[ci_key],
                                        d[metric] + d[ci_key],
                                        alpha=0.15,
                                        color=policy_colors.get(p, 'gray'))
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
        for p in policies:
            if p == 'none': continue
            d = df[df['policy'] == p]
            ax3.plot(d['snr_db'], d['cache_hit_rate'],
                     marker=policy_markers.get(p, '^'), label=lbl(p),
                     color=policy_colors.get(p, 'gray'), linewidth=2.0, markersize=6)
        ax3.set_xlabel('SNR (dB)', fontsize=11)
        ax3.set_ylabel('Cache Hit Rate', fontsize=11)
        ax3.set_title('Cache Hit Rate vs SNR', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3); ax3.legend(fontsize=9, loc='best')

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

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================

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
                    pdata = high_snr_data[high_snr_data['policy'] == policy]
                    if len(pdata) == 0: continue
                    row  = pdata.iloc[0]
                    name = ('NO-CACHE'      if policy == 'none'
                            else 'DQN (trained)' if policy == 'dqn'
                            else policy.upper())
                    f.write(f"{name}:\n")
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
    print("\n" + "#"*80)
    print("#" + " "*10 + "CACHE-AIDED NOMA COMPARATIVE ANALYSIS" + " "*12 + "#")
    print("#" + " "*15 + "(with Auto-Training DQN)" + " "*20 + "#")
    print("#"*80 + "\n")

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

    os.makedirs('results', exist_ok=True)
    analyzer.save_results(df, save_dir='results')
    analyzer.plot_main_comparison(
        df, save_path='results/cache_aided_vs_traditional_noma.png'
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
        print("  • models/dqn_cache/dqn_cache_final.pth")
    print("\n")


if __name__ == "__main__":
    main()
