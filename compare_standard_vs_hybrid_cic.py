# compare_standard_vs_hybrid_cic.py
"""
Standard CIC vs Hybrid CIC Comparison using DQN Caching (16 users)

This script trains a single DQN cache, then evaluates it under two modes:
  1. Standard CIC  — power allocation is cache-BLIND (closedform)
                      CIC still applies at the decoding/SIC level
  2. Hybrid CIC    — power allocation is cache-AWARE
                      CIC applies at both power allocation AND decoding

The DQN cache is trained ONCE and shared between both evaluations so
the ONLY variable is the power allocation strategy.

Author: Cache-Aided NOMA Team
Date: July 2026
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # non-interactive backend
import matplotlib.pyplot as plt
import os, sys, copy, warnings
from collections import defaultdict, Counter
from typing import Dict, List, Optional
from scipy.special import erfc

warnings.filterwarnings('ignore')

# ── project imports ──────────────────────────────────────────────────────────
from src import config as cfg
from src.utils import set_seed, sample_zipf_catalog

from src.noma import (
    generate_user_positions,
    compute_channel_gains,
    pair_users,
    allocate_power,
    simulate_sic_process,
    sinr_threshold_from_rate,
    rate_from_sinr,
)

from src.caching import (
    create_cache,
    StaticTopKCache,
    LRUCache,
    LFUCache,
    RandomCache,
)

try:
    from src.caching import DQNCache
    HAS_DQN = True
except ImportError:
    HAS_DQN = False
    print("⚠️  DQN not available — aborting.")
    sys.exit(1)

from src.simulation.stable_dqn_sim import NOMADQNTrainer


# ═══════════════════════════════════════════════════════════════════════════
# 1.  DQN TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def train_dqn_cache(training_episodes: int = 150) -> 'DQNCache':
    """
    Train a DQN cache using the current config (16 users, debug-scale).

    Returns the trained DQN cache in eval mode.
    """
    # Use debug-scale for fast training
    cfg.set_debug_config()

    print("\n" + "=" * 70)
    print("  TRAINING DQN CACHE")
    print("=" * 70)
    print(f"  NUM_USERS         : {cfg.NUM_USERS}")
    print(f"  NUM_FILES         : {cfg.NUM_FILES}")
    print(f"  CACHE_SIZE        : {cfg.CACHE_SIZE}")
    print(f"  Training episodes : {training_episodes}")
    print(f"  Steps/episode     : {cfg.NUM_USERS * cfg.REQUESTS_PER_USER}")
    print("=" * 70 + "\n")

    trainer = NOMADQNTrainer(cfg, verbose=True)
    trained_cache, train_history = trainer.train(
        num_episodes=training_episodes,
        test_interval=25,
        save_best=True,
    )

    # Switch to evaluation mode
    trained_cache.set_eval_mode(True)
    trained_cache.epsilon = 0.0
    print("\n✅ DQN training complete — switched to eval mode.\n")
    return trained_cache


# ═══════════════════════════════════════════════════════════════════════════
# 2.  EVALUATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class CICComparisonEngine:
    """
    Runs Monte-Carlo evaluations of a trained DQN cache under
    Standard CIC and Hybrid CIC power allocation.
    """

    def __init__(self, trained_dqn: 'DQNCache',
                 snr_range_db: np.ndarray = None,
                 num_realizations: int = 500):
        self.dqn_cache = trained_dqn
        self.snr_db_range = snr_range_db if snr_range_db is not None else np.arange(-10, 31, 2)
        self.num_realizations = num_realizations

        print(f"CICComparisonEngine initialised")
        print(f"  SNR range        : {self.snr_db_range[0]} → {self.snr_db_range[-1]} dB")
        print(f"  Realisations/SNR : {self.num_realizations}")

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _db2lin(db):
        return 10 ** (db / 10.0)

    def _warmup_dqn(self, zipf_probs: np.ndarray, seed: int):
        """Flush and re-fill DQN cache using the trained Q-network."""
        cache = self.dqn_cache
        cache.clear()
        cache.set_eval_mode(False)
        cache.epsilon = 0.0
        rng = np.random.default_rng(seed)
        for fid in rng.choice(cfg.NUM_FILES,
                              size=cfg.CACHE_SIZE * 10,
                              p=zipf_probs):
            cache.request(int(fid))
        cache.set_eval_mode(True)
        cache.reset_stats()

    # ── single SNR point ─────────────────────────────────────────────────

    def _run_single_snr(self, snr_db: float, pa_method: str,
                        seed_offset: int) -> Dict:
        """
        Monte-Carlo simulation for one (SNR, PA-method) point.

        pa_method: 'closedform' → Standard CIC
                   'cache_aware' → Hybrid CIC
        """
        # Zipf popularity
        ranks = np.arange(1, cfg.NUM_FILES + 1)
        zipf_w = 1.0 / np.power(ranks, cfg.ZIPF_ALPHA)
        zipf_probs = zipf_w / zipf_w.sum()

        # Warm-up DQN cache for this SNR point
        self._warmup_dqn(zipf_probs, seed=cfg.RANDOM_SEED + seed_offset)

        metrics = defaultdict(list)
        sinr_threshold = sinr_threshold_from_rate(cfg.TARGET_RATE_BPS)
        cache_rate = getattr(cfg, 'CACHE_DELIVERY_RATE', cfg.TARGET_RATE_BPS)

        for i in range(self.num_realizations):
            seed = cfg.RANDOM_SEED + seed_offset + i
            set_seed(seed)

            # ── channel generation ──
            positions = generate_user_positions(2, cfg.CELL_RADIUS, seed=seed)
            channel_gains = compute_channel_gains(
                positions,
                exponent=cfg.PATHLOSS_EXPONENT,
                fading_type=cfg.FADING_TYPE,
                K_factor_db=cfg.RICIAN_K_FACTOR_DB,
                los_probability=cfg.LOS_PROBABILITY,
            )
            gain_weak = float(np.min(channel_gains))
            gain_strong = float(np.max(channel_gains))

            # normalise noise to match SNR
            gain_avg = float(np.mean(channel_gains))
            noise_power = cfg.TX_POWER * gain_avg / self._db2lin(snr_db)

            # ── file requests ──
            rng = np.random.default_rng(seed)
            file_weak = int(rng.choice(cfg.NUM_FILES, p=zipf_probs))
            file_strong = int(rng.choice(cfg.NUM_FILES, p=zipf_probs))

            # ── cache status ──
            hit_weak = self.dqn_cache.is_hit(file_weak, update_stats=False)
            hit_strong = self.dqn_cache.is_hit(file_strong, update_stats=False)

            # ── both cached → served from cache ──
            if hit_weak and hit_strong:
                metrics['sum_rate'].append(2 * cache_rate)
                metrics['outage_weak'].append(0)
                metrics['outage_strong'].append(0)
                metrics['ber_weak'].append(0.0)
                metrics['ber_strong'].append(0.0)
                metrics['cache_hit_weak'].append(1)
                metrics['cache_hit_strong'].append(1)
                metrics['energy'].append(0.0)
                metrics['cic_opportunity'].append(0)
                metrics['cic_benefit'].append(0)
                metrics['sic_success'].append(1)
                continue

            # ── power allocation (the key difference) ──
            p_w, p_s, feasible, alloc_info = allocate_power(
                gain_w=gain_weak,
                gain_s=gain_strong,
                cfg=cfg,
                method=pa_method,
                weak_cached=hit_weak,
                strong_cached=hit_strong,
                grid_points=cfg.POWER_ALLOC_GRID,
            )

            # ── SIC / CIC simulation ──
            # Pass PARTNER cache flags (BUG-CA-2 convention)
            sic_results = simulate_sic_process(
                P_tx=cfg.TX_POWER,
                p_weak=p_w,
                p_strong=p_s,
                gain_w=gain_weak,
                gain_s=gain_strong,
                noise=noise_power,
                target_sinr=sinr_threshold,
                imperfection_factor=cfg.SIC_IMPERFECTION,
                weak_cached=hit_strong,     # partner's file cached
                strong_cached=hit_weak,     # partner's file cached
            )

            # ── rates ──
            rate_w = cache_rate if hit_weak else (
                sic_results['rate_w'] if sic_results['weak_success'] else 0.0)
            rate_s = cache_rate if hit_strong else (
                sic_results['rate_s'] if sic_results['strong_success'] else 0.0)
            sum_rate = rate_w + rate_s

            outage_w = 0 if (hit_weak or sic_results['weak_success']) else 1
            outage_s = 0 if (hit_strong or sic_results['strong_success']) else 1

            def _ber(sinr):
                return 0.5 * erfc(np.sqrt(max(sinr, 0.0)))

            ber_w = 0.0 if hit_weak else _ber(sic_results['sinr_w'])
            ber_s = 0.0 if hit_strong else _ber(sic_results['sinr_s_after'])

            # CIC tracking
            cic_opp, cic_ben = 0, 0
            if hit_strong and not hit_weak:
                cic_opp += 1
                if sic_results['weak_success']:
                    cic_ben += 1
            if hit_weak and not hit_strong:
                cic_opp += 1
                if sic_results['strong_success']:
                    cic_ben += 1

            energy = cfg.TX_POWER * (p_w + p_s)

            metrics['sum_rate'].append(sum_rate)
            metrics['outage_weak'].append(outage_w)
            metrics['outage_strong'].append(outage_s)
            metrics['ber_weak'].append(ber_w)
            metrics['ber_strong'].append(ber_s)
            metrics['cache_hit_weak'].append(int(hit_weak))
            metrics['cache_hit_strong'].append(int(hit_strong))
            metrics['energy'].append(energy)
            metrics['cic_opportunity'].append(cic_opp)
            metrics['cic_benefit'].append(cic_ben)
            metrics['sic_success'].append(int(sic_results['can_decode_weak']))

        # ── aggregate ──
        agg = {'snr_db': snr_db, 'pa_method': pa_method}
        for k, v in metrics.items():
            arr = np.array(v)
            agg[f'{k}_mean'] = arr.mean()
            agg[f'{k}_std'] = arr.std(ddof=1) if len(arr) > 1 else 0.0
        agg['outage_prob'] = (agg['outage_weak_mean'] + agg['outage_strong_mean']) / 2
        agg['cache_hit_rate'] = (agg['cache_hit_weak_mean'] + agg['cache_hit_strong_mean']) / 2
        agg['cic_benefit_rate'] = (
            agg['cic_benefit_mean'] / max(agg['cic_opportunity_mean'], 1e-9))
        total_e = agg['energy_mean'] * self.num_realizations
        total_b = agg['sum_rate_mean'] * self.num_realizations
        agg['energy_efficiency'] = total_b / max(total_e, 1e-12)
        return agg

    # ── full sweep ───────────────────────────────────────────────────────

    def run_comparison(self) -> pd.DataFrame:
        """Evaluate over all SNR points for both Standard and Hybrid CIC."""
        all_rows: List[Dict] = []

        for label, pa_method in [('Standard CIC', 'closedform'),
                                  ('Hybrid CIC', 'cache_aware')]:
            print(f"\n▶ Evaluating {label} (PA={pa_method}) …")
            for idx, snr_db in enumerate(self.snr_db_range):
                print(f"  SNR = {snr_db:+3.0f} dB  ({idx + 1}/{len(self.snr_db_range)})",
                      end='\r')
                row = self._run_single_snr(snr_db, pa_method,
                                           seed_offset=idx * 10_000)
                row['label'] = label
                all_rows.append(row)
            print(f"  ✅ {label} done" + " " * 30)

        return pd.DataFrame(all_rows)


# ═══════════════════════════════════════════════════════════════════════════
# 3.  PLOTTING
# ═══════════════════════════════════════════════════════════════════════════

def plot_comparison(df: pd.DataFrame, save_dir: str = 'results'):
    """Create publication-quality comparison plots."""
    os.makedirs(save_dir, exist_ok=True)

    colors = {'Standard CIC': '#e74c3c', 'Hybrid CIC': '#2ecc71'}
    markers = {'Standard CIC': 's', 'Hybrid CIC': 'o'}

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Standard CIC vs Hybrid CIC  (DQN Cache, 16 Users)',
                 fontsize=16, fontweight='bold', y=0.99)

    plot_specs = [
        ('sum_rate_mean',   'Sum Rate (bps/Hz)',         'Sum Rate vs SNR',               False),
        ('outage_prob',     'Outage Probability',        'Outage Probability vs SNR',     True),
        ('ber_weak_mean',   'BER (Weak User)',           'BER Weak User vs SNR',          True),
        ('ber_strong_mean', 'BER (Strong User)',         'BER Strong User vs SNR',        True),
        ('energy_efficiency', 'Energy Efficiency (bps/Hz/W)', 'Energy Efficiency vs SNR', False),
        ('cic_benefit_rate', 'CIC Benefit Rate',         'CIC Benefit Rate vs SNR',       False),
    ]

    for ax, (metric, ylabel, title, logscale) in zip(axes.flat, plot_specs):
        for label in ['Standard CIC', 'Hybrid CIC']:
            sub = df[df['label'] == label].sort_values('snr_db')
            ax.plot(sub['snr_db'], sub[metric],
                    color=colors[label], marker=markers[label],
                    linewidth=2.2, markersize=6, label=label)
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11, fontweight='bold')
        if logscale:
            ax.set_yscale('log')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(save_dir, 'standard_vs_hybrid_cic_comparison.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Plot saved → {out_path}")


def print_summary(df: pd.DataFrame):
    """Print a tabular summary of improvements."""
    print("\n" + "=" * 70)
    print("  STANDARD CIC vs HYBRID CIC — SUMMARY  (DQN, 16 users)")
    print("=" * 70)

    std = df[df['label'] == 'Standard CIC']
    hyb = df[df['label'] == 'Hybrid CIC']

    metrics = [
        ('sum_rate_mean',      'Avg Sum Rate',       'bps/Hz', False),
        ('outage_prob',        'Avg Outage Prob',     '',       True),
        ('ber_weak_mean',      'Avg BER (weak)',      '',       True),
        ('ber_strong_mean',    'Avg BER (strong)',    '',       True),
        ('energy_efficiency',  'Avg Energy Eff',      'bps/Hz/W', False),
        ('cic_benefit_rate',   'Avg CIC Benefit',     '',       False),
    ]

    print(f"\n{'Metric':<22} {'Standard CIC':>14} {'Hybrid CIC':>14} {'Improvement':>14}")
    print("-" * 66)

    for key, name, unit, lower_better in metrics:
        val_s = std[key].mean()
        val_h = hyb[key].mean()
        if lower_better:
            if val_s > 1e-12:
                imp = (val_s - val_h) / val_s * 100
            else:
                imp = 0.0
            sign = '↓'
        else:
            if val_s > 1e-12:
                imp = (val_h - val_s) / val_s * 100
            else:
                imp = 0.0
            sign = '↑'
        unit_str = f" {unit}" if unit else ""
        print(f"  {name:<20} {val_s:>13.4f}{unit_str} {val_h:>13.4f}{unit_str} {imp:>+10.1f}% {sign}")

    print("=" * 70 + "\n")


# ═══════════════════════════════════════════════════════════════════════════
# 4.  MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # ── Step 1: Train DQN ──
    trained_dqn = train_dqn_cache(training_episodes=80)

    # ── Step 2: Evaluate both modes ──
    engine = CICComparisonEngine(
        trained_dqn,
        snr_range_db=np.arange(-10, 31, 2),
        num_realizations=500,
    )
    results_df = engine.run_comparison()

    # ── Step 3: Save CSV ──
    os.makedirs('results_csv', exist_ok=True)
    csv_path = 'results_csv/standard_vs_hybrid_cic.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"✅ CSV saved → {csv_path}")

    # ── Step 4: Plot ──
    plot_comparison(results_df, save_dir='results')

    # ── Step 5: Summary ──
    print_summary(results_df)

    print("🎉 Comparison complete!")
