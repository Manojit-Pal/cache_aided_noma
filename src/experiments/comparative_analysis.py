# src/experiments/comparative_analysis.py
"""
Comparative Analysis: Cache-Aided NOMA vs Traditional NOMA
Implements:
1. Average Sum-Rate vs SNR (both: wireless-transmissions-only & per-request delivered)
2. Individual User Rates (Far/Near) vs SNR
3. Outage Probability vs SNR
4. BER vs SNR

Outputs:
 - cache_vs_nocache_comparison.png
 - performance_gain_with_cache.png
 - results_cache_aided_noma.csv
 - results_traditional_noma.csv
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
import pandas as pd
from typing import Dict, List, Tuple
from collections import Counter

from src import config as cfg
from src.noma import channel_model
from src.noma.power_allocation import allocate_power_gridsearch
from src.utils import set_seed, sample_zipf_catalog
from src.caching.static_cache import StaticTopKCache

# ---------------------------------------------------------
class ComparativeNOMAAnalysis:
    def __init__(self, cfg):
        self.cfg = cfg
        self.snr_db_range = np.arange(-10, 30, 2)  # SNR from -10 to 28 dB
        self.num_realizations = 1000  # Monte Carlo realizations per SNR

    def db_to_linear(self, db_value):
        return 10 ** (db_value / 10.0)

    def linear_to_db(self, linear_value):
        return 10 * np.log10(linear_value + 1e-12)

    def setup_cache(self, cache_enabled=True):
        """Setup static Top-K cache based on popularity (if requested)."""
        if not cache_enabled:
            return None

        requests = sample_zipf_catalog(
            self.cfg.NUM_FILES,
            self.cfg.ZIPF_ALPHA,
            size=self.cfg.NUM_USERS * self.cfg.REQUESTS_PER_USER
        )
        cnt = Counter(requests)
        ranking = [item for item, _ in cnt.most_common()]

        cache = StaticTopKCache(self.cfg.CACHE_SIZE)
        cache.populate(ranking)
        return cache

    def generate_user_pair_channels(self, snr_db):
        """
        Generate channel gains for a weak-strong user pair.
        Returns: gain_weak, gain_strong, noise_power
        """
        positions = channel_model.generate_user_positions(2, self.cfg.CELL_RADIUS)
        distances = positions[:, 2]

        pl = np.array([
            channel_model.pathloss(d, self.cfg.PATHLOSS_EXPONENT, self.cfg.MIN_DISTANCE)
            for d in distances
        ])

        small_scale = channel_model.rayleigh_gain(2)
        channel_gains = pl * small_scale

        gain_weak = np.min(channel_gains)
        gain_strong = np.max(channel_gains)

        # set noise_power such that avg SNR equals desired SNR
        gain_avg = np.mean(channel_gains)
        noise_power = self.cfg.TX_POWER * gain_avg / self.db_to_linear(snr_db)

        return gain_weak, gain_strong, noise_power

    def compute_achievable_rate(self, sinr):
        return np.log2(1 + sinr)

    def compute_ber_bpsk(self, sinr):
        # Guard negative/inf:
        sinr_val = np.maximum(sinr, 0.0)
        return 0.5 * erfc(np.sqrt(sinr_val))

    def simulate_noma_transmission(self, gain_weak, gain_strong, noise_power,
                                   cache_hit_weak=False, cache_hit_strong=False):
        """
        Simulate NOMA transmission for a user pair and return metrics.
        """
        P = self.cfg.TX_POWER
        sinr_threshold = 2 ** self.cfg.TARGET_RATE_BPS - 1

        # If both cached => no transmission needed
        if cache_hit_weak and cache_hit_strong:
            # Both served from cache: assume effective local delivery rate (fast)
            cache_rate = getattr(self.cfg, "CACHE_DELIVERY_RATE", self.cfg.TARGET_RATE_BPS)
            return {
                'sinr_weak': np.inf,
                'sinr_strong': np.inf,
                'rate_weak': cache_rate,
                'rate_strong': cache_rate,
                'outage_weak': 0,
                'outage_strong': 0,
                'ber_weak': 0,
                'ber_strong': 0,
                'transmission_needed': False
            }


        # Power allocation (grid-search)
        p_w, p_s, feasible, alloc_info = allocate_power_gridsearch(
            gain_weak, gain_strong, self.cfg, grid_points=self.cfg.POWER_ALLOC_GRID
        )

        # Weak user SINR (if not cached)
        if not cache_hit_weak:
            sinr_weak = (P * p_w * gain_weak) / (P * p_s * gain_weak + noise_power)
        else:
            sinr_weak = np.inf

        # Strong user: decode weak first -> SIC -> decode own signal (if not cached)
        if not cache_hit_strong:
            sinr_strong_decode_weak = (P * p_w * gain_strong) / (P * p_s * gain_strong + noise_power)

            # If strong can decode weak signal, assume SIC imperfectness factor applies
            if sinr_strong_decode_weak >= sinr_threshold:
                residual = self.cfg.SIC_IMPERFECTION * (P * p_w * gain_strong)
            else:
                residual = P * p_w * gain_strong  # SIC failed -> treat as full interference

            sinr_strong = (P * p_s * gain_strong) / (noise_power + residual)
        else:
            sinr_strong = np.inf

        rate_weak = self.compute_achievable_rate(sinr_weak)
        rate_strong = self.compute_achievable_rate(sinr_strong)

        outage_weak = 1 if rate_weak < self.cfg.TARGET_RATE_BPS else 0
        outage_strong = 1 if rate_strong < self.cfg.TARGET_RATE_BPS else 0

        ber_weak = self.compute_ber_bpsk(sinr_weak)
        ber_strong = self.compute_ber_bpsk(sinr_strong)

        return {
            'sinr_weak': sinr_weak,
            'sinr_strong': sinr_strong,
            'rate_weak': rate_weak,
            'rate_strong': rate_strong,
            'outage_weak': outage_weak,
            'outage_strong': outage_strong,
            'ber_weak': ber_weak,
            'ber_strong': ber_strong,
            'transmission_needed': True
        }

    def run_comparison_single_snr(self, snr_db, cache_enabled=True):
        """
        Runs Monte Carlo realizations for a single SNR.
        Returns averaged metrics (per-request delivered and transmissions-only).
        """
        cache = self.setup_cache(cache_enabled) if cache_enabled else None

        # storage for per-realization values
        delivered_rates_per_request = []
        delivered_rates_transmissions_only = []  # counts only actual wireless transmissions
        rate_weak_list = []
        rate_strong_list = []
        outage_weak_list = []
        outage_strong_list = []
        ber_weak_list = []
        ber_strong_list = []

        # Cache hit counters
        cache_hits_weak = 0
        cache_hits_strong = 0

        for _ in range(self.num_realizations):
            gain_weak, gain_strong, noise_power = self.generate_user_pair_channels(snr_db)

            # simulate file requests using Zipf popularity
            if cache is not None:
                p = self._get_zipf_probs()
                file_weak = np.random.choice(self.cfg.NUM_FILES, p=p)
                file_strong = np.random.choice(self.cfg.NUM_FILES, p=p)

                cache_hit_weak = cache.is_hit(file_weak)
                cache_hit_strong = cache.is_hit(file_strong)

                if cache_hit_weak:
                    cache_hits_weak += 1
                if cache_hit_strong:
                    cache_hits_strong += 1
            else:
                cache_hit_weak = False
                cache_hit_strong = False

            outcome = self.simulate_noma_transmission(
                gain_weak, gain_strong, noise_power,
                cache_hit_weak=cache_hit_weak, cache_hit_strong=cache_hit_strong
            )

            # --- delivered rate PER-REQUEST (count cache hits as delivered)
            cache_rate = getattr(self.cfg, "CACHE_DELIVERY_RATE", self.cfg.TARGET_RATE_BPS)

            if not outcome['transmission_needed']:
                # both cached case handled inside simulate
                delivered_rate_weak = cache_rate if cache_hit_weak else 0.0
                delivered_rate_strong = cache_rate if cache_hit_strong else 0.0
            else:
                delivered_rate_weak = cache_rate if cache_hit_weak else outcome['rate_weak']
                delivered_rate_strong = cache_rate if cache_hit_strong else outcome['rate_strong']

            delivered_rates_per_request.append(delivered_rate_weak + delivered_rate_strong)

            # --- wireless transmissions only (if transmission occurs)
            if outcome['transmission_needed']:
                transmitted_sum = 0.0
                if not cache_hit_weak:
                    transmitted_sum += outcome['rate_weak']
                if not cache_hit_strong:
                    transmitted_sum += outcome['rate_strong']
            else:
                transmitted_sum = 0.0
            delivered_rates_transmissions_only.append(transmitted_sum)

            # preserve individual metrics (for averages & BER/outage)
            rate_weak_list.append(delivered_rate_weak)
            rate_strong_list.append(delivered_rate_strong)
            outage_weak_list.append(outcome['outage_weak'])
            outage_strong_list.append(outcome['outage_strong'])
            ber_weak_list.append(outcome['ber_weak'])
            ber_strong_list.append(outcome['ber_strong'])

        # compute averages and standard errors
        def mean_sem(arr):
            a = np.array(arr)
            return a.mean(), a.std(ddof=1) / np.sqrt(len(a))

        avg_sum_rate_per_request, sem_sum_rate_pr = mean_sem(delivered_rates_per_request)
        avg_sum_rate_tx_only, sem_sum_rate_tx = mean_sem(delivered_rates_transmissions_only)

        avg_rate_weak, sem_rate_weak = mean_sem(rate_weak_list)
        avg_rate_strong, sem_rate_strong = mean_sem(rate_strong_list)

        avg_outage_weak, sem_outage_weak = mean_sem(outage_weak_list)
        avg_outage_strong, sem_outage_strong = mean_sem(outage_strong_list)

        avg_ber_weak, sem_ber_weak = mean_sem(ber_weak_list)
        avg_ber_strong, sem_ber_strong = mean_sem(ber_strong_list)

        avg_results = {
            'snr_db': snr_db,
            # per-request (counts cache hits as delivered)
            'avg_sum_rate_per_request': avg_sum_rate_per_request,
            'sem_sum_rate_per_request': sem_sum_rate_pr,
            # transmissions-only (wireless spectral usage)
            'avg_sum_rate_tx_only': avg_sum_rate_tx_only,
            'sem_sum_rate_tx_only': sem_sum_rate_tx,
            # per-user delivered rates (already count cache hits)
            'avg_rate_weak': avg_rate_weak,
            'sem_rate_weak': sem_rate_weak,
            'avg_rate_strong': avg_rate_strong,
            'sem_rate_strong': sem_rate_strong,
            # outage & BER
            'outage_prob_weak': avg_outage_weak,
            'sem_outage_weak': sem_outage_weak,
            'outage_prob_strong': avg_outage_strong,
            'sem_outage_strong': sem_outage_strong,
            'avg_ber_weak': avg_ber_weak,
            'sem_ber_weak': sem_ber_weak,
            'avg_ber_strong': avg_ber_strong,
            'sem_ber_strong': sem_ber_strong,
            # cache hit info
            'cache_hits_weak': cache_hits_weak,
            'cache_hits_strong': cache_hits_strong,
            'cache_hit_rate_fraction': (cache_hits_weak + cache_hits_strong) / (2.0 * self.num_realizations)
        }

        return avg_results

    def _get_zipf_probs(self):
        ranks = np.arange(1, self.cfg.NUM_FILES + 1)
        weights = 1.0 / np.power(ranks, self.cfg.ZIPF_ALPHA)
        return weights / weights.sum()

    def run_full_comparison(self):
        print("Running Cache-Aided NOMA vs Traditional NOMA Comparison...")
        results_with_cache = []
        results_without_cache = []

        for i, snr_db in enumerate(self.snr_db_range):
            print(f"Processing SNR = {snr_db} dB ({i+1}/{len(self.snr_db_range)})...", end='\r')
            res_cache = self.run_comparison_single_snr(snr_db, cache_enabled=True)
            res_no_cache = self.run_comparison_single_snr(snr_db, cache_enabled=False)
            results_with_cache.append(res_cache)
            results_without_cache.append(res_no_cache)

        df_with_cache = pd.DataFrame(results_with_cache)
        df_without_cache = pd.DataFrame(results_without_cache)
        return df_with_cache, df_without_cache

    def plot_all_comparisons(self, df_cache, df_no_cache, save_dir='./'):
        fig = plt.figure(figsize=(18, 12))

        # 1. Average Sum-Rate vs SNR (Per-request, counts cache hits)
        ax1 = plt.subplot(2, 3, 1)
        ax1.errorbar(df_cache['snr_db'], df_cache['avg_sum_rate_per_request'],
                     yerr=df_cache['sem_sum_rate_per_request'], fmt='b-o', label='Cache-Aided (per-request)')
        ax1.errorbar(df_no_cache['snr_db'], df_no_cache['avg_sum_rate_per_request'],
                     yerr=df_no_cache['sem_sum_rate_per_request'], fmt='r--s', label='Traditional (per-request)')
        # also show transmissions-only for clarity (dashed/dotted)
        ax1.plot(df_cache['snr_db'], df_cache['avg_sum_rate_tx_only'], 'b:', label='Cache-Aided (tx-only)')
        ax1.plot(df_no_cache['snr_db'], df_no_cache['avg_sum_rate_tx_only'], 'r:', label='Traditional (tx-only)')
        ax1.set_xlabel('SNR (dB)'); ax1.set_ylabel('Average Sum-Rate (bits/s/Hz)')
        ax1.set_title('Sum-Rate: per-request (solid) & tx-only (dotted)')
        ax1.grid(True); ax1.legend(fontsize=9)

        # 2. Far User (Weak) Rate vs SNR
        ax2 = plt.subplot(2, 3, 2)
        ax2.errorbar(df_cache['snr_db'], df_cache['avg_rate_weak'], yerr=df_cache['sem_rate_weak'],
                     fmt='b-o', label='Cache-Aided')
        ax2.errorbar(df_no_cache['snr_db'], df_no_cache['avg_rate_weak'], yerr=df_no_cache['sem_rate_weak'],
                     fmt='r--s', label='Traditional')
        ax2.set_xlabel('SNR (dB)'); ax2.set_ylabel('Average Rate (bits/s/Hz)')
        ax2.set_title('Far User (R1) Rate vs SNR'); ax2.grid(True); ax2.legend(fontsize=9)

        # 3. Near User (Strong) Rate vs SNR
        ax3 = plt.subplot(2, 3, 3)
        ax3.errorbar(df_cache['snr_db'], df_cache['avg_rate_strong'], yerr=df_cache['sem_rate_strong'],
                     fmt='b-o', label='Cache-Aided')
        ax3.errorbar(df_no_cache['snr_db'], df_no_cache['avg_rate_strong'], yerr=df_no_cache['sem_rate_strong'],
                     fmt='r--s', label='Traditional')
        ax3.set_xlabel('SNR (dB)'); ax3.set_ylabel('Average Rate (bits/s/Hz)')
        ax3.set_title('Near User (R2) Rate vs SNR'); ax3.grid(True); ax3.legend(fontsize=9)

        # 4. Average Outage Probability vs SNR (average of weak & strong)
        ax4 = plt.subplot(2, 3, 4)
        outage_cache = (df_cache['outage_prob_weak'] + df_cache['outage_prob_strong']) / 2
        outage_no_cache = (df_no_cache['outage_prob_weak'] + df_no_cache['outage_prob_strong']) / 2
        ax4.semilogy(df_cache['snr_db'], outage_cache, 'b-o', label='Cache-Aided')
        ax4.semilogy(df_no_cache['snr_db'], outage_no_cache, 'r--s', label='Traditional')
        ax4.set_xlabel('SNR (dB)'); ax4.set_ylabel('Outage Probability')
        ax4.set_title('Average Outage Probability vs SNR'); ax4.grid(True, which='both'); ax4.legend(fontsize=9)

        # 5. BER vs SNR (Far User)
        ax5 = plt.subplot(2, 3, 5)
        ax5.semilogy(df_cache['snr_db'], df_cache['avg_ber_weak'], 'b-o', label='Cache-Aided')
        ax5.semilogy(df_no_cache['snr_db'], df_no_cache['avg_ber_weak'], 'r--s', label='Traditional')
        ax5.set_xlabel('SNR (dB)'); ax5.set_ylabel('Bit Error Rate (BER)')
        ax5.set_title('BER vs SNR (Far User)'); ax5.grid(True, which='both'); ax5.legend(fontsize=9)

        # 6. BER vs SNR (Near User)
        ax6 = plt.subplot(2, 3, 6)
        ax6.semilogy(df_cache['snr_db'], df_cache['avg_ber_strong'], 'b-o', label='Cache-Aided')
        ax6.semilogy(df_no_cache['snr_db'], df_no_cache['avg_ber_strong'], 'r--s', label='Traditional')
        ax6.set_xlabel('SNR (dB)'); ax6.set_ylabel('Bit Error Rate (BER)')
        ax6.set_title('BER vs SNR (Near User)'); ax6.grid(True, which='both'); ax6.legend(fontsize=9)

        plt.tight_layout()
        plt.savefig(f'{save_dir}/cache_vs_nocache_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\nSaved: {save_dir}/cache_vs_nocache_comparison.png")
        plt.close()

        # Performance gain plots
        self._plot_performance_gain(df_cache, df_no_cache, save_dir)

    def _plot_performance_gain(self, df_cache, df_no_cache, save_dir):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Sum-rate gain (per-request)
        sum_rate_gain = ((df_cache['avg_sum_rate_per_request'] - df_no_cache['avg_sum_rate_per_request']) /
                         (df_no_cache['avg_sum_rate_per_request'] + 1e-12) * 100)
        axes[0, 0].plot(df_cache['snr_db'], sum_rate_gain, 'g-o')
        axes[0, 0].set_title('Sum-Rate Improvement with Cache (%)'); axes[0, 0].grid(True)
        axes[0, 0].axhline(y=0, color='black', linestyle='--')

        # Outage reduction
        oc_cache = (df_cache['outage_prob_weak'] + df_cache['outage_prob_strong']) / 2
        oc_no_cache = (df_no_cache['outage_prob_weak'] + df_no_cache['outage_prob_strong']) / 2
        outage_reduction = ((oc_no_cache - oc_cache) / (oc_no_cache + 1e-12) * 100)
        axes[0, 1].plot(df_cache['snr_db'], outage_reduction, 'm-o')
        axes[0, 1].set_title('Outage Probability Reduction with Cache (%)'); axes[0, 1].grid(True)
        axes[0, 1].axhline(y=0, color='black', linestyle='--')

        # BER reduction (Far User)
        ber_weak_reduction = ((df_no_cache['avg_ber_weak'] - df_cache['avg_ber_weak']) /
                              (df_no_cache['avg_ber_weak'] + 1e-12) * 100)
        axes[1, 0].plot(df_cache['snr_db'], ber_weak_reduction, 'c-o')
        axes[1, 0].set_title('BER Reduction with Cache - Far User (%)'); axes[1, 0].grid(True)
        axes[1, 0].axhline(y=0, color='black', linestyle='--')

        # BER reduction (Near User)
        ber_strong_reduction = ((df_no_cache['avg_ber_strong'] - df_cache['avg_ber_strong']) /
                                (df_no_cache['avg_ber_strong'] + 1e-12) * 100)
        axes[1, 1].plot(df_cache['snr_db'], ber_strong_reduction, color='orange', marker='o')
        axes[1, 1].set_title('BER Reduction with Cache - Near User (%)'); axes[1, 1].grid(True)
        axes[1, 1].axhline(y=0, color='black', linestyle='--')

        plt.tight_layout()
        plt.savefig(f'{save_dir}/performance_gain_with_cache.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {save_dir}/performance_gain_with_cache.png")
        plt.close()

    def print_summary(self, df_cache, df_no_cache):
        print("\n" + "="*70)
        print("COMPARATIVE ANALYSIS SUMMARY: Cache-Aided vs Traditional NOMA")
        print("="*70)

        high_snr_idx = -1
        snr_val = df_cache['snr_db'].iloc[high_snr_idx]

        print(f"\nAt SNR = {snr_val} dB:")
        print("-" * 70)

        # Sum-rate (per-request)
        sum_cache = df_cache['avg_sum_rate_per_request'].iloc[high_snr_idx]
        sum_no_cache = df_no_cache['avg_sum_rate_per_request'].iloc[high_snr_idx]
        gain = (sum_cache - sum_no_cache) / (sum_no_cache + 1e-12) * 100
        print(f"  Sum-Rate (per-request) - Cache-Aided: {sum_cache:.4f}, Traditional: {sum_no_cache:.4f}, Improvement: {gain:+.2f}%")

        # Outage
        out_cache = (df_cache['outage_prob_weak'].iloc[high_snr_idx] + df_cache['outage_prob_strong'].iloc[high_snr_idx]) / 2
        out_no_cache = (df_no_cache['outage_prob_weak'].iloc[high_snr_idx] + df_no_cache['outage_prob_strong'].iloc[high_snr_idx]) / 2
        reduction = (out_no_cache - out_cache) / (out_no_cache + 1e-12) * 100
        print(f"  Outage (avg) - Cache-Aided: {out_cache:.6f}, Traditional: {out_no_cache:.6f}, Reduction: {reduction:+.2f}%")

        # BER far user
        ber_cache = df_cache['avg_ber_weak'].iloc[high_snr_idx]
        ber_no_cache = df_no_cache['avg_ber_weak'].iloc[high_snr_idx]
        ber_reduction = (ber_no_cache - ber_cache) / (ber_no_cache + 1e-12) * 100
        print(f"  BER Far - Cache-Aided: {ber_cache:.6e}, Traditional: {ber_no_cache:.6e}, Reduction: {ber_reduction:+.2f}%")

        # Cache hit rates (last SNR)
        print("\nCache hit rates (per pair selection):")
        print(f"  Cache-Aided hits fraction (weak+strong averaged): {df_cache['cache_hit_rate_fraction'].iloc[high_snr_idx]:.4f}")

        print("\n" + "="*70)

# -------------------------
def main():
    print("="*70)
    print("CACHE-AIDED NOMA vs TRADITIONAL NOMA COMPARATIVE ANALYSIS")
    print("="*70)

    analyzer = ComparativeNOMAAnalysis(cfg)
    df_cache, df_no_cache = analyzer.run_full_comparison()

    df_cache.to_csv('results_cache_aided_noma.csv', index=False)
    df_no_cache.to_csv('results_traditional_noma.csv', index=False)
    print("\nResults saved:")
    print("  - results_cache_aided_noma.csv")
    print("  - results_traditional_noma.csv")

    analyzer.plot_all_comparisons(df_cache, df_no_cache)
    analyzer.print_summary(df_cache, df_no_cache)

    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()
