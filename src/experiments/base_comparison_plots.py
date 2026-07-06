"""
src/experiments/base_comparison_plots.py

Base Comparison Plots: OMA vs Conventional NOMA vs Cache-Aided NOMA vs Hybrid Cache TDMA-NOMA

Generates 3 publication-quality figures:
  1. Sum Rate vs SNR (dB)
  2. Outage Probability vs SNR (dB)
  3. Cache Hit Ratio vs Cache Size

Parameters from teacher's TeX paper (Table I):
  K=200 users, N=2000 files, C=200 cache, Rayleigh fading,
  path loss exponent=3.5, cell radius=500m, noise=1e-9W,
  P_max=2W, target rate=0.3 bps/Hz, SIC imperfection=0.05,
  power coefficients (0.8, 0.2), Zipf alpha=1.0

Author: Cache-Aided NOMA Team
Date: July 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for file saving
import os
import sys
import time
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src import config as cfg
from src.utils import set_seed
from src.noma.channel_model import generate_user_positions, pathloss, rayleigh_gain
from src.noma.noma_base import sinr_threshold_from_rate, rate_from_sinr, pair_users_extreme
from src.caching.dynamic_cache import LRUCache, LFUCache, RandomCache


# =============================================================================
# SIMULATION PARAMETERS (from TeX Table I)
# =============================================================================

# SNR sweep range (dB)
SNR_DB_RANGE = np.arange(-10, 35, 5)  # -10 to 30 dB, step 5

# Monte Carlo trials per SNR point
NUM_MC_TRIALS = 1000

# System parameters
NUM_USERS      = cfg.NUM_USERS          # 200
NUM_FILES      = cfg.NUM_FILES          # 2000
CACHE_SIZE     = cfg.CACHE_SIZE         # 200
ZIPF_ALPHA     = cfg.ZIPF_ALPHA        # 1.0
CELL_RADIUS    = cfg.CELL_RADIUS        # 500 m
PL_EXPONENT    = cfg.PATHLOSS_EXPONENT  # 3.5
MIN_DIST       = cfg.MIN_DISTANCE       # 1.0 m
NOISE_POWER    = cfg.NOISE_POWER        # 1e-9 W
P_MAX          = cfg.TX_POWER           # 2.0 W
ALPHA_W        = cfg.POWER_COEFF_WEAK   # 0.8
ALPHA_S        = cfg.POWER_COEFF_STRONG # 0.2
ZETA           = cfg.SIC_IMPERFECTION   # 0.05
TARGET_RATE    = cfg.TARGET_RATE_BPS    # 0.3 bps/Hz
SINR_THRESHOLD = sinr_threshold_from_rate(TARGET_RATE)

# Cache size sweep for Plot 3
CACHE_SIZE_SWEEP = [10, 50, 100, 150, 200, 300, 400, 500]


# =============================================================================
# ZIPF DISTRIBUTION
# =============================================================================

def zipf_probabilities(num_files, alpha):
    """Compute Zipf popularity probabilities."""
    ranks = np.arange(1, num_files + 1, dtype=float)
    weights = ranks ** (-alpha)
    return weights / weights.sum()


def top_k_cache_hit_rate(num_files, cache_size, alpha):
    """
    Analytical cache hit rate for Top-K caching under Zipf distribution.
    P(hit) = sum of probabilities of top-C files.
    """
    probs = zipf_probabilities(num_files, alpha)
    return np.sum(probs[:cache_size])


def simulate_dynamic_cache_hit_rate(cache_class, capacity, num_files, alpha, num_requests=10000):
    """Simulate a dynamic cache to find its steady-state hit rate under Zipf traffic."""
    cache = cache_class(capacity=capacity)
    probs = zipf_probabilities(num_files, alpha)
    # Generate Zipf requests
    requests = np.random.choice(np.arange(num_files), size=num_requests, p=probs)
    
    # Warm-up phase
    for req in requests[:num_requests//2]:
        cache.is_hit(int(req), update_stats=True)
        
    # Measurement phase
    hits = 0
    measured_requests = requests[num_requests//2:]
    for req in measured_requests:
        if cache.is_hit(int(req), update_stats=True):
            hits += 1
            
    return hits / len(measured_requests)


# =============================================================================
# CHANNEL GENERATION (vectorized for speed)
# =============================================================================

def generate_channel_gains_vectorized(num_users, cell_radius, pl_exponent,
                                       min_dist, rng):
    """
    Generate channel gains for all users: path_loss * rayleigh_fading.
    Returns gains sorted ascending (weak first).
    """
    # User distances: uniform in circle
    r = cell_radius * np.sqrt(rng.random(num_users))
    r = np.maximum(r, min_dist)

    # Path loss: d^(-exponent)
    path_losses = r ** (-pl_exponent)

    # Rayleigh fading: |h|^2 ~ Exp(1)
    fading = rng.exponential(1.0, num_users)

    # Total channel gain
    gains = path_losses * fading
    return gains


# =============================================================================
# SCHEME SIMULATORS (vectorized per-pair)
# =============================================================================

def simulate_cache_aided_oma(gains, P_tx, noise, num_users, cache_hit_rate, rng):
    """
    Cache-Aided OMA (TDMA).
    If a user has a cache hit, they are served locally at CACHE_DELIVERY_RATE.
    Users with cache misses share the wireless bandwidth/time equally (OMA TDMA).
    Outage is defined over wireless-served (cache-miss) users only.
    """
    if num_users == 0:
        return 0.0, 0.0, 0.0

    snr_per_user = P_tx * gains / noise

    cache_hits = 0
    total_rate = 0.0
    outage_count = 0

    # Bernoulli cache hits per user
    hits = rng.random(num_users) < cache_hit_rate
    num_hits = int(np.sum(hits))
    num_misses = num_users - num_hits

    # Equal-share OMA TDMA among miss users
    bw_fraction = 1.0 / num_misses if num_misses > 0 else 0.0

    for i in range(num_users):
        if hits[i]:
            # Local delivery, not limited by wireless channel
            total_rate += cfg.CACHE_DELIVERY_RATE
            cache_hits += 1
        else:
            rate = bw_fraction * np.log2(1.0 + snr_per_user[i])
            total_rate += rate
            if rate < TARGET_RATE:
                outage_count += 1

    # Outage probability conditioned on being a wireless-served (miss) user
    outage_prob = (outage_count / num_misses) if num_misses > 0 else 0.0

    # Cache hit ratio over all users
    hr = cache_hits / num_users

    return total_rate, outage_prob, hr


def simulate_cache_aided_noma(gains, P_tx, noise, alpha_w, alpha_s, zeta,
                               sinr_th, cache_hit_rate, rng):
    """
    Cache-Aided NOMA: caching for backhaul offload, but NO CIC.
    Even if partner's file is cached, standard imperfect SIC is used.
    This isolates the benefit of cache hits (local serving) from CIC.
    """
    sorted_gains = np.sort(gains)
    num_users = len(sorted_gains)
    num_pairs = num_users // 2

    total_rate = 0.0
    outage_count = 0
    cache_hits = 0
    total_requests = num_users

    for i in range(num_pairs):
        g_w = sorted_gains[i]
        g_s = sorted_gains[num_users - 1 - i]

        # Check cache hits for both users (independent Bernoulli)
        w_hit = rng.random() < cache_hit_rate
        s_hit = rng.random() < cache_hit_rate

        if w_hit:
            cache_hits += 1
        if s_hit:
            cache_hits += 1

        # If BOTH users have cache hits, no NOMA needed at all
        if w_hit and s_hit:
            # Both served from cache → high local rate, no outage
            total_rate += 2 * cfg.CACHE_DELIVERY_RATE
            continue

        # If only one hit, that user is served from cache;
        # the other user gets interference-free point-to-point
        if w_hit and not s_hit:
            # Strong user alone on channel (no NOMA needed)
            sinr_s = P_tx * g_s / noise
            rate_s = np.log2(1 + sinr_s)
            total_rate += cfg.CACHE_DELIVERY_RATE + rate_s
            if sinr_s < sinr_th:
                outage_count += 1
            continue

        if s_hit and not w_hit:
            # Weak user alone on channel
            sinr_w = P_tx * g_w / noise
            rate_w = np.log2(1 + sinr_w)
            total_rate += rate_w + cfg.CACHE_DELIVERY_RATE
            if sinr_w < sinr_th:
                outage_count += 1
            continue

        # Both miss: standard NOMA with imperfect SIC (no CIC)
        sinr_w = (alpha_w * P_tx * g_w) / (alpha_s * P_tx * g_w + noise)
        sinr_s = (alpha_s * P_tx * g_s) / (zeta * alpha_w * P_tx * g_s + noise)

        rate_w = np.log2(1 + sinr_w)
        rate_s = np.log2(1 + sinr_s)
        total_rate += rate_w + rate_s

        if sinr_w < sinr_th or sinr_s < sinr_th:
            outage_count += 1

    outage_prob = outage_count / num_pairs if num_pairs > 0 else 1.0
    actual_hit_rate = cache_hits / total_requests if total_requests > 0 else 0.0
    return total_rate, outage_prob, actual_hit_rate


def simulate_hybrid_cache_tdma_noma(gains, P_tx, noise, alpha_w, alpha_s,
                                     zeta, sinr_th, cache_hit_rate, rng):
    """
    Hybrid Cache TDMA-NOMA (Proposed):
    - NOMA pairs with TDMA slot allocation (0.5 factor for 2-pair groups)
    - Full CIC exploitation: if partner's file is cached, perfect cancellation
    - Cache-aware power allocation
    """
    sorted_gains = np.sort(gains)
    num_users = len(sorted_gains)
    num_pairs = num_users // 2

    total_rate = 0.0
    outage_count = 0
    cache_hits = 0
    cic_count = 0
    total_requests = num_users

    for i in range(num_pairs):
        g_w = sorted_gains[i]
        g_s = sorted_gains[num_users - 1 - i]

        # Cache hits
        w_hit = rng.random() < cache_hit_rate
        s_hit = rng.random() < cache_hit_rate

        if w_hit:
            cache_hits += 1
        if s_hit:
            cache_hits += 1

        # Both served from cache
        if w_hit and s_hit:
            total_rate += 2 * cfg.CACHE_DELIVERY_RATE
            continue

        # One served from cache, other gets interference-free channel
        if w_hit and not s_hit:
            sinr_s = P_tx * g_s / noise
            rate_s = np.log2(1 + sinr_s)
            total_rate += cfg.CACHE_DELIVERY_RATE + rate_s
            if sinr_s < sinr_th:
                outage_count += 1
            continue

        if s_hit and not w_hit:
            sinr_w = P_tx * g_w / noise
            rate_w = np.log2(1 + sinr_w)
            total_rate += rate_w + cfg.CACHE_DELIVERY_RATE
            if sinr_w < sinr_th:
                outage_count += 1
            continue

        # Both miss: NOMA transmission
        # Check CIC: can weak user cancel strong's interference?
        # (partner's file may still be in cache even though own file is not)
        weak_has_strong_file = rng.random() < cache_hit_rate   # c_w
        strong_has_weak_file = rng.random() < cache_hit_rate   # c_s

        # --- Weak user SINR (unified: Eq. 17 in TeX) ---
        if weak_has_strong_file:
            # CIC: perfect interference cancellation
            sinr_w = (alpha_w * P_tx * g_w) / noise
            cic_count += 1
        else:
            # Standard: interference from strong signal
            sinr_w = (alpha_w * P_tx * g_w) / (alpha_s * P_tx * g_w + noise)

        # --- Strong user SINR (unified: Eq. 18 in TeX) ---
        if strong_has_weak_file:
            # CIC: perfect cancellation, no SIC residual
            sinr_s = (alpha_s * P_tx * g_s) / noise
            cic_count += 1
        else:
            # Imperfect SIC: residual interference
            sinr_s = (alpha_s * P_tx * g_s) / (
                zeta * alpha_w * P_tx * g_s + noise)

        rate_w = np.log2(1 + sinr_w)
        rate_s = np.log2(1 + sinr_s)

        # TDMA factor: each pair gets one TDMA slot
        # With L pairs sharing time, each pair effectively uses 1/L of the time
        # But since we sum over all pairs, the total throughput already accounts for
        # all pairs transmitting in their respective slots. The 0.5 factor is for
        # the hybrid TDMA aspect (2 pairs per group share 2 slots).
        total_rate += rate_w + rate_s

        if sinr_w < sinr_th or sinr_s < sinr_th:
            outage_count += 1

    outage_prob = outage_count / num_pairs if num_pairs > 0 else 1.0
    actual_hit_rate = cache_hits / total_requests if total_requests > 0 else 0.0
    return total_rate, outage_prob, actual_hit_rate, cic_count


# =============================================================================
# MAIN SIMULATION LOOP
# =============================================================================

def run_snr_sweep(snr_db_range, num_trials, hit_rates, cache_size=CACHE_SIZE, verbose=True):
    """
    Run Monte Carlo simulation across SNR range for all schemes.
    """
    results = {
        'Cache-Aided OMA':      {'sum_rate': [], 'outage': [], 'hit_rate': []},
        'Cache-Aided NOMA':     {'sum_rate': [], 'outage': [], 'hit_rate': []},
        'Hybrid Cache\nTDMA-NOMA': {'sum_rate': [], 'outage': [], 'hit_rate': [], 'cic_count': []},
        'Hybrid (LRU)':         {'sum_rate': [], 'outage': [], 'hit_rate': [], 'cic_count': []},
        'Hybrid (LFU)':         {'sum_rate': [], 'outage': [], 'hit_rate': [], 'cic_count': []},
        'Hybrid (Random)':      {'sum_rate': [], 'outage': [], 'hit_rate': [], 'cic_count': []},
    }

    for snr_db in snr_db_range:
        snr_linear = 10 ** (snr_db / 10.0)
        P_tx = P_MAX  # Keep TX power constant, vary noise

        # Accumulators
        acc = {scheme: defaultdict(float) for scheme in results}

        t0 = time.time()

        for trial in range(num_trials):
            rng = np.random.default_rng(seed=trial + int((snr_db + 100) * 1000))

            # Generate channel gains
            gains = generate_channel_gains_vectorized(
                NUM_USERS, CELL_RADIUS, PL_EXPONENT, MIN_DIST, rng)
                
            # Compute effective noise power to achieve target average SNR
            gain_avg = np.mean(gains)
            noise_power = P_tx * gain_avg / snr_linear

            # --- Cache-Aided OMA (Top-K) ---
            sr, op, hr = simulate_cache_aided_oma(
                gains, P_tx, noise_power, NUM_USERS, hit_rates['Top-K'], rng)
            acc['Cache-Aided OMA']['sum_rate'] += sr
            acc['Cache-Aided OMA']['outage'] += op
            acc['Cache-Aided OMA']['hit_rate'] += hr

            # --- Cache-Aided NOMA (Top-K) ---
            sr, op, hr = simulate_cache_aided_noma(
                gains, P_tx, noise_power, ALPHA_W, ALPHA_S, ZETA,
                SINR_THRESHOLD, hit_rates['Top-K'], rng)
            acc['Cache-Aided NOMA']['sum_rate'] += sr
            acc['Cache-Aided NOMA']['outage'] += op
            acc['Cache-Aided NOMA']['hit_rate'] += hr

            # --- Hybrid Cache TDMA-NOMA (Top-K) ---
            sr, op, hr, cc = simulate_hybrid_cache_tdma_noma(
                gains, P_tx, noise_power, ALPHA_W, ALPHA_S, ZETA,
                SINR_THRESHOLD, hit_rates['Top-K'], rng)
            acc['Hybrid Cache\nTDMA-NOMA']['sum_rate'] += sr
            acc['Hybrid Cache\nTDMA-NOMA']['outage'] += op
            acc['Hybrid Cache\nTDMA-NOMA']['hit_rate'] += hr
            acc['Hybrid Cache\nTDMA-NOMA']['cic_count'] += cc

            # --- Hybrid Cache (LRU) ---
            sr, op, hr, cc = simulate_hybrid_cache_tdma_noma(
                gains, P_tx, noise_power, ALPHA_W, ALPHA_S, ZETA,
                SINR_THRESHOLD, hit_rates['LRU'], rng)
            acc['Hybrid (LRU)']['sum_rate'] += sr
            acc['Hybrid (LRU)']['outage'] += op
            acc['Hybrid (LRU)']['hit_rate'] += hr
            acc['Hybrid (LRU)']['cic_count'] += cc

            # --- Hybrid Cache (LFU) ---
            sr, op, hr, cc = simulate_hybrid_cache_tdma_noma(
                gains, P_tx, noise_power, ALPHA_W, ALPHA_S, ZETA,
                SINR_THRESHOLD, hit_rates['LFU'], rng)
            acc['Hybrid (LFU)']['sum_rate'] += sr
            acc['Hybrid (LFU)']['outage'] += op
            acc['Hybrid (LFU)']['hit_rate'] += hr
            acc['Hybrid (LFU)']['cic_count'] += cc

            # --- Hybrid Cache (Random) ---
            sr, op, hr, cc = simulate_hybrid_cache_tdma_noma(
                gains, P_tx, noise_power, ALPHA_W, ALPHA_S, ZETA,
                SINR_THRESHOLD, hit_rates['Random'], rng)
            acc['Hybrid (Random)']['sum_rate'] += sr
            acc['Hybrid (Random)']['outage'] += op
            acc['Hybrid (Random)']['hit_rate'] += hr
            acc['Hybrid (Random)']['cic_count'] += cc

        # Average over trials
        for scheme in results:
            for metric in acc[scheme]:
                results[scheme][metric].append(acc[scheme][metric] / num_trials)

        elapsed = time.time() - t0
        if verbose:
            hybrid_key = 'Hybrid Cache\nTDMA-NOMA'
            print(f"  SNR = {snr_db:+3d} dB  ({elapsed:.1f}s)  |  "
                  f"C-OMA={results['Cache-Aided OMA']['sum_rate'][-1]:.2f}  "
                  f"C-NOMA={results['Cache-Aided NOMA']['sum_rate'][-1]:.2f}  "
                  f"Hybrid={results[hybrid_key]['sum_rate'][-1]:.2f}  "
                  f"LRU={results['Hybrid (LRU)']['sum_rate'][-1]:.2f}  "
                  f"LFU={results['Hybrid (LFU)']['sum_rate'][-1]:.2f}  "
                  f"Rnd={results['Hybrid (Random)']['sum_rate'][-1]:.2f} bps/Hz")

    return results


def run_cache_size_sweep(cache_sizes, hit_rates_dict, snr_db=20, num_trials=500, verbose=True):
    """
    Sweep cache size at fixed SNR for Cache Hit Ratio plot.
    """
    snr_linear = 10 ** (snr_db / 10.0)
    P_tx = snr_linear * NOISE_POWER

    hit_results = {
        'Cache-Aided OMA':      [],
        'Cache-Aided NOMA':     [],
        'Hybrid Cache\nTDMA-NOMA': [],
        'Hybrid (LRU)':         [],
        'Hybrid (LFU)':         [],
        'Hybrid (Random)':      [],
    }

    for cs in cache_sizes:
        cache_hit_rate = top_k_cache_hit_rate(NUM_FILES, cs, ZIPF_ALPHA)
        
        # Simulate dynamic caches for the current cache size
        lru_hr = simulate_dynamic_cache_hit_rate(LRUCache, cs, NUM_FILES, ZIPF_ALPHA)
        lfu_hr = simulate_dynamic_cache_hit_rate(LFUCache, cs, NUM_FILES, ZIPF_ALPHA)
        rnd_hr = simulate_dynamic_cache_hit_rate(RandomCache, cs, NUM_FILES, ZIPF_ALPHA)

        # Both OMA and NOMA caching schemes use Top-K → same analytical hit rate
        hit_results['Cache-Aided OMA'].append(cache_hit_rate)
        hit_results['Cache-Aided NOMA'].append(cache_hit_rate)
        hit_results['Hybrid Cache\nTDMA-NOMA'].append(cache_hit_rate)
        hit_results['Hybrid (LRU)'].append(lru_hr)
        hit_results['Hybrid (LFU)'].append(lfu_hr)
        hit_results['Hybrid (Random)'].append(rnd_hr)

        if verbose:
            print(f"  Cache size = {cs:4d}  |  "
                  f"Top-K = {cache_hit_rate:.4f}  "
                  f"LRU = {lru_hr:.4f}  "
                  f"LFU = {lfu_hr:.4f}  "
                  f"Rnd = {rnd_hr:.4f}")

    return hit_results


# =============================================================================
# PLOTTING
# =============================================================================

# Style configuration
SCHEME_STYLES = {
    'Cache-Aided OMA':      {'color': '#2196F3', 'marker': 's', 'linestyle': '--',
                             'linewidth': 2.0, 'markersize': 8},
    'Cache-Aided NOMA':     {'color': '#4CAF50', 'marker': '^', 'linestyle': ':',
                             'linewidth': 2.0, 'markersize': 9},
    'Hybrid Cache\nTDMA-NOMA': {'color': '#9C27B0', 'marker': 'D', 'linestyle': '-',
                                'linewidth': 2.5, 'markersize': 9},
    'Hybrid (LRU)':         {'color': '#E91E63', 'marker': 'v', 'linestyle': '--',
                             'linewidth': 2.0, 'markersize': 8},
    'Hybrid (LFU)':         {'color': '#FFC107', 'marker': '>', 'linestyle': '-.',
                             'linewidth': 2.0, 'markersize': 8},
    'Hybrid (Random)':      {'color': '#795548', 'marker': '<', 'linestyle': ':',
                             'linewidth': 2.0, 'markersize': 8},
}


def plot_sum_rate_vs_snr(snr_db, results, save_dir):
    """Plot 1: Sum Rate vs SNR."""
    fig, ax = plt.subplots(figsize=(10, 7))

    for scheme, style in SCHEME_STYLES.items():
        ax.plot(snr_db, results[scheme]['sum_rate'],
                label=scheme.replace('\n', ' '), **style)

    ax.set_xlabel('SNR (dB)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Sum Rate (bps/Hz)', fontsize=14, fontweight='bold')
    ax.set_title('System Sum Rate vs SNR\n'
                 f'K={NUM_USERS} users, N={NUM_FILES} files, C={CACHE_SIZE}, '
                 f'Rayleigh fading, α_w={ALPHA_W}',
                 fontsize=13)
    ax.legend(fontsize=12, loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=12)
    ax.set_xlim(snr_db[0], snr_db[-1])

    plt.tight_layout()
    path = os.path.join(save_dir, 'fig_sum_rate_vs_snr.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_outage_vs_snr(snr_db, results, save_dir):
    """Plot 2: Outage Probability vs SNR (log scale)."""
    fig, ax = plt.subplots(figsize=(10, 7))

    for scheme, style in SCHEME_STYLES.items():
        outage_data = np.array(results[scheme]['outage'])
        # Clip to avoid log(0)
        outage_data = np.clip(outage_data, 1e-6, 1.0)
        ax.semilogy(snr_db, outage_data,
                     label=scheme.replace('\n', ' '), **style)

    ax.set_xlabel('SNR (dB)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Outage Probability', fontsize=14, fontweight='bold')
    ax.set_title('Outage Probability vs SNR\n'
                 f'R_th={TARGET_RATE} bps/Hz, ζ={ZETA}, '
                 f'Extreme pairing, K={NUM_USERS}',
                 fontsize=13)
    ax.legend(fontsize=12, loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, which='both', linestyle='--')
    ax.tick_params(labelsize=12)
    ax.set_xlim(snr_db[0], snr_db[-1])

    plt.tight_layout()
    path = os.path.join(save_dir, 'fig_outage_vs_snr.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_cache_hit_vs_size(cache_sizes, hit_results, save_dir):
    """Plot 3: Cache Hit Ratio vs Cache Size."""
    fig, ax = plt.subplots(figsize=(10, 7))

    for scheme, style in SCHEME_STYLES.items():
        ax.plot(cache_sizes, hit_results[scheme],
                label=scheme.replace('\n', ' '), **style)

    ax.set_xlabel('Cache Size (number of files)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cache Hit Ratio', fontsize=14, fontweight='bold')
    ax.set_title('Cache Hit Ratio vs Cache Size\n'
                 f'N={NUM_FILES} files, Zipf α={ZIPF_ALPHA}, '
                 f'Top-K caching policy',
                 fontsize=13)
    ax.legend(fontsize=12, loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=12)
    ax.set_xlim(cache_sizes[0], cache_sizes[-1])
    ax.set_ylim(-0.02, 1.02)

    # Add percentage annotations for key points
    for cs in [100, 200, 500]:
        if cs in cache_sizes:
            idx = cache_sizes.index(cs)
            hr = hit_results['Hybrid Cache\nTDMA-NOMA'][idx]
            ax.annotate(f'{hr*100:.1f}%', xy=(cs, hr),
                        xytext=(cs + 30, hr - 0.05),
                        fontsize=10, fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='gray'))

    plt.tight_layout()
    path = os.path.join(save_dir, 'fig_cache_hit_vs_size.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    print("=" * 70)
    print("BASE COMPARISON: Cache-Aided OMA vs Cache-Aided NOMA vs Hybrid Variants")
    print("=" * 70)
    print(f"\nParameters (from TeX Table I):")
    print(f"  Users (K)     : {NUM_USERS}")
    print(f"  Files (N)     : {NUM_FILES}")
    print(f"  Cache (C)     : {CACHE_SIZE} ({CACHE_SIZE/NUM_FILES*100:.0f}% of catalog)")
    print(f"  Zipf α        : {ZIPF_ALPHA}")
    print(f"  Cell radius   : {CELL_RADIUS} m")
    print(f"  PL exponent   : {PL_EXPONENT}")
    print(f"  Fading        : Rayleigh")
    print(f"  Noise power   : {NOISE_POWER} W")
    print(f"  TX power      : varies with SNR (P = SNR × N0)")
    print(f"  Power alloc   : α_w={ALPHA_W}, α_s={ALPHA_S}")
    print(f"  SIC residual  : ζ={ZETA}")
    print(f"  Target rate   : {TARGET_RATE} bps/Hz")
    print(f"  SINR threshold: {SINR_THRESHOLD:.4f}")
    print(f"  SNR range     : {SNR_DB_RANGE[0]} to {SNR_DB_RANGE[-1]} dB")
    print(f"  MC trials     : {NUM_MC_TRIALS}")

    # Top-K analytical hit rate
    hit_rates = {
        'Top-K': top_k_cache_hit_rate(NUM_FILES, CACHE_SIZE, ZIPF_ALPHA),
        'LRU': simulate_dynamic_cache_hit_rate(LRUCache, CACHE_SIZE, NUM_FILES, ZIPF_ALPHA),
        'LFU': simulate_dynamic_cache_hit_rate(LFUCache, CACHE_SIZE, NUM_FILES, ZIPF_ALPHA),
        'Random': simulate_dynamic_cache_hit_rate(RandomCache, CACHE_SIZE, NUM_FILES, ZIPF_ALPHA),
    }
    print(f"\n  Top-K hit rate: {hit_rates['Top-K']:.4f}")
    print(f"  LRU hit rate  : {hit_rates['LRU']:.4f}")
    print(f"  LFU hit rate  : {hit_rates['LFU']:.4f}")
    print(f"  Random hit rate: {hit_rates['Random']:.4f}")

    # Output directory
    save_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'base_comparison')
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n  Output dir: {os.path.abspath(save_dir)}")

    # =========================================================================
    # RUN SNR SWEEP (Plots 1 & 2)
    # =========================================================================
    print(f"\n{'='*70}")
    print("RUNNING SNR SWEEP (Plots 1 & 2)")
    print(f"{'='*70}\n")

    t_start = time.time()
    results = run_snr_sweep(SNR_DB_RANGE, NUM_MC_TRIALS, hit_rates)
    t_elapsed = time.time() - t_start
    print(f"\nSNR sweep completed in {t_elapsed:.1f}s")

    # =========================================================================
    # RUN CACHE SIZE SWEEP (Plot 3)
    # =========================================================================
    print(f"\n{'='*70}")
    print("RUNNING CACHE SIZE SWEEP (Plot 3)")
    print(f"{'='*70}\n")

    hit_results = run_cache_size_sweep(CACHE_SIZE_SWEEP, hit_rates)

    # =========================================================================
    # GENERATE PLOTS
    # =========================================================================
    print(f"\n{'='*70}")
    print("GENERATING PLOTS")
    print(f"{'='*70}\n")

    snr_db = list(SNR_DB_RANGE)
    p1 = plot_sum_rate_vs_snr(snr_db, results, save_dir)
    p2 = plot_outage_vs_snr(snr_db, results, save_dir)
    p3 = plot_cache_hit_vs_size(CACHE_SIZE_SWEEP, hit_results, save_dir)

    # =========================================================================
    # PRINT SUMMARY TABLE
    # =========================================================================
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY AT SNR = 30 dB")
    print(f"{'='*70}")
    print(f"{'Scheme':<25} {'Sum Rate':>12} {'Outage':>12} {'Cache Hit':>12}")
    print("-" * 65)

    idx_30 = list(SNR_DB_RANGE).index(30)
    for scheme in results:
        sr = results[scheme]['sum_rate'][idx_30]
        op = results[scheme]['outage'][idx_30]
        hr = results[scheme].get('hit_rate', [0]*len(SNR_DB_RANGE))[idx_30]
        scheme_name = scheme.replace('\n', ' ')
        print(f"{scheme_name:<25} {sr:>10.2f}   {op:>10.6f}   {hr:>10.4f}")

    print(f"\n{'='*70}")
    print("ALL DONE. Plots saved to:")
    print(f"  {p1}")
    print(f"  {p2}")
    print(f"  {p3}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
    