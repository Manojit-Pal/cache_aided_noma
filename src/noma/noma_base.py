# src/noma/noma_base.py
"""
NOMA Base Module - Core NOMA Transmission and Pairing Logic

This module implements:
- NOMA transmission simulation for user pairs
- User pairing strategies (extreme, random, sequential)
- Cache-Aided Interference Cancellation (CIC)
- BER and outage probability calculations
- Performance metrics (sum rate, spectral efficiency, fairness)
- Multi-user NOMA scheduling

✅ FIX #1 APPLIED: CIC tracking now uses list to properly handle both users cached
✅ FIX #2 APPLIED: Cache-aware power allocation integrated into simulation
✅ BUG-NOMA-3 FIXED: Fairness divide-by-zero guarded (Jain's index denominator)

Author: Cache-Aided NOMA Team
Date: December 2025 | Revised: March 2026
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from .sic import sinr_weak_user, sinr_strong_decode_weak, sinr_strong_after_sic


# ============================================================================
# SINR AND RATE CONVERSIONS
# ============================================================================

def sinr_threshold_from_rate(rate_bps: float) -> float:
    """
    Convert target data rate to required SINR threshold.
    Based on Shannon: C = log2(1 + SINR) => SINR = 2^C - 1
    """
    return 2 ** rate_bps - 1


def rate_from_sinr(sinr: float) -> float:
    """Convert SINR to achievable data rate: C = log2(1 + SINR)"""
    return np.log2(1 + sinr)


def sinr_to_ber_bpsk(sinr: float) -> float:
    """BER for BPSK: 0.5 * erfc(sqrt(SINR))"""
    from scipy.special import erfc
    return 0.5 * erfc(np.sqrt(sinr))


def sinr_to_ber_qpsk(sinr: float) -> float:
    """BER for QPSK: 0.5 * erfc(sqrt(SINR/2))"""
    from scipy.special import erfc
    return 0.5 * erfc(np.sqrt(sinr / 2.0))


# ============================================================================
# BASIC NOMA PAIR SIMULATION (✅ FIX #1 & #2 APPLIED)
# ============================================================================

def simulate_noma_pair(gain_weak: float, gain_strong: float, cfg,
                       p_w: Optional[float] = None, p_s: Optional[float] = None,
                       weak_cached: bool = False, strong_cached: bool = False,
                       optimize_power: bool = True
                       ) -> Tuple[bool, bool, Dict]:
    """
    Simulate NOMA transmission for a two-user pair with optional cache-aided SIC.

    ✅ FIX #1: CIC tracking uses 'cic_users' list (handles both users cached)
    ✅ FIX #2: Integrates cache-aware power allocation when enabled
    ✅ BUG-NOMA-3: Fairness denominator guarded against zero division
    """
    P  = cfg.TX_POWER
    N0 = cfg.NOISE_POWER

    # -------------------------------------------------------------------------
    # POWER ALLOCATION
    # -------------------------------------------------------------------------
    if p_w is None or p_s is None:
        if (optimize_power
                and hasattr(cfg, 'POWER_ALLOC_METHOD')
                and cfg.POWER_ALLOC_METHOD == 'cache_aware'):
            try:
                from .power_allocation import allocate_power_cache_aware
                p_w, p_s, power_feasible, power_info = allocate_power_cache_aware(
                    gain_weak, gain_strong, cfg, weak_cached, strong_cached
                )
                if not power_feasible:
                    p_w = cfg.POWER_COEFF_WEAK
                    p_s = cfg.POWER_COEFF_STRONG
                    power_info['method'] = 'cache_aware_failed_fallback'
            except (ImportError, AttributeError) as e:
                p_w = cfg.POWER_COEFF_WEAK
                p_s = cfg.POWER_COEFF_STRONG
                power_info = {'method': 'config_default', 'reason': str(e)}
        else:
            p_w = cfg.POWER_COEFF_WEAK
            p_s = cfg.POWER_COEFF_STRONG
            power_info = {'method': 'config_default'}
    else:
        power_info = {'method': 'provided', 'p_w': p_w, 'p_s': p_s}

    zeta    = cfg.SIC_IMPERFECTION
    sinr_th = sinr_threshold_from_rate(cfg.TARGET_RATE_BPS)

    info = {
        'p_w': p_w, 'p_s': p_s,
        'gain_weak': gain_weak, 'gain_strong': gain_strong,
        'weak_cached': weak_cached, 'strong_cached': strong_cached,
        'cic_applied': False,
        'cic_users': [],
        'power_allocation': power_info,
    }

    # -------------------------------------------------------------------------
    # WEAK USER DECODING (with optional CIC)
    # -------------------------------------------------------------------------
    if weak_cached:
        sinr_w = (P * p_w * gain_weak) / N0   # Perfect CIC: no interference
        info['cic_applied'] = True
        info['cic_users'].append('weak')
    else:
        sinr_w = sinr_weak_user(P, p_w, gain_weak, p_s, N0)

    weak_success        = sinr_w >= sinr_th
    achievable_rate_weak = rate_from_sinr(sinr_w)
    ber_weak            = sinr_to_ber_qpsk(sinr_w)

    info.update({
        'sinr_w': sinr_w,
        'achievable_rate_w': achievable_rate_weak,
        'ber_w': ber_weak,
        'weak_success': weak_success,
    })

    # -------------------------------------------------------------------------
    # STRONG USER DECODES WEAK SIGNAL (needed for SIC)
    # -------------------------------------------------------------------------
    sinr_s_decode_w = sinr_strong_decode_weak(P, p_w, gain_strong, p_s, N0)
    can_decode_weak = sinr_s_decode_w >= sinr_th

    info['sinr_s_decode_w'] = sinr_s_decode_w
    info['can_decode_weak'] = can_decode_weak

    # -------------------------------------------------------------------------
    # STRONG USER AFTER SIC
    # -------------------------------------------------------------------------
    if strong_cached:
        residual = 0.0   # Perfect cancellation via cache
        info['cic_applied'] = True
        info['cic_users'].append('strong')
    else:
        if can_decode_weak:
            residual = zeta * (P * p_w * gain_strong)   # Imperfect SIC
        else:
            residual = P * p_w * gain_strong              # Failed SIC

    sinr_s_after        = sinr_strong_after_sic(P, p_s, gain_strong, N0, residual)
    strong_success       = sinr_s_after >= sinr_th
    achievable_rate_strong = rate_from_sinr(sinr_s_after)
    ber_strong           = sinr_to_ber_qpsk(sinr_s_after)

    info.update({
        'residual_interference': residual,
        'sinr_s_after': sinr_s_after,
        'achievable_rate_s': achievable_rate_strong,
        'ber_s': ber_strong,
        'strong_success': strong_success,
    })

    # -------------------------------------------------------------------------
    # PERFORMANCE METRICS
    # -------------------------------------------------------------------------
    info['pair_success'] = weak_success and strong_success
    info['sum_rate']     = achievable_rate_weak + achievable_rate_strong
    info['outage']       = not info['pair_success']

    # Jain's fairness index for 2 users
    # BUG-NOMA-3 FIX: guard denominator to avoid ZeroDivisionError when
    # both rates are 0 (extreme low-SNR Rayleigh pair).
    rates = [achievable_rate_weak, achievable_rate_strong]
    denom = max(2.0 * sum(r ** 2 for r in rates), 1e-12)
    info['fairness'] = (sum(rates) ** 2) / denom

    return weak_success, strong_success, info


# ============================================================================
# USER PAIRING STRATEGIES
# ============================================================================

def pair_users_extreme(channel_gains: np.ndarray,
                       user_indices: Optional[np.ndarray] = None
                       ) -> List[Tuple[int, int]]:
    """
    Extreme pairing: pair users with most different channel conditions.
    Sort by gain (ascending), pair weakest with strongest.
    """
    num_users = len(channel_gains)
    if user_indices is None:
        user_indices = np.arange(num_users)
    sorted_indices  = np.argsort(channel_gains)
    sorted_user_ids = user_indices[sorted_indices]
    num_pairs = num_users // 2
    return [
        (sorted_user_ids[i], sorted_user_ids[-(i + 1)])
        for i in range(num_pairs)
    ]


def pair_users_random(channel_gains: np.ndarray,
                      user_indices: Optional[np.ndarray] = None,
                      seed: Optional[int] = None
                      ) -> List[Tuple[int, int]]:
    """Random pairing: shuffle then pair consecutive users."""
    if seed is not None:
        np.random.seed(seed)
    num_users = len(channel_gains)
    if user_indices is None:
        user_indices = np.arange(num_users)
    shuffled  = np.random.permutation(user_indices)
    num_pairs = num_users // 2
    pairs = []
    for i in range(num_pairs):
        idx1, idx2 = shuffled[2 * i], shuffled[2 * i + 1]
        if channel_gains[idx1] > channel_gains[idx2]:
            pairs.append((idx2, idx1))
        else:
            pairs.append((idx1, idx2))
    return pairs


def pair_users_sequential(channel_gains: np.ndarray,
                          user_indices: Optional[np.ndarray] = None
                          ) -> List[Tuple[int, int]]:
    """Sequential pairing: sort by gain, pair consecutive."""
    num_users = len(channel_gains)
    if user_indices is None:
        user_indices = np.arange(num_users)
    sorted_indices  = np.argsort(channel_gains)
    sorted_user_ids = user_indices[sorted_indices]
    num_pairs = num_users // 2
    return [
        (sorted_user_ids[2 * i], sorted_user_ids[2 * i + 1])
        for i in range(num_pairs)
    ]


def pair_users(users: List[int], channel_gains: np.ndarray,
               method: str = 'extreme'
               ) -> Tuple[List[Tuple[int, int]], Optional[int]]:
    """
    Universal user pairing with multiple strategies.

    NOTE: channel_gains must be indexed by user ID (users must be
    contiguous 0..N-1 indices). In simulate_noma_system() this is
    always satisfied. If called with non-contiguous user IDs, the
    caller must pass a gain array indexed accordingly.
    """
    num_users    = len(users)
    leftover_user = None
    if num_users % 2 == 1:
        leftover_user = users[-1]
        users_to_pair = users[:-1]
    else:
        users_to_pair = users

    user_gains = channel_gains[users_to_pair]

    if method == 'extreme':
        pairs = pair_users_extreme(user_gains, user_indices=np.array(users_to_pair))
    elif method == 'random':
        pairs = pair_users_random(user_gains, user_indices=np.array(users_to_pair))
    elif method == 'sequential':
        pairs = pair_users_sequential(user_gains, user_indices=np.array(users_to_pair))
    else:
        raise ValueError(
            f"Unknown pairing method: '{method}'. Use 'extreme', 'random', or 'sequential'."
        )

    return pairs, leftover_user


# ============================================================================
# MULTI-USER NOMA SIMULATION
# ============================================================================

def simulate_noma_system(channel_gains: np.ndarray, cfg,
                         pairing_method: str = 'extreme',
                         cache_status: Optional[Dict[int, bool]] = None,
                         requested_files: Optional[Dict[int, int]] = None,
                         optimize_power: bool = True
                         ) -> Dict:
    """
    Simulate complete NOMA system for all user pairs.
    """
    num_users = len(channel_gains)
    all_users = list(range(num_users))

    pairs, _ = pair_users(all_users, channel_gains, method=pairing_method)

    if cache_status is None:
        cache_status = {i: False for i in range(num_users)}
    if requested_files is None:
        requested_files = {i: -1 for i in range(num_users)}

    pair_results = []
    for weak_idx, strong_idx in pairs:
        weak_ok, strong_ok, info = simulate_noma_pair(
            channel_gains[weak_idx], channel_gains[strong_idx], cfg,
            weak_cached=cache_status.get(weak_idx, False),
            strong_cached=cache_status.get(strong_idx, False),
            optimize_power=optimize_power
        )
        info['weak_idx']   = weak_idx
        info['strong_idx'] = strong_idx
        pair_results.append(info)

    num_pairs = len(pairs)

    weak_success_count  = sum(1 for r in pair_results if r['weak_success'])
    strong_success_count = sum(1 for r in pair_results if r['strong_success'])
    pair_success_count  = sum(1 for r in pair_results if r['pair_success'])

    total_sum_rate = sum(r['sum_rate'] for r in pair_results)
    outage_count   = sum(1 for r in pair_results if r['outage'])

    average_ber_weak   = np.mean([r['ber_w'] for r in pair_results]) if pair_results else 0
    average_ber_strong = np.mean([r['ber_s'] for r in pair_results]) if pair_results else 0
    average_fairness   = np.mean([r['fairness'] for r in pair_results]) if pair_results else 1.0

    cache_hit_count   = sum(1 for i in range(num_users) if cache_status.get(i, False))
    cic_applied_count = sum(1 for r in pair_results if r.get('cic_applied', False))

    weak_cic_count   = sum(1 for r in pair_results if 'weak'   in r.get('cic_users', []))
    strong_cic_count = sum(1 for r in pair_results if 'strong' in r.get('cic_users', []))
    both_cic_count   = sum(1 for r in pair_results if len(r.get('cic_users', [])) == 2)

    power_methods      = [r.get('power_allocation', {}).get('method', 'unknown') for r in pair_results]
    cache_aware_count  = sum(1 for m in power_methods if 'cache_aware' in m and 'failed' not in m)

    def _safe_div(num, den): return num / den if den > 0 else 0

    system_metrics = {
        'num_users':              num_users,
        'num_pairs':              num_pairs,
        'weak_success_rate':      _safe_div(weak_success_count,  num_pairs),
        'strong_success_rate':    _safe_div(strong_success_count, num_pairs),
        'overall_success_rate':   _safe_div(pair_success_count,  num_pairs),
        'average_sum_rate':       _safe_div(total_sum_rate,       num_pairs),
        'system_throughput':      total_sum_rate,
        'outage_probability':     _safe_div(outage_count,         num_pairs),
        'average_ber_weak':       average_ber_weak,
        'average_ber_strong':     average_ber_strong,
        'average_ber':            (average_ber_weak + average_ber_strong) / 2,
        'average_fairness':       average_fairness,
        'cache_hit_rate':         _safe_div(cache_hit_count,   num_users),
        'cic_benefit_rate':       _safe_div(cic_applied_count, num_pairs),
        'weak_cic_count':         weak_cic_count,
        'strong_cic_count':       strong_cic_count,
        'both_cic_count':         both_cic_count,
        'pairing_method':         pairing_method,
        'power_optimization_enabled': optimize_power,
        'cache_aware_power_count':    cache_aware_count,
    }

    return {
        'pairs':          pairs,
        'pair_results':   pair_results,
        'system_metrics': system_metrics,
    }


# ============================================================================
# OUTAGE / BER AGGREGATION
# ============================================================================

def compute_outage_probability(results_list: List[Dict]) -> float:
    total, outages = 0, 0
    for result in results_list:
        for pr in result['pair_results']:
            total   += 1
            outages += int(pr['outage'])
    return outages / total if total > 0 else 0


def compute_average_ber(results_list: List[Dict]) -> float:
    ber_values = []
    for result in results_list:
        for pr in result['pair_results']:
            ber_values.extend([pr['ber_w'], pr['ber_s']])
    return np.mean(ber_values) if ber_values else 0
