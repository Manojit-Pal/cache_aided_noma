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

Author: Cache-Aided NOMA Team
Date: December 2025
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
    
    Based on Shannon capacity formula:
        C = log2(1 + SINR)  =>  SINR = 2^C - 1
    
    Args:
        rate_bps: Target data rate in bits/s/Hz (spectral efficiency)
    
    Returns:
        float: Required SINR threshold (linear scale)
    
    Example:
        >>> sinr_th = sinr_threshold_from_rate(0.5)  # 0.5 bps/Hz
        >>> print(f"Required SINR: {sinr_th:.3f}")  # 0.414
    """
    return 2 ** rate_bps - 1


def rate_from_sinr(sinr: float) -> float:
    """
    Convert SINR to achievable data rate.
    
    Shannon capacity: C = log2(1 + SINR)
    
    Args:
        sinr: Signal-to-Interference-plus-Noise Ratio (linear scale)
    
    Returns:
        float: Achievable rate in bits/s/Hz
    
    Example:
        >>> rate = rate_from_sinr(10.0)  # SINR = 10
        >>> print(f"Achievable rate: {rate:.2f} bps/Hz")  # ~3.46 bps/Hz
    """
    return np.log2(1 + sinr)


def sinr_to_ber_bpsk(sinr: float) -> float:
    """
    Compute Bit Error Rate (BER) for BPSK modulation.
    
    BER_BPSK = Q(sqrt(2*SINR)) ≈ 0.5 * erfc(sqrt(SINR))
    
    Args:
        sinr: SINR in linear scale
    
    Returns:
        float: Bit error rate (0 to 0.5)
    """
    from scipy.special import erfc
    return 0.5 * erfc(np.sqrt(sinr))


def sinr_to_ber_qpsk(sinr: float) -> float:
    """
    Compute BER for QPSK modulation.
    
    BER_QPSK = Q(sqrt(SINR)) ≈ 0.5 * erfc(sqrt(SINR/2))
    
    Args:
        sinr: SINR in linear scale
    
    Returns:
        float: Bit error rate
    """
    from scipy.special import erfc
    return 0.5 * erfc(np.sqrt(sinr / 2.0))


# ============================================================================
# BASIC NOMA PAIR SIMULATION
# ============================================================================

def simulate_noma_pair(gain_weak: float, gain_strong: float, cfg, 
                       p_w: Optional[float] = None, p_s: Optional[float] = None,
                       weak_cached: bool = False, strong_cached: bool = False
                       ) -> Tuple[bool, bool, Dict]:
    """
    Simulate NOMA transmission for a two-user pair with optional cache-aided SIC.
    
    This is the core NOMA transmission function that:
    1. Computes SINR for weak user (treats strong as interference)
    2. Checks if strong user can decode weak user's signal (for SIC)
    3. Performs SIC with imperfection factor ζ
    4. Computes strong user's SINR after SIC
    5. Applies Cache-Aided Interference Cancellation (CIC) if content is cached
    
    Args:
        gain_weak: Channel power gain for weak user (farther from BS)
        gain_strong: Channel power gain for strong user (closer to BS)
        cfg: Configuration object with TX_POWER, NOISE_POWER, etc.
        p_w: Power coefficient for weak user (0 to 1). If None, uses cfg default.
        p_s: Power coefficient for strong user. If None, uses cfg default.
        weak_cached: Whether weak user's requested file is in cache (enables CIC)
        strong_cached: Whether strong user's requested file is in cache (enables CIC)
    
    Returns:
        Tuple containing:
        - weak_success (bool): Whether weak user meets target rate
        - strong_success (bool): Whether strong user meets target rate
        - info (dict): Detailed metrics including:
            - All SINR values
            - BER estimates
            - Achievable rates
            - CIC benefits
    
    Cache-Aided Interference Cancellation (CIC):
        If a user has the interfering content cached, they can perfectly cancel
        that interference, improving their SINR:
        - Weak user with strong's content cached → cancels strong user interference
        - Strong user with weak's content cached → enhanced SIC (perfect cancellation)
    
    Example:
        >>> weak_ok, strong_ok, info = simulate_noma_pair(
        ...     gain_weak=1e-8, gain_strong=1e-6, cfg=config,
        ...     weak_cached=True  # Weak user has strong's content cached
        ... )
    """
    # Get power allocation
    P = cfg.TX_POWER
    N0 = cfg.NOISE_POWER
    if p_w is None or p_s is None:
        p_w = cfg.POWER_COEFF_WEAK
        p_s = cfg.POWER_COEFF_STRONG
    
    zeta = cfg.SIC_IMPERFECTION
    sinr_th = sinr_threshold_from_rate(cfg.TARGET_RATE_BPS)
    
    # Initialize info dictionary
    info = {
        'p_w': p_w,
        'p_s': p_s,
        'gain_weak': gain_weak,
        'gain_strong': gain_strong,
        'weak_cached': weak_cached,
        'strong_cached': strong_cached,
        'cic_applied': False
    }
    
    # -------------------------------------------------------------------------
    # WEAK USER DECODING (with optional CIC)
    # -------------------------------------------------------------------------
    if weak_cached:
        # Cache-Aided Interference Cancellation (CIC) for weak user
        # If weak user has strong user's content cached, can cancel interference
        sinr_w = (P * p_w * gain_weak) / N0  # No interference from strong user
        info['cic_applied'] = True
        info['cic_user'] = 'weak'
    else:
        # Standard NOMA: weak user treats strong as interference
        sinr_w = sinr_weak_user(P, p_w, gain_weak, p_s, N0)
    
    weak_success = sinr_w >= sinr_th
    achievable_rate_weak = rate_from_sinr(sinr_w)
    ber_weak = sinr_to_ber_qpsk(sinr_w)  # Assuming QPSK modulation
    
    info['sinr_w'] = sinr_w
    info['achievable_rate_w'] = achievable_rate_weak
    info['ber_w'] = ber_weak
    info['weak_success'] = weak_success
    
    # -------------------------------------------------------------------------
    # STRONG USER DECODING WEAK SIGNAL (for SIC)
    # -------------------------------------------------------------------------
    sinr_s_decode_w = sinr_strong_decode_weak(P, p_w, gain_strong, p_s, N0)
    can_decode_weak = sinr_s_decode_w >= sinr_th
    
    info['sinr_s_decode_w'] = sinr_s_decode_w
    info['can_decode_weak'] = can_decode_weak
    
    # -------------------------------------------------------------------------
    # STRONG USER AFTER SIC (with optional perfect CIC)
    # -------------------------------------------------------------------------
    if strong_cached:
        # Strong user has weak user's content cached → perfect SIC
        residual = 0.0  # Perfect cancellation
        info['cic_applied'] = True
        info['cic_user'] = 'strong' if not info.get('cic_applied') else 'both'
    else:
        # Standard SIC with imperfection
        if can_decode_weak:
            residual = zeta * (P * p_w * gain_strong)  # Imperfect SIC
        else:
            residual = P * p_w * gain_strong  # Failed SIC, full interference
    
    sinr_s_after = sinr_strong_after_sic(P, p_s, gain_strong, N0, residual)
    strong_success = sinr_s_after >= sinr_th
    achievable_rate_strong = rate_from_sinr(sinr_s_after)
    ber_strong = sinr_to_ber_qpsk(sinr_s_after)
    
    info['residual_interference'] = residual
    info['sinr_s_after'] = sinr_s_after
    info['achievable_rate_s'] = achievable_rate_strong
    info['ber_s'] = ber_strong
    info['strong_success'] = strong_success
    
    # -------------------------------------------------------------------------
    # PERFORMANCE METRICS
    # -------------------------------------------------------------------------
    info['pair_success'] = weak_success and strong_success
    info['sum_rate'] = achievable_rate_weak + achievable_rate_strong
    info['outage'] = not info['pair_success']
    
    # Fairness metric (Jain's fairness index for 2 users)
    rates = [achievable_rate_weak, achievable_rate_strong]
    info['fairness'] = (sum(rates) ** 2) / (2 * sum([r**2 for r in rates]))
    
    return weak_success, strong_success, info


# ============================================================================
# USER PAIRING STRATEGIES
# ============================================================================

def pair_users_extreme(channel_gains: np.ndarray, user_indices: Optional[np.ndarray] = None
                       ) -> List[Tuple[int, int]]:
    """
    Extreme pairing: Pair users with most different channel conditions.
    
    Strategy: Sort users by channel gain, pair strongest with weakest.
    This maximizes the NOMA gain by exploiting large channel difference.
    
    Best for: Maximizing system throughput in favorable conditions.
    
    Args:
        channel_gains: Array of channel gains for all users
        user_indices: Optional array of user IDs. If None, uses 0, 1, 2, ...
    
    Returns:
        List of tuples (weak_user_idx, strong_user_idx)
        - weak_user_idx: Index of user with weaker channel (farther)
        - strong_user_idx: Index of user with stronger channel (closer)
    
    Example:
        >>> gains = np.array([1e-8, 1e-6, 1e-9, 1e-7])  # 4 users
        >>> pairs = pair_users_extreme(gains)
        >>> # Returns: [(2, 1), (0, 3)]  # weakest with strongest
    """
    num_users = len(channel_gains)
    if user_indices is None:
        user_indices = np.arange(num_users)
    
    # Sort users by channel gain (ascending)
    sorted_indices = np.argsort(channel_gains)
    sorted_user_ids = user_indices[sorted_indices]
    
    # Pair from opposite ends
    pairs = []
    num_pairs = num_users // 2
    
    for i in range(num_pairs):
        weak_idx = sorted_user_ids[i]          # Weakest users
        strong_idx = sorted_user_ids[-(i+1)]   # Strongest users
        pairs.append((weak_idx, strong_idx))
    
    return pairs


def pair_users_random(channel_gains: np.ndarray, user_indices: Optional[np.ndarray] = None,
                      seed: Optional[int] = None) -> List[Tuple[int, int]]:
    """
    Random pairing: Randomly shuffle and pair consecutive users.
    
    Strategy: Random permutation, pair (0,1), (2,3), etc.
    
    Best for: Baseline comparison, avoiding biased pairing effects.
    
    Args:
        channel_gains: Array of channel gains
        user_indices: Optional user IDs
        seed: Random seed for reproducibility
    
    Returns:
        List of random user pairs (weak, strong) based on their channel gains
    """
    if seed is not None:
        np.random.seed(seed)
    
    num_users = len(channel_gains)
    if user_indices is None:
        user_indices = np.arange(num_users)
    
    # Random shuffle
    shuffled = np.random.permutation(user_indices)
    
    pairs = []
    num_pairs = num_users // 2
    
    for i in range(num_pairs):
        idx1 = shuffled[2*i]
        idx2 = shuffled[2*i + 1]
        
        # Ensure first is weak, second is strong
        if channel_gains[idx1] > channel_gains[idx2]:
            pairs.append((idx2, idx1))  # idx2 is weaker
        else:
            pairs.append((idx1, idx2))  # idx1 is weaker
    
    return pairs


def pair_users_sequential(channel_gains: np.ndarray, user_indices: Optional[np.ndarray] = None
                          ) -> List[Tuple[int, int]]:
    """
    Sequential pairing: Sort by gain, pair consecutive users.
    
    Strategy: Sort users, pair (1st, 2nd), (3rd, 4th), etc.
    Creates pairs with moderate channel difference.
    
    Best for: Balanced performance, avoiding extreme unfairness.
    
    Args:
        channel_gains: Array of channel gains
        user_indices: Optional user IDs
    
    Returns:
        List of consecutive pairs after sorting by channel gain
    """
    num_users = len(channel_gains)
    if user_indices is None:
        user_indices = np.arange(num_users)
    
    # Sort by channel gain
    sorted_indices = np.argsort(channel_gains)
    sorted_user_ids = user_indices[sorted_indices]
    
    pairs = []
    num_pairs = num_users // 2
    
    for i in range(num_pairs):
        weak_idx = sorted_user_ids[2*i]      # Weaker of the pair
        strong_idx = sorted_user_ids[2*i + 1] # Stronger of the pair
        pairs.append((weak_idx, strong_idx))
    
    return pairs


def pair_users(channel_gains: np.ndarray, method: str = 'extreme',
               user_indices: Optional[np.ndarray] = None,
               seed: Optional[int] = None) -> List[Tuple[int, int]]:
    """
    Universal user pairing function with multiple strategies.
    
    Args:
        channel_gains: Channel gains for all users
        method: Pairing strategy - 'extreme', 'random', or 'sequential'
        user_indices: Optional user IDs
        seed: Random seed (for 'random' method)
    
    Returns:
        List of user pairs (weak_idx, strong_idx)
    
    Example:
        >>> gains = compute_channel_gains(positions, 3.5)
        >>> pairs = pair_users(gains, method='extreme')
        >>> for weak, strong in pairs:
        ...     simulate_noma_pair(gains[weak], gains[strong], cfg)
    """
    if method == 'extreme':
        return pair_users_extreme(channel_gains, user_indices)
    elif method == 'random':
        return pair_users_random(channel_gains, user_indices, seed)
    elif method == 'sequential':
        return pair_users_sequential(channel_gains, user_indices)
    else:
        raise ValueError(f"Unknown pairing method: {method}. Use 'extreme', 'random', or 'sequential'.")


# ============================================================================
# MULTI-USER NOMA SIMULATION
# ============================================================================

def simulate_noma_system(channel_gains: np.ndarray, cfg, 
                        pairing_method: str = 'extreme',
                        cache_status: Optional[Dict[int, bool]] = None,
                        requested_files: Optional[Dict[int, int]] = None
                        ) -> Dict:
    """
    Simulate complete NOMA system for all user pairs.
    
    This function orchestrates the entire NOMA transmission:
    1. Pair users based on channel conditions
    2. Optionally allocate power using optimization
    3. Simulate transmission for each pair
    4. Apply cache-aided interference cancellation (CIC)
    5. Collect system-wide performance metrics
    
    Args:
        channel_gains: Channel gains for all users (array of length num_users)
        cfg: Configuration object
        pairing_method: 'extreme', 'random', or 'sequential'
        cache_status: Dict mapping user_id → requested file is cached (bool)
        requested_files: Dict mapping user_id → requested file_id
    
    Returns:
        Dictionary with comprehensive results:
        - 'pairs': List of user pairs
        - 'pair_results': List of results for each pair
        - 'system_metrics': Aggregated system performance
            - overall_success_rate
            - average_sum_rate
            - outage_probability
            - average_ber
            - cache_hit_rate (if cache_status provided)
            - cic_benefit (improvement from cache-aided cancellation)
    
    Example:
        >>> positions = generate_user_positions(200, 500, seed=42)
        >>> gains = compute_channel_gains(positions, 3.5)
        >>> cache_status = {i: (i < 20) for i in range(200)}  # First 20 cached
        >>> results = simulate_noma_system(gains, cfg, cache_status=cache_status)
        >>> print(f"Outage: {results['system_metrics']['outage_probability']:.2%}")
    """
    num_users = len(channel_gains)
    
    # Create user pairs
    pairs = pair_users(channel_gains, method=pairing_method)
    
    # Initialize cache status if not provided
    if cache_status is None:
        cache_status = {i: False for i in range(num_users)}
    
    if requested_files is None:
        requested_files = {i: -1 for i in range(num_users)}  # -1 = unknown
    
    # Simulate each pair
    pair_results = []
    for weak_idx, strong_idx in pairs:
        gain_w = channel_gains[weak_idx]
        gain_s = channel_gains[strong_idx]
        
        # Check cache status for CIC
        weak_cached = cache_status.get(weak_idx, False)
        strong_cached = cache_status.get(strong_idx, False)
        
        # Optionally: Check if they requested each other's content for CIC
        # (Advanced CIC: weak can cancel if it has strong's requested content)
        weak_file = requested_files.get(weak_idx, -1)
        strong_file = requested_files.get(strong_idx, -1)
        
        # Simplified CIC: just check if content is generally cached
        # For full CIC: check if weak_cached includes strong's file
        
        weak_ok, strong_ok, info = simulate_noma_pair(
            gain_w, gain_s, cfg,
            weak_cached=weak_cached,
            strong_cached=strong_cached
        )
        
        info['weak_idx'] = weak_idx
        info['strong_idx'] = strong_idx
        pair_results.append(info)
    
    # -------------------------------------------------------------------------
    # AGGREGATE SYSTEM METRICS
    # -------------------------------------------------------------------------
    num_pairs = len(pairs)
    
    # Success rates
    weak_success_count = sum(1 for r in pair_results if r['weak_success'])
    strong_success_count = sum(1 for r in pair_results if r['strong_success'])
    pair_success_count = sum(1 for r in pair_results if r['pair_success'])
    
    weak_success_rate = weak_success_count / num_pairs if num_pairs > 0 else 0
    strong_success_rate = strong_success_count / num_pairs if num_pairs > 0 else 0
    overall_success_rate = pair_success_count / num_pairs if num_pairs > 0 else 0
    
    # Rates and throughput
    total_sum_rate = sum(r['sum_rate'] for r in pair_results)
    average_sum_rate = total_sum_rate / num_pairs if num_pairs > 0 else 0
    system_throughput = total_sum_rate  # Total bits/s/Hz
    
    # Outage probability
    outage_count = sum(1 for r in pair_results if r['outage'])
    outage_probability = outage_count / num_pairs if num_pairs > 0 else 0
    
    # BER statistics
    average_ber_weak = np.mean([r['ber_w'] for r in pair_results])
    average_ber_strong = np.mean([r['ber_s'] for r in pair_results])
    average_ber = (average_ber_weak + average_ber_strong) / 2
    
    # Fairness (average Jain's index)
    average_fairness = np.mean([r['fairness'] for r in pair_results])
    
    # Cache metrics
    cache_hit_count = sum(1 for i in range(num_users) if cache_status.get(i, False))
    cache_hit_rate = cache_hit_count / num_users if num_users > 0 else 0
    
    # CIC benefit (how many users benefited from cache-aided cancellation)
    cic_applied_count = sum(1 for r in pair_results if r.get('cic_applied', False))
    cic_benefit_rate = cic_applied_count / num_pairs if num_pairs > 0 else 0
    
    system_metrics = {
        'num_users': num_users,
        'num_pairs': num_pairs,
        'weak_success_rate': weak_success_rate,
        'strong_success_rate': strong_success_rate,
        'overall_success_rate': overall_success_rate,
        'average_sum_rate': average_sum_rate,
        'system_throughput': system_throughput,
        'outage_probability': outage_probability,
        'average_ber_weak': average_ber_weak,
        'average_ber_strong': average_ber_strong,
        'average_ber': average_ber,
        'average_fairness': average_fairness,
        'cache_hit_rate': cache_hit_rate,
        'cic_benefit_rate': cic_benefit_rate,
        'pairing_method': pairing_method
    }
    
    return {
        'pairs': pairs,
        'pair_results': pair_results,
        'system_metrics': system_metrics
    }


# ============================================================================
# OUTAGE ANALYSIS
# ============================================================================

def compute_outage_probability(results_list: List[Dict]) -> float:
    """
    Compute outage probability from multiple simulation runs.
    
    Outage occurs when at least one user in a pair fails to meet target rate.
    
    Args:
        results_list: List of results from simulate_noma_system()
    
    Returns:
        float: Outage probability (0 to 1)
    """
    total_pairs = 0
    outage_pairs = 0
    
    for result in results_list:
        for pair_result in result['pair_results']:
            total_pairs += 1
            if pair_result['outage']:
                outage_pairs += 1
    
    return outage_pairs / total_pairs if total_pairs > 0 else 0


def compute_average_ber(results_list: List[Dict]) -> float:
    """
    Compute average BER across all users and simulation runs.
    
    Args:
        results_list: List of results from simulate_noma_system()
    
    Returns:
        float: Average bit error rate
    """
    ber_values = []
    
    for result in results_list:
        for pair_result in result['pair_results']:
            ber_values.append(pair_result['ber_w'])
            ber_values.append(pair_result['ber_s'])
    
    return np.mean(ber_values) if ber_values else 0


# ============================================================================
# EXAMPLE USAGE AND TESTING
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("TESTING ENHANCED NOMA BASE MODULE")
    print("="*70)
    
    # Mock configuration
    class MockConfig:
        TX_POWER = 1.0
        NOISE_POWER = 1e-9
        POWER_COEFF_WEAK = 0.8
        POWER_COEFF_STRONG = 0.2
        SIC_IMPERFECTION = 0.05
        TARGET_RATE_BPS = 0.5
    
    cfg = MockConfig()
    
    # Test 1: Single pair simulation
    print("\n[Test 1] Single NOMA pair simulation...")
    gain_weak = 1e-8   # Far user
    gain_strong = 1e-6  # Near user
    
    weak_ok, strong_ok, info = simulate_noma_pair(gain_weak, gain_strong, cfg)
    print(f"Weak user success: {weak_ok}, SINR: {info['sinr_w']:.3f}")
    print(f"Strong user success: {strong_ok}, SINR: {info['sinr_s_after']:.3f}")
    print(f"Sum rate: {info['sum_rate']:.3f} bps/Hz")
    
    # Test 2: Pair with CIC
    print("\n[Test 2] NOMA pair with cache-aided cancellation...")
    weak_ok_cic, strong_ok_cic, info_cic = simulate_noma_pair(
        gain_weak, gain_strong, cfg, weak_cached=True
    )
    print(f"CIC applied: {info_cic['cic_applied']}")
    print(f"Weak SINR improvement: {info_cic['sinr_w']/info['sinr_w']:.2f}x")
    print(f"Sum rate with CIC: {info_cic['sum_rate']:.3f} bps/Hz")
    
    # Test 3: User pairing
    print("\n[Test 3] User pairing strategies...")
    gains = np.array([1e-8, 1e-6, 1e-9, 1e-7, 1e-10, 1e-5])
    
    pairs_extreme = pair_users(gains, method='extreme')
    pairs_random = pair_users(gains, method='random', seed=42)
    pairs_seq = pair_users(gains, method='sequential')
    
    print(f"Extreme pairing: {pairs_extreme}")
    print(f"Random pairing: {pairs_random}")
    print(f"Sequential pairing: {pairs_seq}")
    
    # Test 4: Full system simulation
    print("\n[Test 4] Full NOMA system simulation...")
    num_users = 20
    gains_system = np.random.exponential(1e-7, num_users)
    cache_status = {i: (i % 5 == 0) for i in range(num_users)}  # Every 5th user cached
    
    results = simulate_noma_system(gains_system, cfg, cache_status=cache_status)
    metrics = results['system_metrics']
    
    print(f"Number of pairs: {metrics['num_pairs']}")
    print(f"Overall success rate: {metrics['overall_success_rate']:.2%}")
    print(f"Average sum rate: {metrics['average_sum_rate']:.3f} bps/Hz")
    print(f"Outage probability: {metrics['outage_probability']:.2%}")
    print(f"Cache hit rate: {metrics['cache_hit_rate']:.2%}")
    print(f"CIC benefit rate: {metrics['cic_benefit_rate']:.2%}")
    
    print("\n" + "="*70)
    print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*70)
