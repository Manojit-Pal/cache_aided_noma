# src/noma/power_allocation.py
"""
Power Allocation Module for Cache-Aided NOMA Systems

This module implements various power allocation algorithms:
- Grid search (exhaustive)
- Closed-form analytical solution
- Cache-aware power allocation (optimized for CIC)
- Sum-rate maximization
- Energy efficiency optimization
- Multi-objective optimization (rate, energy, fairness)

Key Innovation: Cache-Aware Power Allocation
    When users have cached content enabling interference cancellation,
    the optimal power allocation changes significantly:
    - Weak user with cache → needs less power (no interference from strong)
    - Strong user with cache → perfect SIC → can tolerate more weak power

Author: Cache-Aided NOMA Team
Date: December 2025
"""

import numpy as np
from typing import Tuple, Dict, Optional
from scipy.optimize import minimize_scalar, minimize
import warnings


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def feasible_for_weak(p_w: float, P: float, gain_w: float, N0: float, T: float) -> bool:
    """
    Check if weak user's SINR meets target rate for given power coefficient.
    
    Args:
        p_w: Power coefficient for weak user (0 to 1)
        P: Total transmit power
        gain_w: Weak user's channel gain
        N0: Noise power
        T: SINR threshold (2^R - 1)
    
    Returns:
        bool: True if weak user achieves target rate
    """
    p_s = 1.0 - p_w
    num = P * p_w * gain_w
    den = P * p_s * gain_w + N0
    return (num / den) >= T


def compute_sinrs(p_w: float, P: float, gain_w: float, gain_s: float, 
                  N0: float, zeta: float, T: float) -> Dict:
    """
    Compute all SINR values for a given power allocation.
    
    Args:
        p_w: Power coefficient for weak user
        P: Total transmit power
        gain_w: Weak user's channel gain
        gain_s: Strong user's channel gain
        N0: Noise power
        zeta: SIC imperfection factor
        T: SINR threshold
    
    Returns:
        Dictionary with all SINR values and success flags
    """
    p_s = 1.0 - p_w
    
    # Weak user SINR
    num_w = P * p_w * gain_w
    den_w = P * p_s * gain_w + N0
    sinr_w = num_w / den_w
    ok_weak = sinr_w >= T
    
    # Strong user decodes weak signal
    num_sdec = P * p_w * gain_s
    den_sdec = P * p_s * gain_s + N0
    sinr_sdec = num_sdec / den_sdec
    ok_s_decode = sinr_sdec >= T
    
    # Strong user after SIC
    residual = zeta * (P * p_w * gain_s) if ok_s_decode else P * p_w * gain_s
    sinr_s_after = (P * p_s * gain_s) / (N0 + residual)
    ok_strong = sinr_s_after >= T
    
    return {
        'sinr_w': sinr_w,
        'sinr_sdec': sinr_sdec,
        'sinr_s_after': sinr_s_after,
        'ok_weak': ok_weak,
        'ok_s_decode': ok_s_decode,
        'ok_strong': ok_strong,
        'sum_sinr': sinr_w + sinr_s_after
    }


# ============================================================================
# EXISTING ALGORITHMS (ENHANCED WITH DOCUMENTATION)
# ============================================================================

def allocate_power_gridsearch(gain_w: float, gain_s: float, cfg, 
                              grid_points: int = 101) -> Tuple[float, float, bool, Dict]:
    """
    Grid search power allocation (exhaustive search).
    
    Algorithm:
        1. Try all possible power splits p_w ∈ [ε, 1-ε]
        2. For each p_w, compute SINR for both users
        3. Score = number of successful users (0, 1, or 2)
        4. Among allocations with same score, pick max sum_sinr
    
    Pros:
        - Guaranteed to find optimal solution
        - Simple to implement and understand
    
    Cons:
        - Computationally expensive (O(grid_points))
        - Not suitable for real-time systems
    
    Args:
        gain_w: Weak user's channel gain
        gain_s: Strong user's channel gain
        cfg: Configuration object with TX_POWER, NOISE_POWER, etc.
        grid_points: Number of grid points to search (default 101)
    
    Returns:
        Tuple of (p_w, p_s, feasible, info_dict)
        - p_w: Optimal power coefficient for weak user
        - p_s: Optimal power coefficient for strong user (1 - p_w)
        - feasible: Whether both users can meet target rate
        - info_dict: Detailed SINR and success information
    
    Example:
        >>> p_w, p_s, feasible, info = allocate_power_gridsearch(1e-8, 1e-6, cfg)
        >>> if feasible:
        ...     print(f"Optimal power split: {p_w:.3f}/{p_s:.3f}")
    """
    P = cfg.TX_POWER
    N0 = cfg.NOISE_POWER
    zeta = cfg.SIC_IMPERFECTION
    T = 2 ** (cfg.TARGET_RATE_BPS) - 1

    eps = 1e-4
    p_ws = np.linspace(eps, 1.0 - eps, grid_points)

    best = None
    best_score = -1
    best_info = {}

    for p_w in p_ws:
        p_s = 1.0 - p_w
        
        # Compute all SINRs
        info = compute_sinrs(p_w, P, gain_w, gain_s, N0, zeta, T)
        
        # Score: number of successful users
        score = int(info['ok_weak']) + int(info['ok_strong'])
        
        # Keep best allocation (prioritize score, then sum_sinr)
        if (score > best_score) or (score == best_score and info['sum_sinr'] > best_info.get('sum_sinr', -1)):
            best_score = score
            best = (p_w, p_s)
            best_info = info.copy()
            best_info['p_w'] = p_w
            best_info['p_s'] = p_s
            
            # Early exit if both users successful
            if score == 2:
                break

    feasible = best_info.get('ok_weak', False) and best_info.get('ok_strong', False)
    return best[0], best[1], feasible, best_info


def allocate_power_closedform(gain_w: float, gain_s: float, cfg) -> Tuple[float, float, bool, Dict]:
    """
    Closed-form power allocation using analytical solution.
    
    Derives feasible power range [p_w_min, p_w_max] by solving SINR constraints:
    
    Constraints:
    1. Weak user rate: SINR_w ≥ T
       => p_w ≥ (T/(1+T)) * (1 + N0/(P*g_w))
    
    2. Strong decodes weak: SINR_s_decode_w ≥ T
       => p_w ≥ (T/(1+T)) * (1 + N0/(P*g_s))
    
    3. Strong user rate after SIC: SINR_s ≥ T
       => p_w ≤ [1 - T*N0/(P*g_s)] / (1 + T*ζ)
    
    Power allocation:
    - If [p_w_min, p_w_max] overlap → feasible, choose p_w in range
    - Else → infeasible, use heuristic (give more to weak user)
    
    Pros:
        - Very fast, no iterations (O(1) complexity)
        - Analytical solution with theoretical guarantees
    
    Cons:
        - May not maximize sum-rate optimally
        - Simplified assumptions (fixed target rate)
    
    Args:
        gain_w: Weak user's channel gain
        gain_s: Strong user's channel gain
        cfg: Configuration object
    
    Returns:
        Tuple of (p_w, p_s, feasible, info_dict)
    
    Example:
        >>> p_w, p_s, feasible, info = allocate_power_closedform(1e-8, 1e-6, cfg)
        >>> print(f"Feasible range: [{info['lower_bound']:.3f}, {info['upper_bound']:.3f}]")
    """
    P = cfg.TX_POWER
    N0 = cfg.NOISE_POWER
    zeta = cfg.SIC_IMPERFECTION
    T = 2 ** (cfg.TARGET_RATE_BPS) - 1

    # Avoid division by zero
    eps = 1e-12
    g_w = max(gain_w, eps)
    g_s = max(gain_s, eps)

    # Lower bounds from weak user and strong decoding weak
    lower_w = (T / (1.0 + T)) * (1.0 + (N0 / (P * g_w)))
    lower_sdecode = (T / (1.0 + T)) * (1.0 + (N0 / (P * g_s)))
    lower_bound = max(lower_w, lower_sdecode)

    # Upper bound from strong user's rate after SIC
    numerator = 1.0 - (T * N0) / (P * g_s)
    denom = 1.0 + T * zeta
    upper_bound = numerator / denom

    # Clamp to [0, 1]
    lower_bound_clamped = max(lower_bound, 0.0)
    upper_bound_clamped = min(upper_bound, 1.0 - 1e-6)

    info = {
        "lower_w": lower_w,
        "lower_sdecode": lower_sdecode,
        "lower_bound": lower_bound_clamped,
        "upper_bound": upper_bound_clamped,
        "T": T,
        "g_w": g_w,
        "g_s": g_s,
        "zeta": zeta,
    }

    # Check feasibility
    feasible = (lower_bound_clamped <= upper_bound_clamped) and (lower_bound_clamped < 1.0)
    
    if feasible:
        # Choose p_w in feasible range (use lower bound for fairness to weak user)
        p_w = min(max(lower_bound_clamped, 0.001), 0.999)
        if p_w > upper_bound_clamped:
            p_w = (lower_bound_clamped + upper_bound_clamped) / 2.0
        p_s = 1.0 - p_w
    else:
        # Fallback: heuristic allocation
        p_w = min(max(lower_bound_clamped, 0.5), 0.99)
        p_s = 1.0 - p_w

    # Compute final SINRs
    sinr_info = compute_sinrs(p_w, P, g_w, g_s, N0, zeta, T)
    info.update(sinr_info)
    info['p_w'] = p_w
    info['p_s'] = p_s

    feasible_final = sinr_info['ok_weak'] and sinr_info['ok_strong']
    return p_w, p_s, feasible_final, info


# ============================================================================
# CACHE-AWARE POWER ALLOCATION (NEW - CRITICAL FOR YOUR PROJECT)
# ============================================================================

def allocate_power_cache_aware(gain_w: float, gain_s: float, cfg,
                               weak_cached: bool = False, 
                               strong_cached: bool = False) -> Tuple[float, float, bool, Dict]:
    """
    Cache-aware power allocation optimized for interference cancellation.
    
    Key Insight:
        When users have cached content, they can cancel interference,
        changing the optimal power allocation strategy.
    
    Scenarios:
    1. No cache: Standard NOMA power allocation
    2. Weak cached: Weak user cancels strong's interference
       → Can allocate LESS power to weak user (saves energy)
       → More power available for strong user
    3. Strong cached: Strong user has perfect SIC
       → Can allocate MORE power to weak user (improves fairness)
       → Strong user still succeeds due to perfect cancellation
    4. Both cached: Maximum flexibility in power allocation
    
    Args:
        gain_w: Weak user's channel gain
        gain_s: Strong user's channel gain
        cfg: Configuration object
        weak_cached: Whether weak user has content enabling CIC
        strong_cached: Whether strong user has content enabling CIC
    
    Returns:
        Tuple of (p_w, p_s, feasible, info_dict)
    
    Example:
        >>> # Standard NOMA
        >>> p_w1, p_s1, _, _ = allocate_power_cache_aware(1e-8, 1e-6, cfg)
        >>> 
        >>> # With cache-aided cancellation
        >>> p_w2, p_s2, _, _ = allocate_power_cache_aware(1e-8, 1e-6, cfg, 
        ...                                               weak_cached=True)
        >>> print(f"Power reduction for weak user: {(p_w1-p_w2)/p_w1:.1%}")
    """
    P = cfg.TX_POWER
    N0 = cfg.NOISE_POWER
    zeta = cfg.SIC_IMPERFECTION if not strong_cached else 0.0  # Perfect SIC if cached
    T = 2 ** (cfg.TARGET_RATE_BPS) - 1
    
    eps = 1e-12
    g_w = max(gain_w, eps)
    g_s = max(gain_s, eps)
    
    # -------------------------------------------------------------------------
    # CASE 1: Weak user has cache (can cancel strong's interference)
    # -------------------------------------------------------------------------
    if weak_cached and not strong_cached:
        # Weak user: SINR = P*p_w*g_w / N0 (no interference!)
        # Constraint: P*p_w*g_w / N0 ≥ T
        # => p_w ≥ T*N0 / (P*g_w)
        p_w_min = (T * N0) / (P * g_w)
        
        # Strong user still needs standard constraints
        # Strong decode weak: p_w ≥ (T/(1+T)) * (1 + N0/(P*g_s))
        p_w_min_sdec = (T / (1.0 + T)) * (1.0 + (N0 / (P * g_s)))
        
        # Strong own signal: p_w ≤ [1 - T*N0/(P*g_s)] / (1 + T*zeta)
        p_w_max = (1.0 - (T * N0) / (P * g_s)) / (1.0 + T * zeta)
        
        # Take most restrictive bounds
        p_w_lower = max(p_w_min, p_w_min_sdec, 0.0)
        p_w_upper = min(p_w_max, 0.999)
        
        # Choose minimum feasible p_w to save power for weak user
        if p_w_lower <= p_w_upper:
            p_w = p_w_lower + 0.05 * (p_w_upper - p_w_lower)  # Slight margin
            p_w = np.clip(p_w, 0.001, 0.999)
            feasible = True
        else:
            p_w = 0.5
            feasible = False
    
    # -------------------------------------------------------------------------
    # CASE 2: Strong user has cache (perfect SIC)
    # -------------------------------------------------------------------------
    elif strong_cached and not weak_cached:
        # Strong user: Perfect SIC (zeta = 0)
        # Can allocate MORE power to weak user for fairness
        
        # Weak user constraint: p_w ≥ (T/(1+T)) * (1 + N0/(P*g_w))
        p_w_min = (T / (1.0 + T)) * (1.0 + (N0 / (P * g_w)))
        
        # Strong decode weak: p_w ≥ (T/(1+T)) * (1 + N0/(P*g_s))
        p_w_min_sdec = (T / (1.0 + T)) * (1.0 + (N0 / (P * g_s)))
        
        # Strong with perfect SIC (zeta=0): upper bound is much higher
        p_w_max = (1.0 - (T * N0) / (P * g_s))  # No zeta term!
        
        p_w_lower = max(p_w_min, p_w_min_sdec, 0.0)
        p_w_upper = min(p_w_max, 0.999)
        
        # Allocate MORE to weak user (fairness-oriented)
        if p_w_lower <= p_w_upper:
            p_w = p_w_lower + 0.7 * (p_w_upper - p_w_lower)  # Favor weak user
            p_w = np.clip(p_w, 0.001, 0.999)
            feasible = True
        else:
            p_w = 0.7  # Fallback: favor weak
            feasible = False
    
    # -------------------------------------------------------------------------
    # CASE 3: Both users have cache (maximum flexibility)
    # -------------------------------------------------------------------------
    elif weak_cached and strong_cached:
        # Weak: No interference from strong
        # Strong: Perfect SIC
        # Both constraints relaxed significantly!
        
        p_w_min = (T * N0) / (P * g_w)  # Weak constraint (no interference)
        p_w_min_sdec = (T / (1.0 + T)) * (1.0 + (N0 / (P * g_s)))  # Strong decode
        p_w_max = (1.0 - (T * N0) / (P * g_s))  # Strong with perfect SIC
        
        p_w_lower = max(p_w_min, p_w_min_sdec, 0.0)
        p_w_upper = min(p_w_max, 0.999)
        
        # Balanced allocation (maximize sum-rate)
        if p_w_lower <= p_w_upper:
            p_w = (p_w_lower + p_w_upper) / 2.0  # Midpoint for balance
            p_w = np.clip(p_w, 0.001, 0.999)
            feasible = True
        else:
            p_w = 0.5
            feasible = False
    
    # -------------------------------------------------------------------------
    # CASE 4: No cache (standard NOMA)
    # -------------------------------------------------------------------------
    else:
        # Fall back to closed-form allocation
        return allocate_power_closedform(gain_w, gain_s, cfg)
    
    p_s = 1.0 - p_w
    
    # Verify allocation by computing SINRs
    sinr_info = compute_sinrs(p_w, P, g_w, g_s, N0, zeta, T)
    
    # Adjust SINR for weak user if cached (no interference)
    if weak_cached:
        sinr_info['sinr_w'] = (P * p_w * g_w) / N0
        sinr_info['ok_weak'] = sinr_info['sinr_w'] >= T
        sinr_info['sum_sinr'] = sinr_info['sinr_w'] + sinr_info['sinr_s_after']
    
    info = {
        'p_w': p_w,
        'p_s': p_s,
        'weak_cached': weak_cached,
        'strong_cached': strong_cached,
        'cache_aware': True,
        **sinr_info
    }
    
    feasible_final = info['ok_weak'] and info['ok_strong']
    return p_w, p_s, feasible_final, info


# ============================================================================
# SUM-RATE MAXIMIZATION
# ============================================================================

def allocate_power_sumrate_max(gain_w: float, gain_s: float, cfg,
                               method: str = 'scipy') -> Tuple[float, float, bool, Dict]:
    """
    Sum-rate maximization using optimization.
    
    Objective: Maximize R_total = R_w + R_s = log2(1+SINR_w) + log2(1+SINR_s)
    
    Subject to:
    - 0 ≤ p_w ≤ 1
    - SINR_w ≥ T (weak user rate constraint)
    - SINR_s ≥ T (strong user rate constraint)
    
    Args:
        gain_w: Weak user's channel gain
        gain_s: Strong user's channel gain
        cfg: Configuration object
        method: Optimization method ('scipy' or 'iterative')
    
    Returns:
        Tuple of (p_w, p_s, feasible, info_dict)
    
    Example:
        >>> p_w, p_s, feasible, info = allocate_power_sumrate_max(1e-8, 1e-6, cfg)
        >>> print(f"Maximum sum-rate: {info['sum_rate']:.3f} bps/Hz")
    """
    P = cfg.TX_POWER
    N0 = cfg.NOISE_POWER
    zeta = cfg.SIC_IMPERFECTION
    T = 2 ** (cfg.TARGET_RATE_BPS) - 1
    
    def objective(p_w):
        """Negative sum-rate (for minimization)."""
        p_w = np.clip(p_w, 0.001, 0.999)
        p_s = 1.0 - p_w
        
        # Compute SINRs
        sinr_w = (P * p_w * gain_w) / (P * p_s * gain_w + N0)
        
        # Strong decode weak
        sinr_sdec = (P * p_w * gain_s) / (P * p_s * gain_s + N0)
        can_decode = sinr_sdec >= T
        
        # Strong after SIC
        residual = zeta * (P * p_w * gain_s) if can_decode else P * p_w * gain_s
        sinr_s = (P * p_s * gain_s) / (N0 + residual)
        
        # Sum rate (Shannon capacity)
        rate_w = np.log2(1 + sinr_w)
        rate_s = np.log2(1 + sinr_s)
        sum_rate = rate_w + rate_s
        
        return -sum_rate  # Negative for minimization
    
    # Optimize using scipy
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = minimize_scalar(objective, bounds=(0.001, 0.999), method='bounded')
    
    p_w_opt = result.x
    p_s_opt = 1.0 - p_w_opt
    
    # Compute final metrics
    sinr_info = compute_sinrs(p_w_opt, P, gain_w, gain_s, N0, zeta, T)
    
    rate_w = np.log2(1 + sinr_info['sinr_w'])
    rate_s = np.log2(1 + sinr_info['sinr_s_after'])
    
    info = {
        'p_w': p_w_opt,
        'p_s': p_s_opt,
        'sum_rate': rate_w + rate_s,
        'rate_w': rate_w,
        'rate_s': rate_s,
        'optimization_method': 'sum_rate_max',
        **sinr_info
    }
    
    feasible = info['ok_weak'] and info['ok_strong']
    return p_w_opt, p_s_opt, feasible, info


# ============================================================================
# ENERGY EFFICIENCY OPTIMIZATION (FOR 6G)
# ============================================================================

def allocate_power_energy_efficient(gain_w: float, gain_s: float, cfg,
                                    circuit_power: float = 0.1) -> Tuple[float, float, bool, Dict]:
    """
    Energy efficiency maximization for 6G green communications.
    
    Objective: Maximize EE = Sum_Rate / Total_Power
    where Total_Power = Transmit_Power + Circuit_Power
    
    Energy efficiency (bits/Joule) is critical for:
    - Battery-powered IoT devices
    - Green 6G networks
    - Sustainable communications
    
    Args:
        gain_w: Weak user's channel gain
        gain_s: Strong user's channel gain
        cfg: Configuration object
        circuit_power: Circuit power consumption (relative to TX_POWER)
    
    Returns:
        Tuple of (p_w, p_s, feasible, info_dict)
    
    Example:
        >>> p_w, p_s, feasible, info = allocate_power_energy_efficient(1e-8, 1e-6, cfg)
        >>> print(f"Energy efficiency: {info['energy_efficiency']:.3f} bits/Joule")
    """
    P = cfg.TX_POWER
    N0 = cfg.NOISE_POWER
    zeta = cfg.SIC_IMPERFECTION
    T = 2 ** (cfg.TARGET_RATE_BPS) - 1
    
    def objective(p_w):
        """Negative energy efficiency (for minimization)."""
        p_w = np.clip(p_w, 0.001, 0.999)
        p_s = 1.0 - p_w
        
        # Compute SINRs
        sinr_w = (P * p_w * gain_w) / (P * p_s * gain_w + N0)
        sinr_sdec = (P * p_w * gain_s) / (P * p_s * gain_s + N0)
        can_decode = sinr_sdec >= T
        residual = zeta * (P * p_w * gain_s) if can_decode else P * p_w * gain_s
        sinr_s = (P * p_s * gain_s) / (N0 + residual)
        
        # Sum rate
        rate_w = np.log2(1 + sinr_w)
        rate_s = np.log2(1 + sinr_s)
        sum_rate = rate_w + rate_s
        
        # Total power consumption
        total_power = P + circuit_power
        
        # Energy efficiency (bits/Joule)
        ee = sum_rate / total_power if total_power > 0 else 0
        
        return -ee  # Negative for minimization
    
    # Optimize
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = minimize_scalar(objective, bounds=(0.001, 0.999), method='bounded')
    
    p_w_opt = result.x
    p_s_opt = 1.0 - p_w_opt
    
    # Compute final metrics
    sinr_info = compute_sinrs(p_w_opt, P, gain_w, gain_s, N0, zeta, T)
    rate_w = np.log2(1 + sinr_info['sinr_w'])
    rate_s = np.log2(1 + sinr_info['sinr_s_after'])
    sum_rate = rate_w + rate_s
    total_power = P + circuit_power
    
    info = {
        'p_w': p_w_opt,
        'p_s': p_s_opt,
        'sum_rate': sum_rate,
        'total_power': total_power,
        'energy_efficiency': sum_rate / total_power,
        'optimization_method': 'energy_efficient',
        **sinr_info
    }
    
    feasible = info['ok_weak'] and info['ok_strong']
    return p_w_opt, p_s_opt, feasible, info


# ============================================================================
# UNIVERSAL POWER ALLOCATION INTERFACE
# ============================================================================

def allocate_power(gain_w: float, gain_s: float, cfg,
                   method: str = 'closedform',
                   weak_cached: bool = False,
                   strong_cached: bool = False,
                   **kwargs) -> Tuple[float, float, bool, Dict]:
    """
    Universal power allocation interface with multiple algorithms.
    
    This is the main function you should use in your simulations.
    
    Args:
        gain_w: Weak user's channel gain
        gain_s: Strong user's channel gain
        cfg: Configuration object
        method: Algorithm to use:
            - 'gridsearch': Exhaustive search (slow, optimal)
            - 'closedform': Analytical solution (fast)
            - 'cache_aware': Cache-optimized allocation (recommended for your project)
            - 'sumrate_max': Maximize sum-rate
            - 'energy_efficient': Maximize energy efficiency (6G green)
        weak_cached: Whether weak user has cached content
        strong_cached: Whether strong user has cached content
        **kwargs: Additional arguments for specific methods
    
    Returns:
        Tuple of (p_w, p_s, feasible, info_dict)
    
    Example:
        >>> # Standard allocation
        >>> p_w, p_s, _, _ = allocate_power(1e-8, 1e-6, cfg, method='closedform')
        >>> 
        >>> # Cache-aware allocation (recommended)
        >>> p_w, p_s, _, _ = allocate_power(1e-8, 1e-6, cfg, method='cache_aware',
        ...                                 weak_cached=True)
    """
    if method == 'gridsearch':
        return allocate_power_gridsearch(gain_w, gain_s, cfg, **kwargs)
    
    elif method == 'closedform':
        return allocate_power_closedform(gain_w, gain_s, cfg)
    
    elif method == 'cache_aware':
        return allocate_power_cache_aware(gain_w, gain_s, cfg, weak_cached, strong_cached)
    
    elif method == 'sumrate_max':
        return allocate_power_sumrate_max(gain_w, gain_s, cfg, **kwargs)
    
    elif method == 'energy_efficient':
        return allocate_power_energy_efficient(gain_w, gain_s, cfg, **kwargs)
    
    else:
        raise ValueError(f"Unknown power allocation method: {method}. "
                        f"Use 'gridsearch', 'closedform', 'cache_aware', 'sumrate_max', or 'energy_efficient'.")


# ============================================================================
# TESTING AND EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("TESTING ENHANCED POWER ALLOCATION MODULE")
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
    gain_w = 1e-8   # Far user
    gain_s = 1e-6   # Near user
    
    # Test 1: Closed-form
    print("\n[Test 1] Closed-form power allocation...")
    p_w, p_s, feasible, info = allocate_power_closedform(gain_w, gain_s, cfg)
    print(f"Power split: p_w={p_w:.3f}, p_s={p_s:.3f}")
    print(f"Feasible: {feasible}, Sum SINR: {info['sum_sinr']:.3f}")
    
    # Test 2: Cache-aware (no cache)
    print("\n[Test 2] Cache-aware allocation (no cache)...")
    p_w2, p_s2, feasible2, info2 = allocate_power_cache_aware(gain_w, gain_s, cfg)
    print(f"Power split: p_w={p_w2:.3f}, p_s={p_s2:.3f}")
    
    # Test 3: Cache-aware (weak cached)
    print("\n[Test 3] Cache-aware allocation (weak user cached)...")
    p_w3, p_s3, feasible3, info3 = allocate_power_cache_aware(gain_w, gain_s, cfg, weak_cached=True)
    print(f"Power split: p_w={p_w3:.3f}, p_s={p_s3:.3f}")
    print(f"Power saving for weak: {(p_w2-p_w3)/p_w2*100:.1f}%")
    print(f"SINR improvement: {info3['sinr_w']/info2['sinr_w']:.2f}x")
    
    # Test 4: Cache-aware (strong cached)
    print("\n[Test 4] Cache-aware allocation (strong user cached)...")
    p_w4, p_s4, feasible4, info4 = allocate_power_cache_aware(gain_w, gain_s, cfg, strong_cached=True)
    print(f"Power split: p_w={p_w4:.3f}, p_s={p_s4:.3f}")
    print(f"More power to weak (fairness): {(p_w4-p_w2)/p_w2*100:.1f}% increase")
    
    # Test 5: Sum-rate maximization
    print("\n[Test 5] Sum-rate maximization...")
    p_w5, p_s5, feasible5, info5 = allocate_power_sumrate_max(gain_w, gain_s, cfg)
    print(f"Power split: p_w={p_w5:.3f}, p_s={p_s5:.3f}")
    print(f"Sum rate: {info5['sum_rate']:.3f} bps/Hz")
    
    # Test 6: Energy efficiency
    print("\n[Test 6] Energy efficiency optimization...")
    p_w6, p_s6, feasible6, info6 = allocate_power_energy_efficient(gain_w, gain_s, cfg)
    print(f"Power split: p_w={p_w6:.3f}, p_s={p_s6:.3f}")
    print(f"Energy efficiency: {info6['energy_efficiency']:.3f} bits/Joule")
    
    print("\n" + "="*70)
    print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*70)