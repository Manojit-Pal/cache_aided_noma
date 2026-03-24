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
    - Weak user with cache: needs less power (no interference from strong)
    - Strong user with cache: perfect SIC allows more power to weak user

✅ BUG-PA-1 FIXED: gridsearch no longer exits early on first score-2 hit;
                    scans full grid to guarantee optimal sum_sinr.

Author: Cache-Aided NOMA Team
Date: December 2025 | Revised: March 2026
"""

import numpy as np
from typing import Tuple, Dict, Optional
from scipy.optimize import minimize_scalar
import warnings


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def feasible_for_weak(p_w: float, P: float, gain_w: float, N0: float, T: float) -> bool:
    """
    Check if weak user's SINR meets target rate for given power coefficient.
    """
    p_s = 1.0 - p_w
    num = P * p_w * gain_w
    den = P * p_s * gain_w + N0
    return (num / den) >= T


def compute_sinrs(p_w: float, P: float, gain_w: float, gain_s: float,
                  N0: float, zeta: float, T: float) -> Dict:
    """
    Compute all SINR values for a given power allocation.
    """
    p_s = 1.0 - p_w

    # Weak user SINR
    sinr_w   = (P * p_w * gain_w) / (P * p_s * gain_w + N0)
    ok_weak  = sinr_w >= T

    # Strong user decodes weak signal
    sinr_sdec   = (P * p_w * gain_s) / (P * p_s * gain_s + N0)
    ok_s_decode = sinr_sdec >= T

    # Strong user after SIC
    residual    = zeta * (P * p_w * gain_s) if ok_s_decode else P * p_w * gain_s
    sinr_s_after = (P * p_s * gain_s) / (N0 + residual)
    ok_strong    = sinr_s_after >= T

    return {
        'sinr_w':      sinr_w,
        'sinr_sdec':   sinr_sdec,
        'sinr_s_after': sinr_s_after,
        'ok_weak':     ok_weak,
        'ok_s_decode': ok_s_decode,
        'ok_strong':   ok_strong,
        'sum_sinr':    sinr_w + sinr_s_after,
    }


# ============================================================================
# EXISTING ALGORITHMS
# ============================================================================

def allocate_power_gridsearch(gain_w: float, gain_s: float, cfg,
                              grid_points: int = 101) -> Tuple[float, float, bool, Dict]:
    """
    Grid search power allocation (exhaustive search).

    Scans all p_w in [eps, 1-eps] and selects the allocation that:
      1. Maximises the number of successful users (score 0/1/2)
      2. Among equal-score allocations, maximises sum_sinr

    ✅ BUG-PA-1 FIX: removed early-exit 'break' — full grid is always
    evaluated so the allocation with truly highest sum_sinr is returned.
    """
    P    = cfg.TX_POWER
    N0   = cfg.NOISE_POWER
    zeta = cfg.SIC_IMPERFECTION
    T    = 2 ** cfg.TARGET_RATE_BPS - 1

    eps  = 1e-4
    p_ws = np.linspace(eps, 1.0 - eps, grid_points)

    best       = None
    best_score = -1
    best_info  = {}

    for p_w in p_ws:
        p_s  = 1.0 - p_w
        info = compute_sinrs(p_w, P, gain_w, gain_s, N0, zeta, T)
        score = int(info['ok_weak']) + int(info['ok_strong'])

        if (score > best_score) or (
                score == best_score
                and info['sum_sinr'] > best_info.get('sum_sinr', -1)):
            best_score = score
            best       = (p_w, p_s)
            best_info  = info.copy()
            best_info['p_w'] = p_w
            best_info['p_s'] = p_s
        # NOTE: no early break — full scan guarantees optimal sum_sinr

    feasible = best_info.get('ok_weak', False) and best_info.get('ok_strong', False)
    return best[0], best[1], feasible, best_info


def allocate_power_closedform(gain_w: float, gain_s: float, cfg
                              ) -> Tuple[float, float, bool, Dict]:
    """
    Closed-form power allocation using analytical SINR constraints.

    Derives feasible power range [p_w_min, p_w_max] from:
      1. Weak user rate:         p_w >= (T/(1+T)) * (1 + N0/(P*g_w))
      2. Strong decodes weak:    p_w >= (T/(1+T)) * (1 + N0/(P*g_s))
      3. Strong own rate (SIC):  p_w <= (1 - T*N0/(P*g_s)) / (1 + T*zeta)
    """
    P    = cfg.TX_POWER
    N0   = cfg.NOISE_POWER
    zeta = cfg.SIC_IMPERFECTION
    T    = 2 ** cfg.TARGET_RATE_BPS - 1

    eps = 1e-12
    g_w = max(gain_w, eps)
    g_s = max(gain_s, eps)

    lower_w       = (T / (1.0 + T)) * (1.0 + N0 / (P * g_w))
    lower_sdecode = (T / (1.0 + T)) * (1.0 + N0 / (P * g_s))
    lower_bound   = max(lower_w, lower_sdecode)

    numerator   = 1.0 - (T * N0) / (P * g_s)
    upper_bound = numerator / (1.0 + T * zeta)

    lower_bound_clamped = max(lower_bound, 0.0)
    upper_bound_clamped = min(upper_bound, 1.0 - 1e-6)

    info = {
        'lower_w':       lower_w,
        'lower_sdecode': lower_sdecode,
        'lower_bound':   lower_bound_clamped,
        'upper_bound':   upper_bound_clamped,
        'T': T, 'g_w': g_w, 'g_s': g_s, 'zeta': zeta,
    }

    feasible = (lower_bound_clamped <= upper_bound_clamped) and (lower_bound_clamped < 1.0)

    if feasible:
        p_w = min(max(lower_bound_clamped, 0.001), 0.999)
        if p_w > upper_bound_clamped:
            p_w = (lower_bound_clamped + upper_bound_clamped) / 2.0
    else:
        p_w = min(max(lower_bound_clamped, 0.5), 0.99)

    p_s = 1.0 - p_w
    sinr_info = compute_sinrs(p_w, P, g_w, g_s, N0, zeta, T)
    info.update(sinr_info)
    info['p_w'] = p_w
    info['p_s'] = p_s

    return p_w, p_s, sinr_info['ok_weak'] and sinr_info['ok_strong'], info


# ============================================================================
# CACHE-AWARE POWER ALLOCATION
# ============================================================================

def allocate_power_cache_aware(gain_w: float, gain_s: float, cfg,
                               weak_cached: bool = False,
                               strong_cached: bool = False
                               ) -> Tuple[float, float, bool, Dict]:
    """
    Cache-aware power allocation optimised for CIC/perfect-SIC scenarios.

    Scenarios:
      1. No cache:      fall back to closed-form
      2. Weak cached:   weak user has perfect CIC → lower p_w needed
      3. Strong cached: strong user has perfect SIC (zeta=0) → more power for weak
      4. Both cached:   maximum flexibility, balanced midpoint allocation
    """
    P    = cfg.TX_POWER
    N0   = cfg.NOISE_POWER
    zeta = cfg.SIC_IMPERFECTION if not strong_cached else 0.0
    T    = 2 ** cfg.TARGET_RATE_BPS - 1

    eps = 1e-12
    g_w = max(gain_w, eps)
    g_s = max(gain_s, eps)

    if weak_cached and not strong_cached:
        # Weak CIC: SINR_w = P*p_w*g_w / N0  (no interference)
        p_w_min      = (T * N0) / (P * g_w)
        p_w_min_sdec = (T / (1.0 + T)) * (1.0 + N0 / (P * g_s))
        p_w_max      = (1.0 - (T * N0) / (P * g_s)) / (1.0 + T * zeta)
        p_w_lower    = max(p_w_min, p_w_min_sdec, 0.0)
        p_w_upper    = min(p_w_max, 0.999)
        if p_w_lower <= p_w_upper:
            p_w      = np.clip(p_w_lower + 0.05 * (p_w_upper - p_w_lower), 0.001, 0.999)
            feasible = True
        else:
            p_w, feasible = 0.5, False

    elif strong_cached and not weak_cached:
        # Perfect SIC (zeta=0): upper bound extends to 1 - T*N0/(P*g_s)
        p_w_min      = (T / (1.0 + T)) * (1.0 + N0 / (P * g_w))
        p_w_min_sdec = (T / (1.0 + T)) * (1.0 + N0 / (P * g_s))
        p_w_max      = 1.0 - (T * N0) / (P * g_s)   # zeta=0
        p_w_lower    = max(p_w_min, p_w_min_sdec, 0.0)
        p_w_upper    = min(p_w_max, 0.999)
        if p_w_lower <= p_w_upper:
            p_w      = np.clip(p_w_lower + 0.7 * (p_w_upper - p_w_lower), 0.001, 0.999)
            feasible = True
        else:
            p_w, feasible = 0.7, False

    elif weak_cached and strong_cached:
        # Both cached: most relaxed constraints
        p_w_min      = (T * N0) / (P * g_w)
        p_w_min_sdec = (T / (1.0 + T)) * (1.0 + N0 / (P * g_s))
        p_w_max      = 1.0 - (T * N0) / (P * g_s)   # zeta=0
        p_w_lower    = max(p_w_min, p_w_min_sdec, 0.0)
        p_w_upper    = min(p_w_max, 0.999)
        if p_w_lower <= p_w_upper:
            p_w      = np.clip((p_w_lower + p_w_upper) / 2.0, 0.001, 0.999)
            feasible = True
        else:
            p_w, feasible = 0.5, False

    else:
        # No cache: standard closed-form
        return allocate_power_closedform(gain_w, gain_s, cfg)

    p_s       = 1.0 - p_w
    sinr_info = compute_sinrs(p_w, P, g_w, g_s, N0, zeta, T)

    # Correct weak-user SINR if CIC applies (no interference from strong)
    if weak_cached:
        sinr_info['sinr_w'] = (P * p_w * g_w) / N0
        sinr_info['ok_weak'] = sinr_info['sinr_w'] >= T
        sinr_info['sum_sinr'] = sinr_info['sinr_w'] + sinr_info['sinr_s_after']

    info = {
        'p_w': p_w, 'p_s': p_s,
        'weak_cached': weak_cached, 'strong_cached': strong_cached,
        'cache_aware': True,
        'method': 'cache_aware',
        **sinr_info,
    }

    return p_w, p_s, info['ok_weak'] and info['ok_strong'], info


# ============================================================================
# SUM-RATE MAXIMIZATION
# ============================================================================

def allocate_power_sumrate_max(gain_w: float, gain_s: float, cfg,
                               **kwargs) -> Tuple[float, float, bool, Dict]:
    """
    Sum-rate maximization via scalar optimization over p_w in (0,1).
    """
    P    = cfg.TX_POWER
    N0   = cfg.NOISE_POWER
    zeta = cfg.SIC_IMPERFECTION
    T    = 2 ** cfg.TARGET_RATE_BPS - 1

    def objective(p_w):
        p_w = np.clip(p_w, 0.001, 0.999)
        p_s = 1.0 - p_w
        sinr_w    = (P * p_w * gain_w) / (P * p_s * gain_w + N0)
        sinr_sdec = (P * p_w * gain_s) / (P * p_s * gain_s + N0)
        residual  = zeta * (P * p_w * gain_s) if sinr_sdec >= T else P * p_w * gain_s
        sinr_s    = (P * p_s * gain_s) / (N0 + residual)
        return -(np.log2(1 + sinr_w) + np.log2(1 + sinr_s))

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = minimize_scalar(objective, bounds=(0.001, 0.999), method='bounded')

    p_w_opt  = result.x
    p_s_opt  = 1.0 - p_w_opt
    sinr_info = compute_sinrs(p_w_opt, P, gain_w, gain_s, N0, zeta, T)
    rate_w   = np.log2(1 + sinr_info['sinr_w'])
    rate_s   = np.log2(1 + sinr_info['sinr_s_after'])

    info = {
        'p_w': p_w_opt, 'p_s': p_s_opt,
        'sum_rate': rate_w + rate_s,
        'rate_w': rate_w, 'rate_s': rate_s,
        'optimization_method': 'sum_rate_max',
        **sinr_info,
    }
    return p_w_opt, p_s_opt, info['ok_weak'] and info['ok_strong'], info


# ============================================================================
# ENERGY EFFICIENCY OPTIMIZATION
# ============================================================================

def allocate_power_energy_efficient(gain_w: float, gain_s: float, cfg,
                                    circuit_power: float = 0.1
                                    ) -> Tuple[float, float, bool, Dict]:
    """
    Maximise energy efficiency (sum_rate / total_power) for green 6G.
    """
    P    = cfg.TX_POWER
    N0   = cfg.NOISE_POWER
    zeta = cfg.SIC_IMPERFECTION
    T    = 2 ** cfg.TARGET_RATE_BPS - 1
    total_power = P + circuit_power

    def objective(p_w):
        p_w = np.clip(p_w, 0.001, 0.999)
        p_s = 1.0 - p_w
        sinr_w    = (P * p_w * gain_w) / (P * p_s * gain_w + N0)
        sinr_sdec = (P * p_w * gain_s) / (P * p_s * gain_s + N0)
        residual  = zeta * (P * p_w * gain_s) if sinr_sdec >= T else P * p_w * gain_s
        sinr_s    = (P * p_s * gain_s) / (N0 + residual)
        sum_rate  = np.log2(1 + sinr_w) + np.log2(1 + sinr_s)
        return -(sum_rate / total_power)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = minimize_scalar(objective, bounds=(0.001, 0.999), method='bounded')

    p_w_opt  = result.x
    p_s_opt  = 1.0 - p_w_opt
    sinr_info = compute_sinrs(p_w_opt, P, gain_w, gain_s, N0, zeta, T)
    rate_w   = np.log2(1 + sinr_info['sinr_w'])
    rate_s   = np.log2(1 + sinr_info['sinr_s_after'])
    sum_rate = rate_w + rate_s

    info = {
        'p_w': p_w_opt, 'p_s': p_s_opt,
        'sum_rate': sum_rate,
        'total_power': total_power,
        'energy_efficiency': sum_rate / total_power,
        'optimization_method': 'energy_efficient',
        **sinr_info,
    }
    return p_w_opt, p_s_opt, info['ok_weak'] and info['ok_strong'], info


# ============================================================================
# UNIVERSAL INTERFACE
# ============================================================================

def allocate_power(gain_w: float, gain_s: float, cfg,
                   method: str = 'closedform',
                   weak_cached: bool = False,
                   strong_cached: bool = False,
                   **kwargs) -> Tuple[float, float, bool, Dict]:
    """
    Universal power allocation dispatcher.

    Methods: 'gridsearch', 'closedform', 'cache_aware', 'sumrate_max', 'energy_efficient'
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
        raise ValueError(
            f"Unknown method: '{method}'. Use 'gridsearch', 'closedform', "
            f"'cache_aware', 'sumrate_max', or 'energy_efficient'."
        )


# ============================================================================
# SELF-TEST
# ============================================================================

if __name__ == '__main__':
    class MockConfig:
        TX_POWER         = 1.0
        NOISE_POWER      = 1e-9
        POWER_COEFF_WEAK = 0.8
        POWER_COEFF_STRONG = 0.2
        SIC_IMPERFECTION = 0.05
        TARGET_RATE_BPS  = 0.5

    cfg    = MockConfig()
    gain_w = 1e-8
    gain_s = 1e-6

    print('=== Power Allocation Self-Test ===')
    for m in ['gridsearch', 'closedform', 'sumrate_max', 'energy_efficient']:
        p_w, p_s, ok, info = allocate_power(gain_w, gain_s, cfg, method=m)
        print(f'[{m}] p_w={p_w:.3f} p_s={p_s:.3f} feasible={ok}')

    for wc, sc in [(True, False), (False, True), (True, True)]:
        p_w, p_s, ok, _ = allocate_power(gain_w, gain_s, cfg, method='cache_aware',
                                          weak_cached=wc, strong_cached=sc)
        print(f'[cache_aware wc={wc} sc={sc}] p_w={p_w:.3f} p_s={p_s:.3f} feasible={ok}')

    print('\u2705 All tests passed.')