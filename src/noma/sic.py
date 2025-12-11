# src/noma/sic.py
"""
Successive Interference Cancellation (SIC) Module for Cache-Aided NOMA

This module implements SIC - the core mechanism enabling NOMA:
- Standard SIC for 2-user pairs
- Cache-Aided SIC (perfect cancellation when content is cached)
- Imperfect SIC modeling (residual interference)
- SIC success probability computation
- Multiple imperfection sources (channel errors, hardware)

SIC Principle:
    In NOMA, users with better channel conditions (strong users) can decode
    and cancel interference from weaker users before decoding their own signal.
    
    Process:
    1. Strong user decodes weak user's signal (treats own as interference)
    2. If successful, subtracts weak signal from received signal (SIC)
    3. Then decodes own signal with reduced/no interference
    
    Imperfections:
    - Channel estimation errors
    - Decoding errors
    - Timing misalignment
    - Hardware impairments
    Result: Residual interference = ζ × (original interference)

Cache-Aided SIC Innovation:
    When a user has the interfering content cached:
    - Can reconstruct and perfectly cancel interference
    - Residual = 0 (perfect cancellation)
    - Significant SINR improvement!

Author: Cache-Aided NOMA Team
Date: December 2025
"""

import numpy as np
from typing import Tuple, Optional
from scipy.special import erfc


# ============================================================================
# STANDARD SIC FUNCTIONS (ENHANCED WITH DOCUMENTATION)
# ============================================================================

def sinr_weak_user(P_tx: float, p_weak: float, gain_w: float, 
                   p_strong: float, noise: float) -> float:
    """
    Compute SINR for weak user when decoding its own signal.
    
    The weak user (farther from base station) has:
    - Weaker channel gain
    - Higher power allocation (to compensate)
    - Decodes FIRST (before strong user)
    - Treats strong user's signal as interference (cannot cancel it)
    
    Formula:
        SINR_w = (P * p_w * g_w) / (P * p_s * g_w + N0)
    
    where:
        P = total transmit power
        p_w = power coefficient for weak user (typically 0.7-0.9)
        g_w = weak user's channel gain
        p_s = power coefficient for strong user (1 - p_w)
        N0 = noise power
    
    Args:
        P_tx: Total transmit power (Watts)
        p_weak: Power coefficient allocated to weak user [0, 1]
        gain_w: Weak user's channel power gain
        p_strong: Power coefficient allocated to strong user [0, 1]
        noise: Noise power (Watts)
    
    Returns:
        float: SINR for weak user (linear scale, not dB)
    
    Example:
        >>> sinr = sinr_weak_user(P_tx=1.0, p_weak=0.8, gain_w=1e-8, 
        ...                       p_strong=0.2, noise=1e-9)
        >>> print(f"Weak user SINR: {sinr:.3f}")
    """
    numerator = P_tx * p_weak * gain_w
    denominator = P_tx * p_strong * gain_w + noise
    return numerator / denominator


def sinr_strong_decode_weak(P_tx: float, p_weak: float, gain_s: float, 
                            p_strong: float, noise: float) -> float:
    """
    Compute SINR at strong user when decoding weak user's signal (before SIC).
    
    The strong user must FIRST decode the weak user's signal to enable SIC:
    - Strong user has better channel gain (closer to BS)
    - Treats its own signal as interference when decoding weak signal
    - If decoding succeeds → can perform SIC
    - If decoding fails → SIC fails, full interference remains
    
    Formula:
        SINR_s_decode_w = (P * p_w * g_s) / (P * p_s * g_s + N0)
    
    Note: Same interference pattern as weak user, but with better gain g_s
    
    Args:
        P_tx: Total transmit power
        p_weak: Power coefficient for weak user
        gain_s: Strong user's channel power gain (> gain_w)
        p_strong: Power coefficient for strong user
        noise: Noise power
    
    Returns:
        float: SINR for strong user decoding weak signal
    
    Example:
        >>> # Strong user has much better channel
        >>> sinr = sinr_strong_decode_weak(P_tx=1.0, p_weak=0.8, gain_s=1e-6,
        ...                                p_strong=0.2, noise=1e-9)
        >>> threshold = 0.414  # For 0.5 bps/Hz
        >>> can_decode = sinr >= threshold
        >>> print(f"Can decode weak signal: {can_decode}")
    """
    numerator = P_tx * p_weak * gain_s
    denominator = P_tx * p_strong * gain_s + noise
    return numerator / denominator


def sinr_strong_after_sic(P_tx: float, p_strong: float, gain_s: float, 
                          noise: float, residual_from_weak: float = 0.0) -> float:
    """
    Compute SINR at strong user for decoding its own signal after SIC.
    
    After successfully decoding weak user's signal, strong user:
    1. Reconstructs weak user's signal
    2. Subtracts it from received signal (SIC)
    3. Decodes own signal with reduced interference
    
    SIC Quality:
    - Perfect SIC: residual = 0 (ideal case)
    - Imperfect SIC: residual = ζ × P × p_w × g_s (realistic)
    - Failed SIC: residual = P × p_w × g_s (full interference)
    
    where ζ (zeta) is the SIC imperfection factor (typically 0.01-0.1)
    
    Formula:
        SINR_s = (P * p_s * g_s) / (N0 + residual)
    
    Args:
        P_tx: Total transmit power
        p_strong: Power coefficient for strong user
        gain_s: Strong user's channel power gain
        noise: Noise power
        residual_from_weak: Residual interference after SIC (default 0.0)
                           - 0.0 = perfect SIC
                           - > 0 = imperfect SIC
    
    Returns:
        float: SINR for strong user after SIC
    
    Example:
        >>> # Perfect SIC
        >>> sinr_perfect = sinr_strong_after_sic(1.0, 0.2, 1e-6, 1e-9, residual=0)
        >>> 
        >>> # Imperfect SIC (ζ = 0.05)
        >>> residual = 0.05 * (1.0 * 0.8 * 1e-6)
        >>> sinr_imperfect = sinr_strong_after_sic(1.0, 0.2, 1e-6, 1e-9, residual)
        >>> print(f"SINR loss: {(sinr_perfect - sinr_imperfect)/sinr_perfect:.1%}")
    """
    numerator = P_tx * p_strong * gain_s
    denominator = noise + residual_from_weak
    return numerator / denominator


# ============================================================================
# CACHE-AWARE SIC FUNCTIONS (NEW - CRITICAL FOR YOUR PROJECT)
# ============================================================================

def sinr_weak_user_with_cache(P_tx: float, p_weak: float, gain_w: float, 
                               noise: float) -> float:
    """
    Compute SINR for weak user when it has cached content enabling perfect CIC.
    
    Cache-Aided Interference Cancellation (CIC):
        When weak user has strong user's content cached:
        - Can reconstruct strong user's transmitted signal perfectly
        - Subtract it from received signal (perfect cancellation)
        - NO interference from strong user!
    
    Formula:
        SINR_w_cached = (P * p_w * g_w) / N0
    
    Compare to standard:
        SINR_w_standard = (P * p_w * g_w) / (P * p_s * g_w + N0)
    
    Improvement factor:
        SINR_w_cached / SINR_w_standard = 1 + (P * p_s * g_w / N0)
        Typically 2x - 10x improvement!
    
    Args:
        P_tx: Total transmit power
        p_weak: Power coefficient for weak user
        gain_w: Weak user's channel gain
        noise: Noise power
    
    Returns:
        float: SINR for weak user with cache-aided cancellation
    
    Example:
        >>> # Standard NOMA
        >>> sinr_std = sinr_weak_user(1.0, 0.8, 1e-8, 0.2, 1e-9)
        >>> 
        >>> # With cache-aided cancellation
        >>> sinr_cic = sinr_weak_user_with_cache(1.0, 0.8, 1e-8, 1e-9)
        >>> 
        >>> improvement = sinr_cic / sinr_std
        >>> print(f"CIC improvement: {improvement:.1f}x")
    """
    numerator = P_tx * p_weak * gain_w
    denominator = noise  # No interference!
    return numerator / denominator


def sinr_strong_after_perfect_sic(P_tx: float, p_strong: float, 
                                   gain_s: float, noise: float) -> float:
    """
    Compute SINR for strong user with perfect SIC (cache-aided).
    
    When strong user has weak user's content cached:
    - Perfect reconstruction of weak signal
    - Perfect cancellation (residual = 0)
    - Same as sinr_strong_after_sic with residual=0, but explicit for clarity
    
    Formula:
        SINR_s_perfect = (P * p_s * g_s) / N0
    
    Args:
        P_tx: Total transmit power
        p_strong: Power coefficient for strong user
        gain_s: Strong user's channel gain
        noise: Noise power
    
    Returns:
        float: SINR for strong user with perfect SIC
    
    Example:
        >>> # Imperfect SIC (ζ = 0.05)
        >>> residual = 0.05 * (1.0 * 0.8 * 1e-6)
        >>> sinr_imperfect = sinr_strong_after_sic(1.0, 0.2, 1e-6, 1e-9, residual)
        >>> 
        >>> # Perfect SIC (cache-aided)
        >>> sinr_perfect = sinr_strong_after_perfect_sic(1.0, 0.2, 1e-6, 1e-9)
        >>> 
        >>> print(f"Perfect vs Imperfect: {sinr_perfect/sinr_imperfect:.2f}x")
    """
    return sinr_strong_after_sic(P_tx, p_strong, gain_s, noise, residual_from_weak=0.0)


# ============================================================================
# RESIDUAL INTERFERENCE COMPUTATION
# ============================================================================

def compute_residual_interference(P_tx: float, p_weak: float, gain_s: float,
                                 imperfection_factor: float = 0.05,
                                 sic_success: bool = True,
                                 cached: bool = False) -> float:
    """
    Compute residual interference after SIC based on different scenarios.
    
    Three scenarios:
    1. Cache-aided SIC: residual = 0 (perfect cancellation)
    2. Successful SIC: residual = ζ × (weak signal power)
    3. Failed SIC: residual = full weak signal power
    
    Args:
        P_tx: Total transmit power
        p_weak: Power coefficient for weak user
        gain_s: Strong user's channel gain
        imperfection_factor: SIC imperfection ζ (0 to 1, typically 0.01-0.1)
        sic_success: Whether strong user successfully decoded weak signal
        cached: Whether strong user has cached content (enables perfect SIC)
    
    Returns:
        float: Residual interference power after SIC
    
    Example:
        >>> # Perfect SIC (cached)
        >>> residual1 = compute_residual_interference(1.0, 0.8, 1e-6, 
        ...                                           cached=True)
        >>> print(f"Cached: {residual1}")  # 0
        >>> 
        >>> # Imperfect SIC
        >>> residual2 = compute_residual_interference(1.0, 0.8, 1e-6, 
        ...                                           imperfection_factor=0.05)
        >>> 
        >>> # Failed SIC
        >>> residual3 = compute_residual_interference(1.0, 0.8, 1e-6, 
        ...                                           sic_success=False)
        >>> print(f"Failed SIC has {residual3/residual2:.0f}x more interference")
    """
    weak_signal_power = P_tx * p_weak * gain_s
    
    if cached:
        # Cache-aided: perfect cancellation
        return 0.0
    elif sic_success:
        # Imperfect SIC: residual interference
        return imperfection_factor * weak_signal_power
    else:
        # Failed SIC: full interference remains
        return weak_signal_power


# ============================================================================
# SIC SUCCESS PROBABILITY
# ============================================================================

def sic_success_probability(sinr_decode_weak: float, target_sinr: float) -> float:
    """
    Compute probability that SIC succeeds based on SINR.
    
    SIC succeeds if strong user can correctly decode weak signal.
    Simplified model: Success if SINR >= threshold, with smooth transition.
    
    Args:
        sinr_decode_weak: SINR when strong user decodes weak signal
        target_sinr: Required SINR threshold for successful decoding
    
    Returns:
        float: Probability of SIC success (0 to 1)
    
    Note:
        In reality, success probability depends on:
        - Modulation/coding scheme
        - Channel conditions
        - Receiver quality
        This is a simplified model for simulation.
    """
    if sinr_decode_weak >= target_sinr:
        return 1.0
    else:
        # Smooth transition (avoid hard threshold)
        ratio = sinr_decode_weak / target_sinr
        return min(ratio ** 2, 1.0)  # Quadratic falloff


def ber_from_sinr(sinr: float, modulation: str = 'QPSK') -> float:
    """
    Estimate Bit Error Rate (BER) from SINR for different modulations.
    
    Args:
        sinr: Signal-to-Interference-plus-Noise Ratio (linear)
        modulation: 'BPSK' or 'QPSK'
    
    Returns:
        float: Bit error rate (0 to 0.5)
    """
    if modulation == 'BPSK':
        return 0.5 * erfc(np.sqrt(sinr))
    elif modulation == 'QPSK':
        return 0.5 * erfc(np.sqrt(sinr / 2.0))
    else:
        raise ValueError(f"Unknown modulation: {modulation}")


# ============================================================================
# MULTI-SOURCE IMPERFECTION MODELING (ADVANCED)
# ============================================================================

def compute_imperfect_sic_residual(P_tx: float, p_weak: float, gain_s: float,
                                   channel_error: float = 0.01,
                                   timing_error: float = 0.01,
                                   hardware_evm: float = 0.02) -> float:
    """
    Compute residual interference considering multiple imperfection sources.
    
    SIC imperfections arise from:
    1. Channel estimation errors (CSI mismatch)
    2. Timing/synchronization errors
    3. Hardware impairments (EVM, phase noise)
    4. Quantization noise (ADC/DAC)
    
    Total imperfection: ζ_total ≈ ζ_channel + ζ_timing + ζ_hardware
    
    Args:
        P_tx: Total transmit power
        p_weak: Power coefficient for weak user
        gain_s: Strong user's channel gain
        channel_error: Channel estimation error factor (0 to 1)
        timing_error: Timing misalignment error factor (0 to 1)
        hardware_evm: Hardware EVM (Error Vector Magnitude) factor (0 to 1)
    
    Returns:
        float: Residual interference power considering all imperfections
    
    Example:
        >>> # Perfect conditions
        >>> residual1 = compute_imperfect_sic_residual(1.0, 0.8, 1e-6, 0, 0, 0)
        >>> 
        >>> # Realistic imperfections
        >>> residual2 = compute_imperfect_sic_residual(1.0, 0.8, 1e-6,
        ...     channel_error=0.01, timing_error=0.01, hardware_evm=0.02)
        >>> print(f"Total imperfection: {residual2/(1.0*0.8*1e-6):.1%}")
    """
    weak_signal_power = P_tx * p_weak * gain_s
    
    # Combine imperfection sources (additive model)
    total_imperfection = channel_error + timing_error + hardware_evm
    total_imperfection = min(total_imperfection, 1.0)  # Cap at 100%
    
    return total_imperfection * weak_signal_power


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def sinr_to_db(sinr_linear: float) -> float:
    """Convert SINR from linear to dB scale."""
    return 10 * np.log10(sinr_linear) if sinr_linear > 0 else -np.inf


def db_to_sinr(sinr_db: float) -> float:
    """Convert SINR from dB to linear scale."""
    return 10 ** (sinr_db / 10.0)


def rate_from_sinr(sinr: float) -> float:
    """Compute achievable rate from SINR using Shannon formula."""
    return np.log2(1 + sinr)


# ============================================================================
# COMPREHENSIVE SIC SIMULATION WRAPPER
# ============================================================================

def simulate_sic_process(P_tx: float, p_weak: float, p_strong: float,
                        gain_w: float, gain_s: float, noise: float,
                        target_sinr: float, imperfection_factor: float = 0.05,
                        weak_cached: bool = False, 
                        strong_cached: bool = False) -> dict:
    """
    Complete SIC process simulation with all scenarios.
    
    This function orchestrates the complete SIC workflow:
    1. Compute weak user SINR (with/without cache)
    2. Check if strong can decode weak
    3. Compute residual interference
    4. Compute strong user SINR after SIC
    5. Return comprehensive results
    
    Args:
        P_tx: Total transmit power
        p_weak: Power coefficient for weak user
        p_strong: Power coefficient for strong user
        gain_w: Weak user's channel gain
        gain_s: Strong user's channel gain
        noise: Noise power
        target_sinr: SINR threshold for success
        imperfection_factor: SIC imperfection ζ
        weak_cached: Whether weak user has cache (enables CIC)
        strong_cached: Whether strong user has cache (perfect SIC)
    
    Returns:
        dict: Complete SIC results including all SINR values and success flags
    
    Example:
        >>> results = simulate_sic_process(
        ...     P_tx=1.0, p_weak=0.8, p_strong=0.2,
        ...     gain_w=1e-8, gain_s=1e-6, noise=1e-9,
        ...     target_sinr=0.414, weak_cached=True
        ... )
        >>> print(f"Weak success: {results['weak_success']}")
        >>> print(f"Strong success: {results['strong_success']}")
        >>> print(f"CIC benefit: {results['cic_applied']}")
    """
    # Step 1: Weak user SINR
    if weak_cached:
        sinr_w = sinr_weak_user_with_cache(P_tx, p_weak, gain_w, noise)
        cic_applied = True
    else:
        sinr_w = sinr_weak_user(P_tx, p_weak, gain_w, p_strong, noise)
        cic_applied = False
    
    weak_success = sinr_w >= target_sinr
    
    # Step 2: Strong user decodes weak signal
    sinr_s_decode_w = sinr_strong_decode_weak(P_tx, p_weak, gain_s, p_strong, noise)
    can_decode_weak = sinr_s_decode_w >= target_sinr
    
    # Step 3: Compute residual interference
    residual = compute_residual_interference(
        P_tx, p_weak, gain_s,
        imperfection_factor=imperfection_factor,
        sic_success=can_decode_weak,
        cached=strong_cached
    )
    
    # Step 4: Strong user SINR after SIC
    sinr_s_after = sinr_strong_after_sic(P_tx, p_strong, gain_s, noise, residual)
    strong_success = sinr_s_after >= target_sinr
    
    # Compute rates
    rate_w = rate_from_sinr(sinr_w)
    rate_s = rate_from_sinr(sinr_s_after)
    
    return {
        'sinr_w': sinr_w,
        'sinr_w_db': sinr_to_db(sinr_w),
        'sinr_s_decode_w': sinr_s_decode_w,
        'sinr_s_after': sinr_s_after,
        'sinr_s_after_db': sinr_to_db(sinr_s_after),
        'weak_success': weak_success,
        'strong_success': strong_success,
        'can_decode_weak': can_decode_weak,
        'residual_interference': residual,
        'rate_w': rate_w,
        'rate_s': rate_s,
        'sum_rate': rate_w + rate_s,
        'weak_cached': weak_cached,
        'strong_cached': strong_cached,
        'cic_applied': cic_applied or strong_cached,
        'sic_success_prob': sic_success_probability(sinr_s_decode_w, target_sinr)
    }


# ============================================================================
# TESTING AND EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("TESTING ENHANCED SIC MODULE")
    print("="*70)
    
    # Test parameters
    P_tx = 1.0
    p_weak = 0.8
    p_strong = 0.2
    gain_w = 1e-8
    gain_s = 1e-6
    noise = 1e-9
    target_sinr = 0.414  # For 0.5 bps/Hz
    
    # Test 1: Standard SIC
    print("\n[Test 1] Standard SIC (no cache)...")
    results_std = simulate_sic_process(
        P_tx, p_weak, p_strong, gain_w, gain_s, noise, target_sinr
    )
    print(f"Weak SINR: {results_std['sinr_w']:.3f} ({results_std['sinr_w_db']:.1f} dB)")
    print(f"Strong SINR: {results_std['sinr_s_after']:.3f} ({results_std['sinr_s_after_db']:.1f} dB)")
    print(f"Sum rate: {results_std['sum_rate']:.3f} bps/Hz")
    
    # Test 2: Weak user with cache
    print("\n[Test 2] Weak user with cache (CIC)...")
    results_weak_cache = simulate_sic_process(
        P_tx, p_weak, p_strong, gain_w, gain_s, noise, target_sinr,
        weak_cached=True
    )
    improvement = results_weak_cache['sinr_w'] / results_std['sinr_w']
    print(f"Weak SINR: {results_weak_cache['sinr_w']:.3f} ({results_weak_cache['sinr_w_db']:.1f} dB)")
    print(f"CIC improvement: {improvement:.1f}x")
    print(f"Sum rate: {results_weak_cache['sum_rate']:.3f} bps/Hz")
    
    # Test 3: Strong user with cache
    print("\n[Test 3] Strong user with cache (Perfect SIC)...")
    results_strong_cache = simulate_sic_process(
        P_tx, p_weak, p_strong, gain_w, gain_s, noise, target_sinr,
        strong_cached=True
    )
    print(f"Strong SINR: {results_strong_cache['sinr_s_after']:.3f} ({results_strong_cache['sinr_s_after_db']:.1f} dB)")
    print(f"Residual interference: {results_strong_cache['residual_interference']:.2e} (zero!)")
    print(f"Sum rate: {results_strong_cache['sum_rate']:.3f} bps/Hz")
    
    # Test 4: Both cached
    print("\n[Test 4] Both users cached...")
    results_both = simulate_sic_process(
        P_tx, p_weak, p_strong, gain_w, gain_s, noise, target_sinr,
        weak_cached=True, strong_cached=True
    )
    print(f"Weak SINR: {results_both['sinr_w']:.3f}")
    print(f"Strong SINR: {results_both['sinr_s_after']:.3f}")
    print(f"Sum rate: {results_both['sum_rate']:.3f} bps/Hz")
    print(f"Improvement over standard: {results_both['sum_rate']/results_std['sum_rate']:.1f}x")
    
    print("\n" + "="*70)
    print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*70)