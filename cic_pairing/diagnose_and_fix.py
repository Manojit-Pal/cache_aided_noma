#!/usr/bin/env python3
"""
Diagnostic Script: Cache-Aware Power Allocation Issues

This script diagnoses why cache-aware power allocation might be failing
and provides recommended fixes.

Common Issues:
1. Poor channel conditions
2. Target rate too high
3. Noise power too high
4. TX power too low

Author: Cache-Aided NOMA Team
Date: December 2025
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.noma import (
    generate_user_positions,
    compute_channel_gains,
    simulate_noma_system
)
import src.config as cfg


def diagnose_system():
    """
    Diagnose why cache-aware power allocation is failing.
    """
    print("\n" + "═" * 80)
    print("🔍 CACHE-AWARE POWER ALLOCATION DIAGNOSTIC".center(80))
    print("═" * 80)
    
    # Generate test scenario
    print("\n📡 Generating test scenario...")
    num_users = 20
    positions = generate_user_positions(num_users, cfg.CELL_RADIUS, seed=42)
    gains = compute_channel_gains(positions, cfg.PATH_LOSS_EXPONENT)
    
    # Analyze channel conditions
    print("\n" + "─" * 80)
    print("📊 CHANNEL CONDITION ANALYSIS")
    print("─" * 80)
    
    min_gain = np.min(gains)
    max_gain = np.max(gains)
    avg_gain = np.mean(gains)
    
    print(f"Channel Gains:")
    print(f"  Min: {min_gain:.3e}")
    print(f"  Max: {max_gain:.3e}")
    print(f"  Avg: {avg_gain:.3e}")
    print(f"  Ratio (max/min): {max_gain/min_gain:.1f}x")
    
    # Check SNR range
    snr_min = (cfg.TX_POWER * min_gain) / cfg.NOISE_POWER
    snr_max = (cfg.TX_POWER * max_gain) / cfg.NOISE_POWER
    snr_min_db = 10 * np.log10(snr_min)
    snr_max_db = 10 * np.log10(snr_max)
    
    print(f"\nSNR Range (P*g/N0):")
    print(f"  Worst user: {snr_min:.3e} ({snr_min_db:.1f} dB)")
    print(f"  Best user:  {snr_max:.3e} ({snr_max_db:.1f} dB)")
    
    # Check if target rate is achievable
    print("\n" + "─" * 80)
    print("🎯 TARGET RATE FEASIBILITY")
    print("─" * 80)
    
    target_sinr = 2 ** cfg.TARGET_RATE_BPS - 1
    print(f"Target Rate: {cfg.TARGET_RATE_BPS} bps/Hz")
    print(f"Required SINR: {target_sinr:.3f} (linear)")
    print(f"Required SNR (no interference): {10*np.log10(target_sinr):.1f} dB")
    
    # Check achievability
    worst_achievable = snr_min >= target_sinr
    print(f"\nWorst user can achieve target (alone): {worst_achievable}")
    
    if not worst_achievable:
        print(f"  ⚠️ WARNING: Worst user SNR ({snr_min_db:.1f} dB) < Required ({10*np.log10(target_sinr):.1f} dB)")
        print(f"  This means even with perfect power allocation, weak users will fail!")
    
    # Test power allocation
    print("\n" + "─" * 80)
    print("⚡ POWER ALLOCATION TEST")
    print("─" * 80)
    
    from src.noma.power_allocation import allocate_power_cache_aware
    
    # Test on worst pair (extreme pairing)
    sorted_indices = np.argsort(gains)
    weak_idx = sorted_indices[0]  # Weakest
    strong_idx = sorted_indices[-1]  # Strongest
    
    print(f"\nTesting worst case pair:")
    print(f"  Weak user {weak_idx}: gain = {gains[weak_idx]:.3e}")
    print(f"  Strong user {strong_idx}: gain = {gains[strong_idx]:.3e}")
    
    # Try different scenarios
    scenarios = [
        (False, False, "No cache"),
        (True, False, "Weak cached"),
        (False, True, "Strong cached"),
        (True, True, "Both cached")
    ]
    
    results = []
    for weak_cached, strong_cached, desc in scenarios:
        try:
            p_w, p_s, feasible, info = allocate_power_cache_aware(
                gains[weak_idx], gains[strong_idx], cfg,
                weak_cached, strong_cached
            )
            results.append((desc, feasible, p_w, p_s, info['sum_sinr']))
        except Exception as e:
            results.append((desc, False, None, None, None))
    
    print("\n" + f"{'Scenario':<20} | {'Feasible':<10} | {'p_w':<8} | {'p_s':<8} | {'Sum SINR'}")
    print("─" * 80)
    for desc, feasible, p_w, p_s, sum_sinr in results:
        if feasible:
            print(f"{desc:<20} | {'✅ Yes':<10} | {p_w:>6.3f}   | {p_s:>6.3f}   | {sum_sinr:.3f}")
        else:
            print(f"{desc:<20} | {'❌ No':<10} | {'N/A':<8} | {'N/A':<8} | N/A")
    
    # Recommendations
    print("\n" + "═" * 80)
    print("💡 RECOMMENDATIONS".center(80))
    print("═" * 80)
    
    recommendations = []
    
    if not worst_achievable:
        recommendations.append((
            "CRITICAL",
            "Target rate too high for channel conditions",
            f"Reduce TARGET_RATE_BPS from {cfg.TARGET_RATE_BPS} to 0.3 or lower"
        ))
    
    if snr_min_db < 10:
        recommendations.append((
            "HIGH",
            "Very poor SNR at cell edge",
            f"Increase TX_POWER from {cfg.TX_POWER}W to 2.0W or 5.0W"
        ))
    
    if any(not r[1] for r in results):
        recommendations.append((
            "MEDIUM",
            "Some cache scenarios infeasible",
            "Adjust SIC_IMPERFECTION or increase TX_POWER"
        ))
    
    if not recommendations:
        print("\n✅ System looks healthy! Cache-aware power allocation should work.")
    else:
        for priority, issue, fix in recommendations:
            print(f"\n[{priority}] {issue}")
            print(f"  → Fix: {fix}")
    
    print("\n" + "═" * 80)
    
    return results


def run_with_recommended_settings():
    """
    Run simulation with recommended settings.
    """
    print("\n" + "═" * 80)
    print("🔧 RUNNING WITH RECOMMENDED SETTINGS".center(80))
    print("═" * 80)
    
    # Temporarily adjust settings
    original_rate = cfg.TARGET_RATE_BPS
    original_power = cfg.TX_POWER
    
    # Recommended: Lower target rate, higher TX power
    cfg.TARGET_RATE_BPS = 0.3  # Lower target (more achievable)
    cfg.TX_POWER = 2.0         # Higher TX power (better SNR)
    
    print(f"\nAdjusted Parameters:")
    print(f"  TARGET_RATE_BPS: {original_rate} → {cfg.TARGET_RATE_BPS}")
    print(f"  TX_POWER: {original_power}W → {cfg.TX_POWER}W")
    
    # Run test
    num_users = 20
    positions = generate_user_positions(num_users, cfg.CELL_RADIUS, seed=42)
    gains = compute_channel_gains(positions, cfg.PATH_LOSS_EXPONENT)
    cache_status = {i: (i % 3 == 0) for i in range(num_users)}
    
    print(f"\n🚀 Running simulation with {num_users} users...")
    results = simulate_noma_system(
        gains, cfg,
        cache_status=cache_status,
        optimize_power=True
    )
    
    metrics = results['system_metrics']
    
    print("\n" + "─" * 80)
    print("📊 RESULTS WITH RECOMMENDED SETTINGS")
    print("─" * 80)
    print(f"Success Rate: {metrics['overall_success_rate']*100:.1f}%")
    print(f"Outage Prob:  {metrics['outage_probability']*100:.1f}%")
    print(f"Avg Sum Rate: {metrics['average_sum_rate']:.3f} bps/Hz")
    print(f"\nCache-Aware Power Used: {metrics.get('cache_aware_power_count', 0)}/{metrics['num_pairs']} pairs")
    print(f"CIC Applied: {metrics['weak_cic_count']} weak, {metrics['strong_cic_count']} strong, {metrics['both_cic_count']} both")
    
    # Check if it worked
    if metrics.get('cache_aware_power_count', 0) > 0:
        print("\n✅ SUCCESS! Cache-aware power allocation is now working!")
    else:
        print("\n⚠️ Still having issues. May need further adjustment.")
    
    # Restore original settings
    cfg.TARGET_RATE_BPS = original_rate
    cfg.TX_POWER = original_power
    
    print("\n" + "═" * 80)
    print("ℹ️  Note: Settings have been restored to original values.")
    print("   To make these changes permanent, edit src/config.py")
    print("═" * 80 + "\n")
    
    return results


if __name__ == "__main__":
    # Run diagnostics
    diagnose_system()
    
    # Offer to run with recommended settings
    print("\n" + "─" * 80)
    response = input("\nWould you like to run simulation with recommended settings? (y/n): ")
    
    if response.lower() == 'y':
        run_with_recommended_settings()
    else:
        print("\n💡 To fix manually, edit src/config.py:")
        print("   TARGET_RATE_BPS = 0.3  # Lower from 0.5")
        print("   TX_POWER = 2.0         # Increase from 1.0")
        print()
