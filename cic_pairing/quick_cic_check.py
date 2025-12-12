#!/usr/bin/env python3
"""
Quick CIC Pairing Check - Minimal Script for Fast Debugging

Use this script to quickly verify CIC pairing in your simulations.
Perfect for debugging and sanity checks.

Usage:
    python cic_pairing/quick_cic_check.py

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


def quick_cic_check():
    """
    Quick check to verify CIC is working correctly.
    """
    print("┌" + "─" * 78 + "┐")
    print("│" + " QUICK CIC PAIRING CHECK ".center(78) + "│")
    print("└" + "─" * 78 + "┘\n")
    
    # Minimal setup
    num_users = 10
    print(f"➤ Creating {num_users} users...")
    positions = generate_user_positions(num_users, cfg.CELL_RADIUS, seed=42)
    gains = compute_channel_gains(positions, cfg.PATH_LOSS_EXPONENT)
    
    # Cache first 3 users (30%)
    cache_status = {i: (i < 3) for i in range(num_users)}
    cached = [i for i, c in cache_status.items() if c]
    print(f"➤ Cached users: {cached}\n")
    
    # Run simulation
    print("➤ Running simulation...\n")
    results = simulate_noma_system(
        gains, cfg,
        cache_status=cache_status,
        optimize_power=True
    )
    
    # Quick summary
    print("┌" + "─" * 78 + "┐")
    print("│" + " RESULTS ".center(78) + "│")
    print("├" + "─" * 78 + "┤")
    
    for i, pr in enumerate(results['pair_results']):
        cic = pr.get('cic_users', [])
        
        if len(cic) == 0:
            cic_str = "No CIC"
            symbol = "  "
        elif len(cic) == 2:
            cic_str = "Both CIC "
            symbol = "🔄🔄"
        elif 'weak' in cic:
            cic_str = "Weak CIC "
            symbol = "🔄←"
        else:
            cic_str = "Strong CIC"
            symbol = "→🔄"
        
        weak_cache = "✅" if pr['weak_cached'] else "❌"
        strong_cache = "✅" if pr['strong_cached'] else "❌"
        success = "✅" if pr['pair_success'] else "❌"
        
        print(f"│ Pair {i+1}: {weak_cache} U{pr['weak_idx']:02d} {symbol} {strong_cache} U{pr['strong_idx']:02d} "
              f"| {cic_str} | Rate: {pr['sum_rate']:.3f} {success}" + " "*(78-64) + "│")
    
    print("├" + "─" * 78 + "┤")
    
    # Statistics
    metrics = results['system_metrics']
    print(f"│ Total: {metrics['num_pairs']} pairs | "
          f"Weak CIC: {metrics['weak_cic_count']} | "
          f"Strong CIC: {metrics['strong_cic_count']} | "
          f"Both: {metrics['both_cic_count']}" + " "*17 + "│")
    print(f"│ Success Rate: {metrics['overall_success_rate']*100:.1f}% | "
          f"Avg Sum Rate: {metrics['average_sum_rate']:.3f} bps/Hz" + " "*27 + "│")
    
    if 'cache_aware_power_count' in metrics:
        print(f"│ Cache-Aware Power: {metrics['cache_aware_power_count']}/{metrics['num_pairs']} pairs" + " "*49 + "│")
    
    print("└" + "─" * 78 + "┘\n")
    
    # Verification
    print("✅ VERIFICATION:")
    
    # Check 1: CIC applied when expected
    cic_count = metrics['weak_cic_count'] + metrics['strong_cic_count']
    if cic_count > 0:
        print(f"  ✓ CIC detected: {cic_count} user instances benefit from CIC")
    else:
        print(f"  ⚠ WARNING: No CIC detected despite {len(cached)} cached users!")
    
    # Check 2: Cache-aware power
    if metrics.get('cache_aware_power_count', 0) > 0:
        print(f"  ✓ Cache-aware power allocation: WORKING")
    else:
        print(f"  ⚠ Cache-aware power not used (check cfg.POWER_ALLOC_METHOD)")
    
    # Check 3: Performance improvement
    if cic_count > 0 and metrics['average_sum_rate'] > 0.5:
        print(f"  ✓ System performance: GOOD ({metrics['average_sum_rate']:.3f} bps/Hz)")
    
    print("\n✨ Quick check complete!\n")
    
    return results


if __name__ == "__main__":
    results = quick_cic_check()
