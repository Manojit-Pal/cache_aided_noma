#!/usr/bin/env python3
"""
Cache-Aided Interference Cancellation (CIC) Pairing Analysis Example

This script demonstrates:
1. How to check which user pairs benefit from CIC
2. How to visualize CIC pairing patterns
3. How to compare standard NOMA vs cache-aided NOMA
4. How to export results for further analysis
5. Performance gains from cache-aware power allocation

Author: Cache-Aided NOMA Team
Date: December 2025
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.noma import (
    generate_user_positions,
    compute_channel_gains,
    simulate_noma_system
)
import src.config as cfg


# ============================================================================
# UTILITY FUNCTIONS FOR CIC ANALYSIS
# ============================================================================

def analyze_cic_pairing(results):
    """
    Analyze CIC pairing patterns in simulation results.
    
    Args:
        results: Output from simulate_noma_system()
    
    Returns:
        Dictionary with detailed CIC statistics
    """
    pair_results = results['pair_results']
    
    # Categorize pairs by CIC type
    no_cic = []
    weak_cic = []
    strong_cic = []
    both_cic = []
    
    for pr in pair_results:
        weak_id = pr['weak_idx']
        strong_id = pr['strong_idx']
        cic_users = pr.get('cic_users', [])
        
        pair_info = {
            'pair': (weak_id, strong_id),
            'weak_cached': pr['weak_cached'],
            'strong_cached': pr['strong_cached'],
            'sum_rate': pr['sum_rate'],
            'weak_success': pr['weak_success'],
            'strong_success': pr['strong_success'],
            'p_w': pr['p_w'],
            'p_s': pr['p_s']
        }
        
        if len(cic_users) == 0:
            no_cic.append(pair_info)
        elif len(cic_users) == 2:
            both_cic.append(pair_info)
        elif 'weak' in cic_users:
            weak_cic.append(pair_info)
        elif 'strong' in cic_users:
            strong_cic.append(pair_info)
    
    return {
        'no_cic': no_cic,
        'weak_cic': weak_cic,
        'strong_cic': strong_cic,
        'both_cic': both_cic,
        'total_pairs': len(pair_results)
    }


def print_cic_visualization(results):
    """
    Print a visual representation of CIC pairing.
    
    Legend:
    ✅ = User has cached content
    ❌ = User does not have cached content
    🔄 = CIC applied
    """
    pairs = results['pairs']
    pair_results = results['pair_results']
    
    print("\n" + "="*80)
    print("CIC PAIRING VISUALIZATION")
    print("="*80)
    print("\nLegend: ✅ = Cached | ❌ = Not Cached | 🔄 = CIC Applied")
    print("-"*80)
    print(f"{'Pair':>4} | {'Weak User':^20} | {'CIC':^6} | {'Strong User':^20} | {'Sum Rate':>10}")
    print("-"*80)
    
    for idx, (weak_id, strong_id) in enumerate(pairs):
        pair_info = pair_results[idx]
        
        # Format user info
        weak_symbol = "✅" if pair_info['weak_cached'] else "❌"
        strong_symbol = "✅" if pair_info['strong_cached'] else "❌"
        
        weak_str = f"{weak_symbol} User {weak_id:3d}"
        strong_str = f"{strong_symbol} User {strong_id:3d}"
        
        # CIC indicator
        cic_users = pair_info.get('cic_users', [])
        if len(cic_users) == 2:
            cic_indicator = "🔄🔄"  # Both
        elif 'weak' in cic_users:
            cic_indicator = "🔄←"  # Weak benefits
        elif 'strong' in cic_users:
            cic_indicator = "→🔄"  # Strong benefits
        else:
            cic_indicator = "  "  # No CIC
        
        # Success indicator
        success = "✅" if pair_info['pair_success'] else "❌"
        
        print(f"{idx+1:4d} | {weak_str:^20} | {cic_indicator:^6} | {strong_str:^20} | "
              f"{pair_info['sum_rate']:9.3f} {success}")
    
    print("-"*80)


def print_cic_statistics(results):
    """
    Print detailed statistics about CIC usage.
    """
    analysis = analyze_cic_pairing(results)
    metrics = results['system_metrics']
    
    print("\n" + "="*80)
    print("CIC STATISTICS")
    print("="*80)
    
    print(f"\n📊 Total Pairs: {analysis['total_pairs']}")
    print(f"\n🔄 CIC Breakdown:")
    print(f"  • No CIC:          {len(analysis['no_cic']):3d} pairs ({len(analysis['no_cic'])/analysis['total_pairs']*100:5.1f}%)")
    print(f"  • Weak User CIC:   {len(analysis['weak_cic']):3d} pairs ({len(analysis['weak_cic'])/analysis['total_pairs']*100:5.1f}%)")
    print(f"  • Strong User CIC: {len(analysis['strong_cic']):3d} pairs ({len(analysis['strong_cic'])/analysis['total_pairs']*100:5.1f}%)")
    print(f"  • Both Users CIC:  {len(analysis['both_cic']):3d} pairs ({len(analysis['both_cic'])/analysis['total_pairs']*100:5.1f}%)")
    
    # Performance comparison
    print(f"\n📈 Performance Metrics:")
    print(f"  • Overall Success Rate: {metrics['overall_success_rate']*100:.1f}%")
    print(f"  • Average Sum Rate:     {metrics['average_sum_rate']:.3f} bps/Hz")
    print(f"  • Outage Probability:   {metrics['outage_probability']*100:.1f}%")
    print(f"  • Cache Hit Rate:       {metrics['cache_hit_rate']*100:.1f}%")
    
    # Power allocation info
    if 'cache_aware_power_count' in metrics:
        print(f"\n⚡ Power Allocation:")
        print(f"  • Cache-Aware Power Used: {metrics['cache_aware_power_count']}/{metrics['num_pairs']} pairs")
        print(f"  • Optimization Enabled:   {metrics['power_optimization_enabled']}")
    
    # Average rates by CIC type
    print(f"\n📋 Average Sum Rates by CIC Type:")
    if len(analysis['no_cic']) > 0:
        avg_no_cic = np.mean([p['sum_rate'] for p in analysis['no_cic']])
        print(f"  • No CIC:          {avg_no_cic:.3f} bps/Hz")
    
    if len(analysis['weak_cic']) > 0:
        avg_weak = np.mean([p['sum_rate'] for p in analysis['weak_cic']])
        improvement_weak = (avg_weak / avg_no_cic - 1) * 100 if len(analysis['no_cic']) > 0 else 0
        print(f"  • Weak User CIC:   {avg_weak:.3f} bps/Hz (+{improvement_weak:.1f}%)")
    
    if len(analysis['strong_cic']) > 0:
        avg_strong = np.mean([p['sum_rate'] for p in analysis['strong_cic']])
        improvement_strong = (avg_strong / avg_no_cic - 1) * 100 if len(analysis['no_cic']) > 0 else 0
        print(f"  • Strong User CIC: {avg_strong:.3f} bps/Hz (+{improvement_strong:.1f}%)")
    
    if len(analysis['both_cic']) > 0:
        avg_both = np.mean([p['sum_rate'] for p in analysis['both_cic']])
        improvement_both = (avg_both / avg_no_cic - 1) * 100 if len(analysis['no_cic']) > 0 else 0
        print(f"  • Both Users CIC:  {avg_both:.3f} bps/Hz (+{improvement_both:.1f}%)")
    
    print("="*80)


def print_detailed_pair_info(results, show_top_n=5):
    """
    Print detailed information for top performing pairs.
    """
    pair_results = sorted(results['pair_results'], key=lambda x: x['sum_rate'], reverse=True)
    
    print(f"\n" + "="*80)
    print(f"TOP {show_top_n} PERFORMING PAIRS (DETAILED)")
    print("="*80)
    
    for i, pr in enumerate(pair_results[:show_top_n]):
        print(f"\n🏆 Rank {i+1}: User {pr['weak_idx']} (weak) ↔ User {pr['strong_idx']} (strong)")
        print(f"  Cache Status:")
        print(f"    - Weak user cached:   {pr['weak_cached']}")
        print(f"    - Strong user cached: {pr['strong_cached']}")
        print(f"  CIC Applied: {pr.get('cic_users', [])}")
        print(f"  Power Allocation:")
        print(f"    - p_w = {pr['p_w']:.3f}, p_s = {pr['p_s']:.3f}")
        
        # ✅ FIX: Use .get() with default value to avoid KeyError
        power_method = pr.get('power_allocation', {}).get('method', 'unknown')
        print(f"    - Method: {power_method}")
        
        print(f"  Performance:")
        print(f"    - Weak SINR:  {pr['sinr_w']:.3f} (Success: {pr['weak_success']})")
        print(f"    - Strong SINR: {pr['sinr_s_after']:.3f} (Success: {pr['strong_success']})")
        print(f"    - Sum Rate:   {pr['sum_rate']:.3f} bps/Hz")
        print(f"    - Fairness:   {pr['fairness']:.3f}")
    
    print("="*80)


def compare_with_without_cache(num_users=50, cache_ratio=0.3, seed=42):
    """
    Compare performance with and without cache-aided NOMA.
    """
    print("\n" + "="*80)
    print("COMPARISON: STANDARD NOMA vs CACHE-AIDED NOMA")
    print("="*80)
    
    # Generate channel conditions
    positions = generate_user_positions(num_users, cfg.CELL_RADIUS, seed=seed)
    gains = compute_channel_gains(positions, cfg.PATHLOSS_EXPONENT)
    
    # Define cache status (some users have cache)
    num_cached = int(num_users * cache_ratio)
    cache_status = {i: (i < num_cached) for i in range(num_users)}
    
    print(f"\n📡 Simulation Parameters:")
    print(f"  • Number of users: {num_users}")
    print(f"  • Cache ratio:     {cache_ratio*100:.0f}% ({num_cached} users)")
    print(f"  • Cell radius:     {cfg.CELL_RADIUS} m")
    
    # Scenario 1: Standard NOMA (no cache)
    print(f"\n🔵 Scenario 1: Standard NOMA (No Cache)...")
    no_cache_status = {i: False for i in range(num_users)}
    results_no_cache = simulate_noma_system(
        gains, cfg,
        cache_status=no_cache_status,
        optimize_power=False  # Standard power allocation
    )
    
    metrics_no_cache = results_no_cache['system_metrics']
    print(f"  • Success Rate: {metrics_no_cache['overall_success_rate']*100:.1f}%")
    print(f"  • Avg Sum Rate:  {metrics_no_cache['average_sum_rate']:.3f} bps/Hz")
    print(f"  • Outage Prob:   {metrics_no_cache['outage_probability']*100:.1f}%")
    
    # Scenario 2: Cache-aided NOMA (standard power)
    print(f"\n🟠 Scenario 2: Cache-Aided NOMA (Standard Power)...")
    results_cache_std_power = simulate_noma_system(
        gains, cfg,
        cache_status=cache_status,
        optimize_power=False  # CIC but no cache-aware power
    )
    
    metrics_cache_std = results_cache_std_power['system_metrics']
    print(f"  • Success Rate: {metrics_cache_std['overall_success_rate']*100:.1f}%")
    print(f"  • Avg Sum Rate:  {metrics_cache_std['average_sum_rate']:.3f} bps/Hz")
    print(f"  • Outage Prob:   {metrics_cache_std['outage_probability']*100:.1f}%")
    print(f"  • CIC Applied:   {metrics_cache_std['weak_cic_count']} weak, "
          f"{metrics_cache_std['strong_cic_count']} strong, "
          f"{metrics_cache_std['both_cic_count']} both")
    
    # Scenario 3: Cache-aided NOMA (cache-aware power)
    print(f"\n🟢 Scenario 3: Cache-Aided NOMA (Cache-Aware Power)...")
    results_cache_opt = simulate_noma_system(
        gains, cfg,
        cache_status=cache_status,
        optimize_power=True  # CIC + cache-aware power allocation
    )
    
    metrics_cache_opt = results_cache_opt['system_metrics']
    print(f"  • Success Rate: {metrics_cache_opt['overall_success_rate']*100:.1f}%")
    print(f"  • Avg Sum Rate:  {metrics_cache_opt['average_sum_rate']:.3f} bps/Hz")
    print(f"  • Outage Prob:   {metrics_cache_opt['outage_probability']*100:.1f}%")
    print(f"  • CIC Applied:   {metrics_cache_opt['weak_cic_count']} weak, "
          f"{metrics_cache_opt['strong_cic_count']} strong, "
          f"{metrics_cache_opt['both_cic_count']} both")
    
    # Performance gains
    print(f"\n📈 Performance Gains:")
    
    gain_cache = (metrics_cache_std['average_sum_rate'] / metrics_no_cache['average_sum_rate'] - 1) * 100
    print(f"  • Cache-aided NOMA (std power) vs Standard: +{gain_cache:.1f}% sum rate")
    
    gain_opt = (metrics_cache_opt['average_sum_rate'] / metrics_no_cache['average_sum_rate'] - 1) * 100
    print(f"  • Cache-aided NOMA (opt power) vs Standard: +{gain_opt:.1f}% sum rate")
    
    gain_power = (metrics_cache_opt['average_sum_rate'] / metrics_cache_std['average_sum_rate'] - 1) * 100
    print(f"  • Cache-aware power allocation benefit:     +{gain_power:.1f}% sum rate")
    
    outage_reduction = (metrics_no_cache['outage_probability'] - metrics_cache_opt['outage_probability']) * 100
    print(f"  • Outage probability reduction:             -{outage_reduction:.1f}%")
    
    print("="*80)
    
    return results_no_cache, results_cache_std_power, results_cache_opt


# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================

def main():
    """
    Main demonstration of CIC pairing analysis.
    """
    print("\n")
    print("█" * 80)
    print("   CACHE-AIDED INTERFERENCE CANCELLATION (CIC) PAIRING ANALYSIS")
    print("█" * 80)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # ========================================================================
    # PART 1: Basic CIC Pairing Analysis
    # ========================================================================
    print("\n" + "►" * 40)
    print(" PART 1: BASIC CIC PAIRING ANALYSIS")
    print("►" * 40)
    
    # Setup
    num_users = 30
    print(f"\n🔧 Setting up simulation with {num_users} users...")
    
    positions = generate_user_positions(num_users, cfg.CELL_RADIUS, seed=42)
    gains = compute_channel_gains(positions, cfg.PATHLOSS_EXPONENT)
    
    # Cache status: 30% of users have cache (every 3rd user)
    cache_status = {i: (i % 3 == 0) for i in range(num_users)}
    cached_users = [i for i in range(num_users) if cache_status[i]]
    print(f"  ✅ Cached users: {cached_users}")
    
    # Run simulation
    print(f"\n🚀 Running cache-aided NOMA simulation...")
    results = simulate_noma_system(
        gains, cfg,
        cache_status=cache_status,
        pairing_method='extreme',
        optimize_power=True  # Enable cache-aware power allocation
    )
    
    # Visualize pairing
    print_cic_visualization(results)
    
    # Print statistics
    print_cic_statistics(results)
    
    # Show detailed info for top pairs
    print_detailed_pair_info(results, show_top_n=3)
    
    # ========================================================================
    # PART 2: Comparison Analysis
    # ========================================================================
    print("\n\n" + "►" * 40)
    print(" PART 2: PERFORMANCE COMPARISON")
    print("►" * 40)
    
    results_no_cache, results_cache_std, results_cache_opt = compare_with_without_cache(
        num_users=50,
        cache_ratio=0.3,
        seed=42
    )
    
    # ========================================================================
    # PART 3: Export Results (Optional)
    # ========================================================================
    print("\n\n" + "►" * 40)
    print(" PART 3: EXPORT OPTIONS")
    print("►" * 40)
    
    print("\n💾 Export Results (Optional):")
    print("  You can export results using:")
    print("\n  import pandas as pd")
    print("  import json")
    print("\n  # Export to CSV")
    print("  df = pd.DataFrame(results['pair_results'])")
    print("  df.to_csv('cic_pairing_results.csv', index=False)")
    print("\n  # Export to JSON")
    print("  with open('cic_results.json', 'w') as f:")
    print("      json.dump(results['system_metrics'], f, indent=2)")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n\n" + "█" * 80)
    print("   ANALYSIS COMPLETE")
    print("█" * 80)
    
    print("\n✅ Key Takeaways:")
    print("  1. CIC significantly improves NOMA performance")
    print("  2. Cache-aware power allocation provides additional gains")
    print("  3. Both weak and strong users can benefit from CIC")
    print("  4. Pairing strategy affects which users get CIC benefits")
    print("\n💡 Next Steps:")
    print("  - Try different cache ratios to see impact")
    print("  - Experiment with different pairing methods")
    print("  - Analyze cache placement strategies")
    print("  - Compare with your baseline results")
    print("\n")


if __name__ == "__main__":
    main()