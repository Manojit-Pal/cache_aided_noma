#!/usr/bin/env python3
"""
Quick Test Script for NOMA-Aware Caching Simulation

Tests the complete integration:
- NOMA module (SIC/CIC)
- Caching module (all policies)
- Simulation framework

Usage:
    python test_noma_sim.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src import config as cfg
from src.simulation.noma_caching_sim import (
    NOMACachingSimulator,
    run_baseline_comparison
)

print("="*70)
print("TESTING NOMA-AWARE CACHE SIMULATION")
print("="*70)

# Set quick test config
original_runs = cfg.NUM_RUNS
cfg.NUM_RUNS = 3  # Quick test
print(f"\nRunning quick test with {cfg.NUM_RUNS} runs...\n")

try:
    # Test 1: Import all modules
    print("[Test 1] Checking imports...")
    from src.noma import (
        generate_user_positions,
        compute_channel_gains,
        pair_users,
        allocate_power,
        simulate_sic_process
    )
    from src.caching import create_cache
    print("✅ All imports successful\n")
    
    # Test 2: Create simulator
    print("[Test 2] Creating simulator...")
    simulator = NOMACachingSimulator(cfg)
    print("✅ Simulator created\n")
    
    # Test 3: Test single episode with TopK cache
    print("[Test 3] Testing single episode with TopK cache...")
    from src.caching import StaticTopKCache
    cache = StaticTopKCache(capacity=cfg.CACHE_SIZE)
    results = simulator.run_single_episode(cache, seed=42)
    
    print(f"  Hit rate: {results['hit_rate']:.3f}")
    print(f"  Outage prob: {results['outage_probability']:.3f}")
    print(f"  CIC benefit: {results['cic_benefit_rate']:.3f}")
    print(f"  Spectral efficiency: {results['spectral_efficiency']:.3f} bps/Hz")
    print("✅ Single episode test passed\n")
    
    # Test 4: Test all baseline policies
    print("[Test 4] Testing all baseline policies...")
    print("This will take ~30 seconds...\n")
    
    results_df = run_baseline_comparison(cfg, num_runs=3)
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    summary = results_df.groupby('policy')[[
        'hit_rate', 
        'outage_probability', 
        'cic_benefit_rate'
    ]].mean()
    
    print(summary)
    print("\n✅ All baseline policies tested successfully!\n")
    
    # Test 5: Check key metrics
    print("[Test 5] Validating metrics...")
    for policy in results_df['policy'].unique():
        policy_df = results_df[results_df['policy'] == policy]
        hit_rate = policy_df['hit_rate'].mean()
        outage = policy_df['outage_probability'].mean()
        cic = policy_df['cic_benefit_rate'].mean()
        
        # Sanity checks
        assert 0 <= hit_rate <= 1, f"{policy}: Invalid hit rate"
        assert 0 <= outage <= 1, f"{policy}: Invalid outage prob"
        assert 0 <= cic <= 1, f"{policy}: Invalid CIC rate"
        
        print(f"  ✅ {policy}: hit={hit_rate:.3f}, outage={outage:.3f}, cic={cic:.3f}")
    
    print("\n✅ All metrics valid!\n")
    
    # Success!
    print("="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)
    print("\nYour NOMA-aware caching simulation is working correctly!")
    print("\nNext steps:")
    print("  1. Run full comparison: python -m src.simulation.noma_caching_sim")
    print("  2. Train DQN agent (if available)")
    print("  3. Generate plots and analysis")
    print("\n" + "="*70)
    
except Exception as e:
    print(f"\n❌ TEST FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

finally:
    # Restore original config
    cfg.NUM_RUNS = original_runs