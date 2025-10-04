#!/usr/bin/env python
# verify_setup.py
"""
Verify your project setup before running the comparison analysis.
Run this first to catch any issues early.
"""

import sys
import os
from src.experiments import comparative_analysis

def print_header(text):
    print("\n" + "="*70)
    print(text)
    print("="*70)

def check_file(filepath):
    """Check if a file exists and return status."""
    exists = os.path.exists(filepath)
    status = "✅" if exists else "❌"
    print(f"{status} {filepath}")
    return exists

def check_import(module_path):
    """Check if a module can be imported."""
    try:
        exec(f"import {module_path}")
        print(f"✅ Can import: {module_path}")
        return True
    except Exception as e:
        print(f"❌ Cannot import: {module_path}")
        print(f"   Error: {e}")
        return False

def check_dependencies():
    """Check if all required packages are installed."""
    print_header("CHECKING DEPENDENCIES")
    
    required = {
        'numpy': 'numpy',
        'pandas': 'pandas', 
        'matplotlib': 'matplotlib.pyplot',
        'scipy': 'scipy.special'
    }
    
    all_good = True
    for package, import_path in required.items():
        if not check_import(import_path):
            all_good = False
            print(f"   → Install with: pip install {package}")
    
    return all_good

def check_project_structure():
    """Check if all necessary files exist."""
    print_header("CHECKING PROJECT STRUCTURE")
    
    critical_files = [
        'src/config.py',
        'src/utils.py',
        'src/noma/channel_model.py',
        'src/noma/power_allocation.py',
        'src/noma/sic.py',
        'src/caching/cache_base.py',
        'src/caching/static_cache.py',
        'src/experiments/comparative_analysis.py',
        'run_comparison.py',
        'test_comparison.py'
    ]
    
    all_good = True
    for filepath in critical_files:
        if not check_file(filepath):
            all_good = False
    
    return all_good

def check_config_parameters():
    """Check if config parameters are reasonable."""
    print_header("CHECKING CONFIGURATION")
    
    try:
        sys.path.insert(0, 'src')
        from src import config as cfg
        
        print("\nKey parameters:")
        print(f"  NUM_FILES:        {cfg.NUM_FILES}")
        print(f"  CACHE_SIZE:       {cfg.CACHE_SIZE} ({cfg.CACHE_SIZE/cfg.NUM_FILES*100:.1f}% of files)")
        print(f"  ZIPF_ALPHA:       {cfg.ZIPF_ALPHA}")
        print(f"  NUM_USERS:        {cfg.NUM_USERS}")
        print(f"  TARGET_RATE_BPS:  {cfg.TARGET_RATE_BPS}")
        
        if hasattr(cfg, 'CACHE_DELIVERY_RATE'):
            print(f"  CACHE_DELIVERY_RATE: {cfg.CACHE_DELIVERY_RATE}")
        
        # Warnings
        warnings = []
        if cfg.CACHE_SIZE / cfg.NUM_FILES > 0.5:
            warnings.append("⚠️  Cache size > 50% of files (may reduce benefit visibility)")
        if cfg.ZIPF_ALPHA < 0.5:
            warnings.append("⚠️  ZIPF_ALPHA < 0.5 (very low skew, less cache benefit)")
        if cfg.NUM_USERS < 50:
            warnings.append("⚠️  NUM_USERS < 50 (may have high variance)")
        
        if warnings:
            print("\nWarnings:")
            for w in warnings:
                print(f"  {w}")
        else:
            print("\n✅ Configuration looks good!")
        
        return True
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return False

def test_basic_functionality():
    """Test if basic functions work."""
    print_header("TESTING BASIC FUNCTIONALITY")
    
    try:
        sys.path.insert(0, 'src')
        
        # Test 1: Import main analysis class
        print("\n1. Testing comparative analysis import...")
        from src.experiments.comparative_analysis import ComparativeNOMAAnalysis
        print("   ✅ Successfully imported ComparativeNOMAAnalysis")
        
        # Test 2: Create analyzer instance
        print("\n2. Testing analyzer creation...")
        from src import config as cfg
        analyzer = ComparativeNOMAAnalysis(cfg)
        print(f"   ✅ Analyzer created with {len(analyzer.snr_db_range)} SNR points")
        
        # Test 3: Test cache setup
        print("\n3. Testing cache setup...")
        cache = analyzer.setup_cache(cache_enabled=True)
        print(f"   ✅ Cache created with capacity {cache.capacity}")
        
        # Test 4: Test channel generation
        print("\n4. Testing channel generation...")
        gain_w, gain_s, noise_power = analyzer.generate_user_pair_channels(snr_db=20)
        print(f"   ✅ Generated channel gains: weak={gain_w:.2e}, strong={gain_s:.2e}")
        
        # Test 5: Test NOMA transmission simulation
        print("\n5. Testing NOMA transmission simulation...")
        outcome = analyzer.simulate_noma_transmission(
            gain_w, gain_s, noise_power,
            cache_hit_weak=False, cache_hit_strong=False
        )
        print(f"   ✅ Simulated transmission: rate_weak={outcome['rate_weak']:.2f}, rate_strong={outcome['rate_strong']:.2f}")
        
        print("\n✅ All basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error in functionality test: {e}")
        import traceback
        traceback.print_exc()
        return False

def estimate_runtime():
    """Estimate runtime for full analysis."""
    print_header("RUNTIME ESTIMATION")
    
    try:
        sys.path.insert(0, 'src')
        from src.experiments.comparative_analysis import ComparativeNOMAAnalysis
        from src import config as cfg
        
        analyzer = ComparativeNOMAAnalysis(cfg)
        
        num_snr = len(analyzer.snr_db_range)
        num_realizations = analyzer.num_realizations
        
        # Rough estimate: ~0.5 seconds per 100 realizations
        estimated_time_per_snr = (num_realizations / 100) * 0.5
        total_time_minutes = (estimated_time_per_snr * num_snr * 2) / 60  # *2 for both scenarios
        
        print(f"\nAnalysis parameters:")
        print(f"  SNR points:       {num_snr}")
        print(f"  Realizations:     {num_realizations}")
        print(f"  Total sims:       {num_snr * num_realizations * 2} (×2 for cache/no-cache)")
        print(f"\n  Estimated time:   {total_time_minutes:.1f} minutes")
        
        if total_time_minutes > 20:
            print("\n⚠️  Analysis will take > 20 minutes")
            print("   Consider running test_comparison.py first for quick validation")
        
        return True
    except Exception as e:
        print(f"❌ Error estimating runtime: {e}")
        return False

def main():
    """Main verification routine."""
    print("\n" + "🔍 " * 23)
    print("PROJECT SETUP VERIFICATION")
    print("🔍 " * 23)
    
    results = {
        'dependencies': check_dependencies(),
        'structure': check_project_structure(),
        'config': check_config_parameters(),
        'functionality': test_basic_functionality(),
        'runtime': estimate_runtime()
    }
    
    print_header("VERIFICATION SUMMARY")
    
    all_passed = all(results.values())
    
    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {check.upper()}")
    
    if all_passed:
        print("\n" + "🎉 " * 23)
        print("ALL CHECKS PASSED - READY TO RUN!")
        print("🎉 " * 23)
        print("\nNext steps:")
        print("  1. Quick test:    python test_comparison.py")
        print("  2. Full analysis: python run_comparison.py")
    else:
        print("\n" + "⚠️ " * 23)
        print("SOME CHECKS FAILED - FIX ISSUES ABOVE")
        print("⚠️ " * 23)
        print("\nPlease fix the issues marked with ❌ before proceeding.")
    
    print("\n" + "="*70 + "\n")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())