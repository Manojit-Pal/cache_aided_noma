# test_comparison.py
"""
Quick test version with fewer simulations - runs in ~1 minute.
Use this to verify everything works before running full analysis.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from experiments.comparative_analysis import ComparativeNOMAAnalysis
from src import config as cfg

def quick_test():
    """Run a quick test with reduced parameters."""
    print("="*70)
    print("QUICK TEST: Cache-Aided NOMA vs Traditional NOMA")
    print("="*70)
    print("\n⚠️  This is a QUICK TEST with reduced parameters")
    print("For full results, run: python run_comparison.py\n")
    
    # Create analyzer with reduced parameters for speed
    analyzer = ComparativeNOMAAnalysis(cfg)
    analyzer.snr_db_range = [0, 10, 20]  # Only 3 SNR points
    analyzer.num_realizations = 100  # Reduced from 1000
    
    print(f"Testing {len(analyzer.snr_db_range)} SNR points")
    print(f"Using {analyzer.num_realizations} realizations per point")
    print("Expected runtime: ~30-60 seconds\n")
    
    # Run comparison
    df_cache, df_no_cache = analyzer.run_full_comparison()
    
    # Print quick summary
    print("\n" + "="*70)
    print("QUICK TEST RESULTS")
    print("="*70)
    
    for idx in range(len(df_cache)):
        snr = df_cache['snr_db'].iloc[idx]
        sum_cache = df_cache['avg_sum_rate'].iloc[idx]
        sum_no_cache = df_no_cache['avg_sum_rate'].iloc[idx]
        gain = (sum_cache - sum_no_cache) / sum_no_cache * 100
        
        print(f"\nSNR = {snr} dB:")
        print(f"  Sum-Rate (Cache):      {sum_cache:.3f} bps/Hz")
        print(f"  Sum-Rate (No Cache):   {sum_no_cache:.3f} bps/Hz")
        print(f"  Improvement:           {gain:+.1f}%")
    
    # Create quick plots
    analyzer.plot_all_comparisons(df_cache, df_no_cache, save_dir='./')
    
    print("\n" + "="*70)
    print("✅ QUICK TEST COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - cache_vs_nocache_comparison.png")
    print("  - performance_gain_with_cache.png")
    print("\nIf results look good, run full analysis:")
    print("  python run_comparison.py")
    print("\nFor presentation, use the FULL analysis results!")
    print("="*70 + "\n")

if __name__ == "__main__":
    quick_test()