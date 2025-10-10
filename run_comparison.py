# run_comparison.py
"""
Quick runner script for Cache vs Non-Cache NOMA comparison.
Place this in your project root directory.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from experiments.comparative_analysis import main

if __name__ == "__main__":
    print("\n🚀 Starting Comparative Analysis...")
    print("This will compare Cache-Aided NOMA vs Traditional NOMA\n")
    
    main()
    
    print("\n📊 Check the generated plots:")
    print("   1. cache_vs_nocache_comparison.png")
    print("   2. performance_gain_with_cache.png")
    print("\n💾 Check the generated CSV files:")
    print("   1. results_cache_aided_noma.csv")
    print("   2. results_traditional_noma.csv")

    