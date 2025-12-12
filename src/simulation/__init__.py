# src/simulation/__init__.py
"""
Simulation Module for Cache-Aided NOMA Systems

This module provides comprehensive simulation tools for evaluating
caching policies in NOMA systems with:

- DQN Training and Evaluation (stable_dqn_sim.py)
- Baseline NOMA Caching Simulator (noma_caching_sim.py)
- Train & Evaluate Pipeline (train_and_evaluate_dqn.py)

Main Classes:
- NOMADQNTrainer: Train DQN caches with NOMA-aware rewards
- CachePolicyEvaluator: Compare different caching policies
- NOMACachingSimulator: General-purpose NOMA caching simulator

Author: Cache-Aided NOMA Team
Date: December 2025
Version: 2.3
"""

__version__ = '2.3.0'
__author__ = 'Cache-Aided NOMA Team'

# Import main simulator classes
try:
    from .stable_dqn_sim import (
        NOMADQNTrainer,
        CachePolicyEvaluator,
        plot_training_curves,
        plot_policy_comparison
    )
    HAS_STABLE_DQN_SIM = True
except ImportError as e:
    HAS_STABLE_DQN_SIM = False
    print(f"⚠️  stable_dqn_sim not fully available: {e}")

try:
    from .noma_caching_sim import (
        NOMACachingSimulator,
        run_baseline_comparison,
        run_dqn_training,
        plot_comparison_results,
        plot_dqn_training,
        compute_popularity_ranking,
        generate_time_varying_channels
    )
    HAS_NOMA_CACHING_SIM = True
except ImportError as e:
    HAS_NOMA_CACHING_SIM = False
    print(f"⚠️  noma_caching_sim not fully available: {e}")

# Export public API
__all__ = [
    # Version info
    '__version__',
    '__author__',
    
    # Main classes from stable_dqn_sim
    'NOMADQNTrainer',
    'CachePolicyEvaluator',
    
    # Main classes from noma_caching_sim
    'NOMACachingSimulator',
    
    # Utility functions
    'run_baseline_comparison',
    'run_dqn_training',
    'compute_popularity_ranking',
    'generate_time_varying_channels',
    
    # Plotting functions
    'plot_training_curves',
    'plot_policy_comparison',
    'plot_comparison_results',
    'plot_dqn_training',
    
    # Status flags
    'HAS_STABLE_DQN_SIM',
    'HAS_NOMA_CACHING_SIM',
]

# Module-level docstring for help()
def get_info():
    """
    Print information about available simulation tools.
    """
    print(f"\nCache-Aided NOMA Simulation Module v{__version__}")
    print("=" * 50)
    print(f"\nStable DQN Simulator: {'✅ Available' if HAS_STABLE_DQN_SIM else '❌ Not Available'}")
    print(f"NOMA Caching Simulator: {'✅ Available' if HAS_NOMA_CACHING_SIM else '❌ Not Available'}")
    print("\nMain Classes:")
    if HAS_STABLE_DQN_SIM:
        print("  - NOMADQNTrainer: Train DQN caches")
        print("  - CachePolicyEvaluator: Compare policies")
    if HAS_NOMA_CACHING_SIM:
        print("  - NOMACachingSimulator: General simulator")
    print("\nUsage:")
    print("  from src.simulation import NOMADQNTrainer")
    print("  trainer = NOMADQNTrainer(config)")
    print("  trained_cache, history = trainer.train(num_episodes=2000)")
    print()

# Add to __all__
__all__.append('get_info')