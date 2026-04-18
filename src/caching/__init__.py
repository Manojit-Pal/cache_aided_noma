# src/caching/__init__.py
"""
Caching Module for Cache-Aided NOMA Systems

Provides caching policies and utilities:
- Base class with NOMA-awareness
- Static policies (Top-K)
- Dynamic policies (LRU, LFU, Random)
- Learning-based policies (DQN)
- Helper functions for NOMA integration

Author: Cache-Aided NOMA Team
Date: December 2025
"""

# Import base class
from .cache_base import (
    CacheBase,
    get_cache_status_for_users,
    compute_cic_matrix
)

# Import static policies
from .static_cache import StaticTopKCache

# Import dynamic policies  
from .dynamic_cache import (
    LRUCache,
    LFUCache,
    RandomCache
)

# DQN imports (conditional - may not exist yet)
try:
    from .dqn_cache_final import DQNCache
    HAS_DQN = True
except ImportError:
    HAS_DQN = False

# DDPG imports (conditional)
try:
    from .ddpg_cache import DDPGCache
    HAS_DDPG = True
except ImportError:
    HAS_DDPG = False

# MADDPG imports (conditional)
try:
    from .maddpg_cache import MADDPGCache
    HAS_MADDPG = True
except ImportError:
    HAS_MADDPG = False

# Define what's available when using "from caching import *"
__all__ = [
    # Base class
    'CacheBase',
    
    # Static policies
    'StaticTopKCache',
    
    # Dynamic policies
    'LRUCache',
    'LFUCache',
    'RandomCache',
    
    # Helper functions
    'get_cache_status_for_users',
    'compute_cic_matrix',
    
    # Factory function
    'create_cache',
]

# Add DRL agents if available
if HAS_DQN:
    __all__.append('DQNCache')
if HAS_DDPG:
    __all__.append('DDPGCache')
if HAS_MADDPG:
    __all__.append('MADDPGCache')


def create_cache(policy: str, capacity: int, **kwargs):
    """
    Factory function to create cache instances.
    
    Args:
        policy: Cache policy name
            - 'topk' or 'static': Static Top-K cache
            - 'lru': Least Recently Used
            - 'lfu': Least Frequently Used
            - 'random': Random replacement
            - 'dqn' or 'stable_dqn': DQN-based cache (if available)
        capacity: Cache capacity (number of files)
        **kwargs: Additional arguments passed to cache constructor
    
    Returns:
        CacheBase: Cache instance
    
    Example:
        >>> cache = create_cache('lru', capacity=100)
        >>> cache = create_cache('topk', capacity=200, enable_noma_awareness=True)
        >>> cache = create_cache('dqn', capacity=150, num_files=1000, num_users=50)
    """
    policy = policy.lower()
    
    if policy in ['topk', 'static', 'top-k']:
        return StaticTopKCache(capacity, **kwargs)
    
    elif policy == 'lru':
        return LRUCache(capacity, **kwargs)
    
    elif policy == 'lfu':
        return LFUCache(capacity, **kwargs)
    
    elif policy == 'random':
        return RandomCache(capacity, **kwargs)
    
    elif policy in ['dqn', 'stable_dqn']:
        if not HAS_DQN:
            raise ImportError(
                "DQN cache not available. "
                "Ensure dqn_cache_final.py exists and PyTorch is installed."
            )
        return DQNCache(capacity, **kwargs)
    
    elif policy == 'ddpg':
        if not HAS_DDPG:
            raise ImportError(
                "DDPG cache not available. "
                "Ensure ddpg_cache.py exists and PyTorch is installed."
            )
        return DDPGCache(capacity, **kwargs)
    
    elif policy == 'maddpg':
        if not HAS_MADDPG:
            raise ImportError(
                "MADDPG cache not available. "
                "Ensure maddpg_cache.py exists and PyTorch is installed."
            )
        return MADDPGCache(capacity, **kwargs)
    
    else:
        available_policies = "topk, lru, lfu, random"
        if HAS_DQN: available_policies += ", dqn"
        if HAS_DDPG: available_policies += ", ddpg"
        if HAS_MADDPG: available_policies += ", maddpg"
        raise ValueError(
            f"Unknown cache policy: '{policy}'. \n"
            f"Available policies: {available_policies}"
        )


__version__ = '1.0.0'
__author__ = 'Cache-Aided NOMA Team'
