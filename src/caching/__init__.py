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

try:
    from .improved_dqn_noma_cache import ImprovedDQNCache
    HAS_IMPROVED_DQN = True
except ImportError:
    HAS_IMPROVED_DQN = False

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

# Add DQN if available
if HAS_DQN:
    __all__.append('DQNCache')

if HAS_IMPROVED_DQN:
    __all__.append('ImprovedDQNCache')


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
            - 'improved_dqn': Improved DQN cache (if available)
        capacity: Cache capacity (number of files)
        **kwargs: Additional arguments passed to cache constructor
    
    Returns:
        CacheBase: Cache instance
    
    Example:
        >>> cache = create_cache('lru', capacity=100)
        >>> cache = create_cache('topk', capacity=200, enable_noma_awareness=True)
        >>> cache = create_cache('dqn', capacity=150, config=cfg)
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
            raise ImportError("DQN cache not available. Check dqn_cache_final.py")
        return DQNCache(capacity, **kwargs)
    
    elif policy == 'improved_dqn':
        if not HAS_IMPROVED_DQN:
            raise ImportError("Improved DQN cache not available. Check improved_dqn_noma_cache.py")
        return ImprovedDQNCache(capacity, **kwargs)
    
    else:
        raise ValueError(f"Unknown cache policy: {policy}. "
                        f"Available: topk, lru, lfu, random" + 
                        (", dqn" if HAS_DQN else "") +
                        (", improved_dqn" if HAS_IMPROVED_DQN else ""))


__version__ = '1.0.0'
__author__ = 'Cache-Aided NOMA Team'
