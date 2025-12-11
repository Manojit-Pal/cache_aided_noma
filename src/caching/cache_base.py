# src/caching/cache_base.py
"""
Base Cache Class for Cache-Aided NOMA Systems

This module provides the abstract base class for all caching policies.
Key features:
- NOMA-aware caching with CIC (Cache-aided Interference Cancellation) support
- Performance tracking (hit rate, evictions, etc.)
- Integration with NOMA channel and user pairing information
- Unified interface for static and dynamic policies

Author: Cache-Aided NOMA Team
Date: December 2025
"""

from abc import ABC, abstractmethod
from typing import Iterable, List, Set, Dict, Optional, Tuple
import numpy as np


class CacheBase(ABC):
    """
    Abstract base class for all caching policies.
    
    This class defines the interface that all caching policies must implement,
    whether static (Top-K), dynamic (LRU, LFU), or learning-based (DQN).
    
    Key Concepts:
    1. **Placement Phase**: populate() is called to fill cache initially
    2. **Delivery Phase**: is_hit() is called for each user request
    3. **Update Phase**: update() is called to adapt cache (for dynamic policies)
    4. **NOMA Integration**: Cache status affects SIC and enables CIC
    
    Cache-Aided Interference Cancellation (CIC):
        When a user has the interfering content cached, they can perfectly
        cancel that interference, significantly improving SINR:
        - Weak user with strong's content cached → no interference
        - Strong user with weak's content cached → perfect SIC
    
    Attributes:
        capacity (int): Maximum number of files the cache can store
        total_requests (int): Total number of requests processed
        total_hits (int): Number of cache hits
        total_misses (int): Number of cache misses
        evictions (int): Number of cache evictions (for dynamic policies)
        cic_opportunities (int): Number of times CIC was enabled
    """
    
    def __init__(self, capacity: int, enable_noma_awareness: bool = True):
        """
        Initialize cache with given capacity.
        
        Args:
            capacity: Maximum number of files to store
            enable_noma_awareness: Whether to track NOMA-specific metrics
        """
        if capacity <= 0:
            raise ValueError(f"Cache capacity must be positive, got {capacity}")
        
        self.capacity = capacity
        self.enable_noma_awareness = enable_noma_awareness
        
        # Performance tracking
        self.total_requests = 0
        self.total_hits = 0
        self.total_misses = 0
        self.evictions = 0
        
        # NOMA-specific tracking
        self.cic_opportunities = 0  # Times cache enabled CIC
        self.noma_paired_hits = 0   # Hits that helped paired user via CIC
        self.channel_gains = {}     # Optional: store user channel gains
        self.user_pairings = {}     # Optional: store NOMA user pairs
    
    # =========================================================================
    # MAGIC METHODS (Python built-ins)
    # =========================================================================
    
    def __contains__(self, item: int) -> bool:
        """
        Enable 'in' operator: if file_id in cache:
        
        Args:
            item: File ID to check
        
        Returns:
            bool: True if item is in cache
        
        Example:
            >>> cache = TopKCache(100)
            >>> cache.populate([1, 2, 3])
            >>> if 2 in cache:
            ...     print("Cache hit!")
        """
        return self.is_hit(item, update_stats=False)
    
    def __len__(self) -> int:
        """
        Enable len() function: current_size = len(cache)
        
        Returns:
            int: Current number of items in cache
        """
        return len(self.get_contents())
    
    def __repr__(self) -> str:
        """
        String representation for debugging.
        
        Returns:
            str: Human-readable cache description
        """
        hit_rate = self.get_hit_rate()
        return (f"{self.__class__.__name__}("
                f"capacity={self.capacity}, "
                f"size={len(self)}, "
                f"hit_rate={hit_rate:.2%})")
    
    # =========================================================================
    # ABSTRACT METHODS (Must be implemented by subclasses)
    # =========================================================================
    
    @abstractmethod
    def populate(self, items: Iterable[int]):
        """
        Populate cache with items (placement phase).
        
        This is called once at the beginning to fill the cache.
        For static policies: stores top-K popular items
        For dynamic policies: may start empty or with initial items
        
        Args:
            items: Iterable of file IDs (may be ordered by popularity)
        
        Example:
            >>> cache.populate([1, 2, 3, 4, 5])  # Cache top-5 files
        """
        pass
    
    @abstractmethod
    def is_hit(self, item: int, update_stats: bool = True) -> bool:
        """
        Check if item is in cache (delivery phase).
        
        For dynamic policies, this may also update internal state
        (e.g., LRU moves item to front, LFU increments counter).
        
        Args:
            item: File ID requested by user
            update_stats: Whether to update hit/miss statistics
        
        Returns:
            bool: True if cache hit, False if cache miss
        
        Example:
            >>> hit = cache.is_hit(5)
            >>> if hit:
            ...     print("Cache hit! Use CIC.")
            ... else:
            ...     print("Cache miss. Fetch via NOMA.")
        """
        pass
    
    @abstractmethod
    def clear(self):
        """
        Clear all cache contents and reset statistics.
        
        Used when starting a new simulation run or resetting cache.
        
        Example:
            >>> cache.clear()
            >>> assert len(cache) == 0
        """
        pass
    
    # =========================================================================
    # CONCRETE METHODS (Provided by base class)
    # =========================================================================
    
    def get_contents(self) -> Set[int]:
        """
        Get current cache contents.
        
        Returns:
            Set[int]: Set of file IDs currently in cache
        
        Note:
            Subclasses should override if they don't use 'store' or 'contents'
        """
        # Try common attribute names used by subclasses
        if hasattr(self, 'contents'):
            return self.contents
        elif hasattr(self, 'store'):
            return self.store
        elif hasattr(self, 'cache'):
            return set(self.cache.keys())
        else:
            return set()
    
    def update(self, item: int, reward: Optional[float] = None, 
               noma_success: bool = True, **kwargs):
        """
        Update cache state after a request (for dynamic/learning policies).
        
        For static policies: does nothing
        For dynamic policies (LRU, LFU): updates internal structures
        For learning policies (DQN): updates based on reward signal
        
        Args:
            item: File ID that was requested
            reward: Reward signal (for RL policies)
            noma_success: Whether NOMA transmission succeeded
            **kwargs: Additional policy-specific parameters
                - channel_gain: User's channel gain
                - paired_user_id: ID of paired user in NOMA
                - sinr: Achieved SINR
                - ber: Bit error rate
        
        Example:
            >>> # For DQN cache
            >>> cache.update(file_id=5, reward=10.0, noma_success=True)
            >>> 
            >>> # For LRU cache  
            >>> cache.update(file_id=5)  # Does nothing (LRU updates in is_hit)
        """
        # Base implementation: do nothing (for static policies)
        # Dynamic/learning policies should override this
        pass
    
    def request(self, item: int, user_id: Optional[int] = None,
                channel_gain: Optional[float] = None,
                paired_user: Optional[int] = None) -> Dict:
        """
        Unified request interface that handles both hit detection and NOMA integration.
        
        This is the recommended high-level interface for simulations.
        It combines is_hit() with NOMA-aware features.
        
        Args:
            item: Requested file ID
            user_id: ID of requesting user
            channel_gain: User's channel gain (for NOMA-aware caching)
            paired_user: ID of NOMA paired user (if any)
        
        Returns:
            Dict containing:
                - 'hit': bool - whether request was a cache hit
                - 'cic_enabled': bool - whether CIC can be applied
                - 'paired_user_cached': bool - whether paired user has content
        
        Example:
            >>> result = cache.request(file_id=5, user_id=10, 
            ...                       paired_user=20, channel_gain=1e-7)
            >>> if result['hit']:
            ...     delivery_rate = config.CACHE_DELIVERY_RATE
            >>> elif result['cic_enabled']:
            ...     # Use NOMA with CIC (better SINR)
            ...     pass
        """
        # Check if item is cached
        hit = self.is_hit(item, update_stats=True)
        
        # Check if CIC can be enabled (paired user has this content)
        cic_enabled = False
        paired_user_cached = False
        
        if self.enable_noma_awareness and paired_user is not None:
            # Check if paired user has this item (enables CIC for them)
            paired_user_cached = hit  # If we have it, they benefit from CIC
            if paired_user_cached:
                self.cic_opportunities += 1
        
        # Store channel gain if provided (for NOMA-aware policies)
        if user_id is not None and channel_gain is not None:
            self.channel_gains[user_id] = channel_gain
        
        # Store pairing information
        if user_id is not None and paired_user is not None:
            self.user_pairings[user_id] = paired_user
        
        return {
            'hit': hit,
            'cic_enabled': paired_user_cached,
            'paired_user_cached': paired_user_cached,
            'cache_size': len(self),
        }
    
    def check_cic_benefit(self, user_id: int, requested_file: int, 
                         paired_user: int) -> Tuple[bool, bool]:
        """
        Check CIC benefits for NOMA user pair.
        
        Args:
            user_id: ID of requesting user
            requested_file: File requested by user
            paired_user: ID of NOMA paired user
        
        Returns:
            Tuple of (user_gets_cic, paired_gets_cic):
                - user_gets_cic: True if user can cancel paired user's interference
                - paired_gets_cic: True if paired user can cancel this user's interference
        
        Example:
            >>> # User 1 requests file 5, paired with user 2 (requesting file 8)
            >>> user_cic, paired_cic = cache.check_cic_benefit(1, 5, 2)
            >>> # user_cic: True if cache has file 8 (user 1 cancels user 2's interference)
            >>> # paired_cic: True if cache has file 5 (user 2 cancels user 1's interference)
        """
        user_gets_cic = False
        paired_gets_cic = False
        
        # User can cancel paired user's interference if paired user's file is cached
        # (We'd need to know what paired user requested - this is simplified)
        
        # Paired user can cancel this user's interference if this file is cached
        paired_gets_cic = self.is_hit(requested_file, update_stats=False)
        
        return user_gets_cic, paired_gets_cic
    
    # =========================================================================
    # STATISTICS AND MONITORING
    # =========================================================================
    
    def get_hit_rate(self) -> float:
        """
        Calculate cache hit rate.
        
        Returns:
            float: Hit rate (0.0 to 1.0)
        """
        if self.total_requests == 0:
            return 0.0
        return self.total_hits / self.total_requests
    
    def get_miss_rate(self) -> float:
        """
        Calculate cache miss rate.
        
        Returns:
            float: Miss rate (0.0 to 1.0)
        """
        return 1.0 - self.get_hit_rate()
    
    def get_cic_rate(self) -> float:
        """
        Calculate rate of CIC opportunities (cache hits that enabled CIC).
        
        Returns:
            float: CIC opportunity rate (0.0 to 1.0)
        """
        if self.total_requests == 0:
            return 0.0
        return self.cic_opportunities / self.total_requests
    
    def stats(self) -> Dict:
        """
        Get comprehensive cache statistics.
        
        Returns:
            Dict with cache performance metrics
        """
        return {
            # Basic cache info
            'capacity': self.capacity,
            'current_size': len(self),
            'utilization': len(self) / self.capacity if self.capacity > 0 else 0,
            
            # Performance metrics
            'total_requests': self.total_requests,
            'total_hits': self.total_hits,
            'total_misses': self.total_misses,
            'hit_rate': self.get_hit_rate(),
            'miss_rate': self.get_miss_rate(),
            
            # Dynamic cache metrics
            'evictions': self.evictions,
            
            # NOMA-specific metrics
            'cic_opportunities': self.cic_opportunities,
            'cic_rate': self.get_cic_rate(),
            'noma_paired_hits': self.noma_paired_hits,
            'noma_awareness_enabled': self.enable_noma_awareness,
        }
    
    def print_stats(self):
        """
        Print cache statistics in human-readable format.
        """
        stats = self.stats()
        print("\n" + "="*60)
        print(f"{self.__class__.__name__} Statistics")
        print("="*60)
        
        print(f"\n📦 Cache Capacity:")
        print(f"   Max: {stats['capacity']} files")
        print(f"   Current: {stats['current_size']} files")
        print(f"   Utilization: {stats['utilization']:.1%}")
        
        print(f"\n📊 Performance:")
        print(f"   Total Requests: {stats['total_requests']}")
        print(f"   Cache Hits: {stats['total_hits']}")
        print(f"   Cache Misses: {stats['total_misses']}")
        print(f"   Hit Rate: {stats['hit_rate']:.2%}")
        print(f"   Miss Rate: {stats['miss_rate']:.2%}")
        
        if stats['evictions'] > 0:
            print(f"\n🔄 Dynamic Cache:")
            print(f"   Evictions: {stats['evictions']}")
        
        if self.enable_noma_awareness:
            print(f"\n📡 NOMA Integration:")
            print(f"   CIC Opportunities: {stats['cic_opportunities']}")
            print(f"   CIC Rate: {stats['cic_rate']:.2%}")
            print(f"   NOMA Paired Hits: {stats['noma_paired_hits']}")
        
        print("="*60 + "\n")
    
    def reset_stats(self):
        """
        Reset statistics counters (but keep cache contents).
        
        Useful when starting a new evaluation phase.
        """
        self.total_requests = 0
        self.total_hits = 0
        self.total_misses = 0
        self.evictions = 0
        self.cic_opportunities = 0
        self.noma_paired_hits = 0
    
    def _record_hit(self):
        """Internal: record a cache hit."""
        self.total_requests += 1
        self.total_hits += 1
    
    def _record_miss(self):
        """Internal: record a cache miss."""
        self.total_requests += 1
        self.total_misses += 1
    
    def _record_eviction(self):
        """Internal: record a cache eviction."""
        self.evictions += 1


# =============================================================================
# HELPER FUNCTIONS FOR NOMA INTEGRATION
# =============================================================================

def get_cache_status_for_users(cache: CacheBase, user_requests: Dict[int, int]) -> Dict[int, bool]:
    """
    Get cache status for multiple users.
    
    Args:
        cache: Cache instance
        user_requests: Dict mapping user_id -> requested_file_id
    
    Returns:
        Dict mapping user_id -> is_cached (bool)
    
    Example:
        >>> requests = {1: 5, 2: 8, 3: 5}  # Users 1 & 3 want file 5, user 2 wants file 8
        >>> cache_status = get_cache_status_for_users(cache, requests)
        >>> # {1: True, 2: False, 3: True} if file 5 cached but not file 8
    """
    return {user_id: cache.is_hit(file_id, update_stats=False) 
            for user_id, file_id in user_requests.items()}


def compute_cic_matrix(cache: CacheBase, user_pairs: List[Tuple[int, int]], 
                       user_requests: Dict[int, int]) -> np.ndarray:
    """
    Compute CIC benefit matrix for NOMA user pairs.
    
    Args:
        cache: Cache instance
        user_pairs: List of (weak_user_id, strong_user_id) tuples
        user_requests: Dict mapping user_id -> requested_file_id
    
    Returns:
        np.ndarray: Matrix where cic_matrix[i, :] = [weak_gets_cic, strong_gets_cic]
            for the i-th user pair
    
    Example:
        >>> pairs = [(1, 2), (3, 4)]  # Two NOMA pairs
        >>> requests = {1: 5, 2: 8, 3: 10, 4: 12}
        >>> cic_matrix = compute_cic_matrix(cache, pairs, requests)
        >>> # cic_matrix[0, 0] = True if user 1 can cancel user 2's interference
        >>> # cic_matrix[0, 1] = True if user 2 can cancel user 1's interference
    """
    num_pairs = len(user_pairs)
    cic_matrix = np.zeros((num_pairs, 2), dtype=bool)
    
    for i, (weak_id, strong_id) in enumerate(user_pairs):
        weak_file = user_requests.get(weak_id)
        strong_file = user_requests.get(strong_id)
        
        if weak_file is not None and strong_file is not None:
            # Weak user gets CIC if strong user's file is cached
            cic_matrix[i, 0] = cache.is_hit(strong_file, update_stats=False)
            
            # Strong user gets CIC if weak user's file is cached
            cic_matrix[i, 1] = cache.is_hit(weak_file, update_stats=False)
    
    return cic_matrix
