# src/caching/static_cache.py
"""
Static Top-K Caching Policy for Cache-Aided NOMA

Caches the K most popular files based on popularity distribution.
Simple but effective baseline for comparison with learning-based policies.

Author: Cache-Aided NOMA Team
Date: December 2025
"""

from .cache_base import CacheBase
from typing import Iterable, Set
import numpy as np


class StaticTopKCache(CacheBase):
    """
    Static cache that stores top-K most popular items.
    
    This is a simple baseline policy that:
    1. Analyzes popularity distribution (e.g., Zipf)
    2. Caches the K most popular files
    3. Never updates during simulation
    
    Pros:
        - Simple, no computational overhead
        - Works well for highly skewed distributions (Zipf)
        - Optimal if popularity is perfectly known and static
    
    Cons:
        - Cannot adapt to changing popularity
        - Requires perfect popularity knowledge
        - Ignores channel conditions and NOMA pairing
    
    Example:
        >>> # Cache top-100 files by Zipf distribution
        >>> cache = StaticTopKCache(capacity=100)
        >>> popularity_sorted = [1, 2, 3, ...]  # Files sorted by popularity
        >>> cache.populate(popularity_sorted)
        >>> 
        >>> # Check if file is cached
        >>> if cache.is_hit(5):
        ...     print("Cache hit!")
    """
    
    def __init__(self, capacity: int, enable_noma_awareness: bool = True):
        """
        Initialize static top-K cache.
        
        Args:
            capacity: Number of files to cache
            enable_noma_awareness: Track NOMA-specific metrics
        """
        super().__init__(capacity, enable_noma_awareness)
        self.contents = set()  # Set of cached file IDs
    
    def populate(self, items: Iterable[int]):
        """
        Populate cache with top-K items.
        
        Args:
            items: Iterable of file IDs ordered by popularity (most popular first)
        
        Example:
            >>> # Files sorted by Zipf popularity
            >>> popularity_sorted = [1, 2, 3, 4, 5, 6, ...]
            >>> cache.populate(popularity_sorted)
            >>> # Caches files [1, 2, 3, ..., K]
        """
        sorted_list = list(items)
        topk = sorted_list[: self.capacity]
        self.contents = set(topk)
    
    def is_hit(self, item: int, update_stats: bool = True) -> bool:
        """
        Check if item is in cache.
        
        Args:
            item: File ID to check
            update_stats: Whether to record hit/miss statistics
        
        Returns:
            bool: True if cache hit, False otherwise
        """
        hit = int(item) in self.contents
        
        # Update statistics
        if update_stats:
            if hit:
                self._record_hit()
            else:
                self._record_miss()
        
        return hit
    
    def get_contents(self) -> Set[int]:
        """
        Get current cache contents.
        
        Returns:
            Set[int]: Set of cached file IDs
        """
        return self.contents
    
    def clear(self):
        """
        Clear cache contents and reset statistics.
        """
        self.contents.clear()
        self.reset_stats()


if __name__ == "__main__":
    print("Testing StaticTopKCache...")
    
    # Create cache for top-10 files
    cache = StaticTopKCache(capacity=10)
    
    # Populate with files sorted by popularity
    popularity_sorted = list(range(1, 101))  # Files 1-100, file 1 most popular
    cache.populate(popularity_sorted)
    
    print(f"\nCache populated with top-{cache.capacity} files")
    print(f"Contents: {sorted(cache.get_contents())}")
    print(f"Cache: {cache}")
    
    # Simulate requests
    print("\nSimulating requests...")
    requests = [1, 1, 2, 3, 15, 20, 1, 5, 100, 2]  # Mix of popular and unpopular
    
    for file_id in requests:
        hit = cache.is_hit(file_id)
        print(f"  Request file {file_id:3d}: {'HIT' if hit else 'MISS'}")
    
    # Print statistics
    cache.print_stats()
    
    print("\n✅ StaticTopKCache test complete!")
