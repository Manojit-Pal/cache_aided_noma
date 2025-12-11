# src/caching/dynamic_cache.py
"""
Dynamic Caching Policies for Cache-Aided NOMA

Implements adaptive caching policies that update during simulation:
- LRU (Least Recently Used)
- LFU (Least Frequently Used)  
- Random Replacement

Author: Cache-Aided NOMA Team
Date: December 2025
"""

import random
from collections import OrderedDict, Counter
from typing import Set
from .cache_base import CacheBase


class LRUCache(CacheBase):
    """
    Least Recently Used (LRU) caching policy.
    
    Evicts the least recently accessed item when cache is full.
    Uses OrderedDict for O(1) access and move-to-end operations.
    
    Pros:
        - Adapts to temporal locality
        - Simple and efficient (O(1) operations)
        - Works well when recent items are likely to be requested again
    
    Cons:
        - Ignores access frequency
        - No awareness of popularity distribution
        - Not NOMA-aware (doesn't consider channel conditions)
    
    Example:
        >>> cache = LRUCache(capacity=100)
        >>> cache.populate([1, 2, 3])  # Optional initial population
        >>> cache.is_hit(5)  # False (miss), adds to cache
        >>> cache.is_hit(5)  # True (hit), moves to front
    """
    
    def __init__(self, capacity: int, enable_noma_awareness: bool = True):
        super().__init__(capacity, enable_noma_awareness)
        self.cache = OrderedDict()  # Maintains insertion order
    
    def populate(self, items=None):
        """
        Optionally pre-populate cache with items.
        
        Args:
            items: Initial items to cache (or None to start empty)
        """
        self.cache.clear()
        if items:
            for it in list(items)[: self.capacity]:
                self.cache[it] = True
    
    def is_hit(self, item: int, update_stats: bool = True) -> bool:
        """
        Check cache and update LRU order.
        
        On hit: Move item to end (most recently used)
        On miss: Add item, evict LRU if full
        
        Args:
            item: File ID to check
            update_stats: Whether to record statistics
        
        Returns:
            bool: True if cache hit
        """
        if item in self.cache:
            # Cache hit: move to end (most recent)
            self.cache.move_to_end(item)
            if update_stats:
                self._record_hit()
            return True
        else:
            # Cache miss: add item
            if len(self.cache) >= self.capacity:
                # Evict least recently used (first item)
                self.cache.popitem(last=False)
                if update_stats:
                    self._record_eviction()
            
            self.cache[item] = True
            if update_stats:
                self._record_miss()
            return False
    
    def get_contents(self) -> Set[int]:
        """Get current cache contents."""
        return set(self.cache.keys())
    
    def clear(self):
        """Clear cache and reset statistics."""
        self.cache.clear()
        self.reset_stats()


class LFUCache(CacheBase):
    """
    Least Frequently Used (LFU) caching policy.
    
    Evicts the least frequently accessed item when cache is full.
    Uses Counter to track access frequencies.
    
    Pros:
        - Adapts to popularity distribution
        - Keeps frequently accessed items
        - Better than LRU for skewed popularity (like Zipf)
    
    Cons:
        - Slow to adapt (new popular items hard to cache)
        - "Pollution" from historically popular items
        - Not NOMA-aware
    
    Example:
        >>> cache = LFUCache(capacity=100)
        >>> cache.is_hit(5)  # False, adds with freq=1
        >>> cache.is_hit(5)  # True, freq=2
        >>> cache.is_hit(5)  # True, freq=3
        >>> # File 5 unlikely to be evicted (high frequency)
    """
    
    def __init__(self, capacity: int, enable_noma_awareness: bool = True):
        super().__init__(capacity, enable_noma_awareness)
        self.store = set()
        self.counter = Counter()  # Tracks access frequency
    
    def populate(self, items=None):
        """
        Optionally pre-populate cache with items.
        
        Args:
            items: Initial items to cache (or None to start empty)
        """
        self.store.clear()
        self.counter.clear()
        if items:
            for it in list(items)[: self.capacity]:
                self.store.add(it)
                self.counter[it] = 1
    
    def is_hit(self, item: int, update_stats: bool = True) -> bool:
        """
        Check cache and update frequency counter.
        
        On hit: Increment access counter
        On miss: Add item, evict LFU if full
        
        Args:
            item: File ID to check
            update_stats: Whether to record statistics
        
        Returns:
            bool: True if cache hit
        """
        if item in self.store:
            # Cache hit: increment frequency
            self.counter[item] += 1
            if update_stats:
                self._record_hit()
            return True
        else:
            # Cache miss: add item
            if len(self.store) >= self.capacity:
                # Evict least frequently used
                lfu_item, _ = min(self.counter.items(), key=lambda x: x[1])
                self.store.remove(lfu_item)
                del self.counter[lfu_item]
                if update_stats:
                    self._record_eviction()
            
            self.store.add(item)
            self.counter[item] = 1
            if update_stats:
                self._record_miss()
            return False
    
    def get_contents(self) -> Set[int]:
        """Get current cache contents."""
        return self.store
    
    def clear(self):
        """Clear cache and reset statistics."""
        self.store.clear()
        self.counter.clear()
        self.reset_stats()


class RandomCache(CacheBase):
    """
    Random replacement caching policy.
    
    Evicts a random item when cache is full.
    Useful as a baseline to compare against smarter policies.
    
    Pros:
        - Simple, minimal overhead
        - Useful baseline for comparison
        - No pathological worst-case behavior
    
    Cons:
        - No adaptation to any pattern
        - Poor performance in practice
        - Only useful for comparison
    
    Example:
        >>> cache = RandomCache(capacity=100)
        >>> cache.is_hit(5)  # False, adds to cache
        >>> # When full, evicts random item to make space
    """
    
    def __init__(self, capacity: int, enable_noma_awareness: bool = True):
        super().__init__(capacity, enable_noma_awareness)
        self.store = set()
    
    def populate(self, items=None):
        """
        Optionally pre-populate cache with items.
        
        Args:
            items: Initial items to cache (or None to start empty)
        """
        self.store.clear()
        if items:
            for it in list(items)[: self.capacity]:
                self.store.add(it)
    
    def is_hit(self, item: int, update_stats: bool = True) -> bool:
        """
        Check cache and randomly evict if full.
        
        Args:
            item: File ID to check
            update_stats: Whether to record statistics
        
        Returns:
            bool: True if cache hit
        """
        if item in self.store:
            # Cache hit
            if update_stats:
                self._record_hit()
            return True
        else:
            # Cache miss: add item
            if len(self.store) >= self.capacity:
                # Evict random item
                self.store.remove(random.choice(list(self.store)))
                if update_stats:
                    self._record_eviction()
            
            self.store.add(item)
            if update_stats:
                self._record_miss()
            return False
    
    def get_contents(self) -> Set[int]:
        """Get current cache contents."""
        return self.store
    
    def clear(self):
        """Clear cache and reset statistics."""
        self.store.clear()
        self.reset_stats()


if __name__ == "__main__":
    print("Testing Dynamic Caching Policies...")
    
    # Test LRU
    print("\n" + "="*60)
    print("LRU Cache Test")
    print("="*60)
    lru = LRUCache(capacity=5)
    lru.populate([1, 2, 3])
    
    requests = [1, 2, 3, 4, 5, 6, 1, 7, 1]
    for file_id in requests:
        hit = lru.is_hit(file_id)
        print(f"Request {file_id}: {'HIT' if hit else 'MISS'} | Cache: {sorted(lru.get_contents())}")
    
    lru.print_stats()
    
    # Test LFU
    print("\n" + "="*60)
    print("LFU Cache Test")
    print("="*60)
    lfu = LFUCache(capacity=5)
    
    requests = [1, 1, 2, 2, 3, 3, 4, 5, 6, 1, 2]
    for file_id in requests:
        hit = lfu.is_hit(file_id)
        print(f"Request {file_id}: {'HIT' if hit else 'MISS'} | Cache: {sorted(lfu.get_contents())}")
    
    lfu.print_stats()
    
    print("\n✅ All tests complete!")
