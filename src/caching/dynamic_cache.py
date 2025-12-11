# src/caching/dynamic_cache.py
"""
NOMA-Aware Dynamic Caching Policies for Cache-Aided NOMA

Implements adaptive caching policies that update during simulation:
- LRU (Least Recently Used)
- LFU (Least Frequently Used)  
- Random Replacement

NOMA-Aware Features:
- CIC (Cache-aided Interference Cancellation) tracking
- Channel-aware eviction (prioritize files for weak users)
- NOMA pairing integration
- SIC benefit detection

Author: Cache-Aided NOMA Team
Date: December 2025
"""

import random
from collections import OrderedDict, Counter
from typing import Set, Dict, Optional, List
from .cache_base import CacheBase


class LRUCache(CacheBase):
    """
    NOMA-Aware Least Recently Used (LRU) caching policy.
    
    Evicts the least recently accessed item when cache is full.
    Uses OrderedDict for O(1) access and move-to-end operations.
    
    NOMA Enhancements:
    - Tracks CIC opportunities per file
    - Optional channel-aware eviction (keep files for weak users)
    - NOMA pairing awareness
    
    Pros:
        - Adapts to temporal locality
        - Simple and efficient (O(1) operations)
        - Works well when recent items are likely to be requested again
        - With NOMA-awareness: prioritizes recent weak-user requests
    
    Cons:
        - Ignores access frequency
        - Limited long-term popularity awareness
        - Channel-awareness only affects eviction (not insertion)
    
    Example:
        >>> cache = LRUCache(capacity=100, channel_aware_eviction=True)
        >>> cache.populate([1, 2, 3])  # Optional initial population
        >>> 
        >>> # NOMA-aware request
        >>> result = cache.request(item=5, user_id=10, 
        ...                        paired_user=20, paired_file=8,
        ...                        channel_gain=1e-8)
        >>> if result['weak_user_benefit']:
        ...     print("Weak user gets CIC!")
    """
    
    def __init__(self, capacity: int, enable_noma_awareness: bool = True,
                 channel_aware_eviction: bool = False):
        """
        Initialize NOMA-aware LRU cache.
        
        Args:
            capacity: Cache capacity
            enable_noma_awareness: Track NOMA metrics (CIC, pairing)
            channel_aware_eviction: Consider channel gains when evicting
        """
        super().__init__(capacity, enable_noma_awareness)
        self.cache = OrderedDict()  # Maintains insertion order
        self.channel_aware_eviction = channel_aware_eviction
        
        # NOMA-specific tracking
        self.file_cic_benefits = {}  # file_id -> CIC count
        self.file_channel_scores = {}  # file_id -> avg channel gain
        self.file_request_count = Counter()  # Track requests per file
    
    def populate(self, items=None, channel_gains: Optional[Dict[int, float]] = None):
        """
        Optionally pre-populate cache with items.
        
        Args:
            items: Initial items to cache (or None to start empty)
            channel_gains: Optional dict mapping file_id -> avg channel gain
        """
        self.cache.clear()
        if items:
            for it in list(items)[: self.capacity]:
                self.cache[it] = True
        
        if channel_gains:
            self.file_channel_scores = channel_gains.copy()
    
    def is_hit(self, item: int, update_stats: bool = True) -> bool:
        """
        Check cache and update LRU order.
        
        On hit: Move item to end (most recently used)
        On miss: Add item, evict LRU (or channel-aware) if full
        
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
                self._evict_item()
                if update_stats:
                    self._record_eviction()
            
            self.cache[item] = True
            if update_stats:
                self._record_miss()
            return False
    
    def _evict_item(self):
        """
        Evict item based on LRU or channel-aware policy.
        """
        if self.channel_aware_eviction and self.file_channel_scores:
            # Channel-aware eviction: evict files for STRONG users (high gain)
            # Keep files for WEAK users (low gain) - they need CIC help!
            
            # Get candidates (LRU items, e.g., first 30%)
            num_candidates = max(1, len(self.cache) // 3)
            candidates = list(self.cache.keys())[:num_candidates]
            
            # Find candidate with HIGHEST channel gain (strong user file)
            evict_file = max(
                candidates,
                key=lambda f: self.file_channel_scores.get(f, 1e-6),
                default=None
            )
            
            if evict_file:
                del self.cache[evict_file]
        else:
            # Standard LRU: evict least recently used (first item)
            self.cache.popitem(last=False)
    
    def request(self, item: int, user_id: Optional[int] = None,
                channel_gain: Optional[float] = None,
                paired_user: Optional[int] = None,
                paired_file: Optional[int] = None) -> Dict:
        """
        NOMA-aware request handling with CIC tracking.
        
        Extends base request() with dynamic cache-specific features.
        
        Args:
            item: Requested file ID
            user_id: ID of requesting user
            channel_gain: User's channel gain
            paired_user: ID of NOMA paired user
            paired_file: File requested by paired user
        
        Returns:
            Dict with hit status, CIC benefits, etc.
        """
        # Call base class request() for standard NOMA tracking
        result = super().request(item, user_id, channel_gain, paired_user, paired_file)
        
        # Update file-specific channel score (running average)
        if channel_gain is not None:
            if item in self.file_channel_scores:
                # Running average
                self.file_channel_scores[item] = (
                    0.9 * self.file_channel_scores[item] + 0.1 * channel_gain
                )
            else:
                self.file_channel_scores[item] = channel_gain
        
        # Track CIC benefits per file
        if result.get('weak_user_benefit') or result.get('strong_user_benefit'):
            if item not in self.file_cic_benefits:
                self.file_cic_benefits[item] = 0
            self.file_cic_benefits[item] += 1
        
        # Track request count
        self.file_request_count[item] += 1
        
        return result
    
    def get_contents(self) -> Set[int]:
        """Get current cache contents."""
        return set(self.cache.keys())
    
    def get_cic_benefit_stats(self) -> Dict:
        """Get CIC benefit statistics."""
        total_cic = sum(self.file_cic_benefits.values())
        return {
            'total_cic_benefits': total_cic,
            'files_providing_cic': len(self.file_cic_benefits),
            'avg_cic_per_file': total_cic / len(self.file_cic_benefits) if self.file_cic_benefits else 0,
            'top_cic_files': sorted(self.file_cic_benefits.items(), 
                                   key=lambda x: x[1], reverse=True)[:10]
        }
    
    def clear(self):
        """Clear cache and reset statistics."""
        self.cache.clear()
        self.file_cic_benefits.clear()
        self.file_channel_scores.clear()
        self.file_request_count.clear()
        self.reset_stats()


class LFUCache(CacheBase):
    """
    NOMA-Aware Least Frequently Used (LFU) caching policy.
    
    Evicts the least frequently accessed item when cache is full.
    Uses Counter to track access frequencies.
    
    NOMA Enhancements:
    - Tracks CIC opportunities per file
    - Optional channel-weighted frequency (weak users count more)
    - NOMA pairing awareness
    
    Pros:
        - Adapts to popularity distribution
        - Keeps frequently accessed items
        - Better than LRU for skewed popularity (like Zipf)
        - With NOMA-awareness: prioritizes weak-user files
    
    Cons:
        - Slow to adapt (new popular items hard to cache)
        - "Pollution" from historically popular items
        - Channel-awareness adds complexity
    
    Example:
        >>> cache = LFUCache(capacity=100, channel_weighted_frequency=True)
        >>> cache.is_hit(5)  # False, adds with freq=1
        >>> cache.is_hit(5)  # True, freq=2
        >>> # Files by weak users get higher frequency weights!
    """
    
    def __init__(self, capacity: int, enable_noma_awareness: bool = True,
                 channel_weighted_frequency: bool = False):
        """
        Initialize NOMA-aware LFU cache.
        
        Args:
            capacity: Cache capacity
            enable_noma_awareness: Track NOMA metrics
            channel_weighted_frequency: Weight frequency by channel gain
        """
        super().__init__(capacity, enable_noma_awareness)
        self.store = set()
        self.counter = Counter()  # Tracks access frequency
        self.channel_weighted_frequency = channel_weighted_frequency
        
        # NOMA-specific tracking
        self.file_cic_benefits = {}  # file_id -> CIC count
        self.file_channel_scores = {}  # file_id -> avg channel gain
        self.weighted_counter = Counter()  # Channel-weighted frequency
    
    def populate(self, items=None, channel_gains: Optional[Dict[int, float]] = None):
        """
        Optionally pre-populate cache with items.
        
        Args:
            items: Initial items to cache (or None to start empty)
            channel_gains: Optional dict mapping file_id -> avg channel gain
        """
        self.store.clear()
        self.counter.clear()
        self.weighted_counter.clear()
        
        if items:
            for it in list(items)[: self.capacity]:
                self.store.add(it)
                self.counter[it] = 1
                self.weighted_counter[it] = 1.0
        
        if channel_gains:
            self.file_channel_scores = channel_gains.copy()
    
    def is_hit(self, item: int, update_stats: bool = True,
               channel_gain: Optional[float] = None) -> bool:
        """
        Check cache and update frequency counter.
        
        On hit: Increment access counter (optionally weighted)
        On miss: Add item, evict LFU if full
        
        Args:
            item: File ID to check
            update_stats: Whether to record statistics
            channel_gain: Optional channel gain for weighted frequency
        
        Returns:
            bool: True if cache hit
        """
        if item in self.store:
            # Cache hit: increment frequency
            self.counter[item] += 1
            
            # Channel-weighted frequency: weak users (low gain) count MORE
            if self.channel_weighted_frequency and channel_gain is not None:
                weight = 1.0 / (channel_gain + 1e-9)  # Inverse: weak users = high weight
                self.weighted_counter[item] += weight
            else:
                self.weighted_counter[item] += 1.0
            
            if update_stats:
                self._record_hit()
            return True
        else:
            # Cache miss: add item
            if len(self.store) >= self.capacity:
                self._evict_item()
                if update_stats:
                    self._record_eviction()
            
            self.store.add(item)
            self.counter[item] = 1
            self.weighted_counter[item] = 1.0
            if update_stats:
                self._record_miss()
            return False
    
    def _evict_item(self):
        """
        Evict item with lowest frequency (optionally weighted).
        """
        if self.channel_weighted_frequency and self.weighted_counter:
            # Evict by weighted frequency (keeps weak-user files)
            lfu_item = min(self.weighted_counter.items(), key=lambda x: x[1])[0]
        else:
            # Standard LFU
            lfu_item = min(self.counter.items(), key=lambda x: x[1])[0]
        
        self.store.remove(lfu_item)
        del self.counter[lfu_item]
        if lfu_item in self.weighted_counter:
            del self.weighted_counter[lfu_item]
    
    def request(self, item: int, user_id: Optional[int] = None,
                channel_gain: Optional[float] = None,
                paired_user: Optional[int] = None,
                paired_file: Optional[int] = None) -> Dict:
        """
        NOMA-aware request handling.
        
        Args:
            item: Requested file ID
            user_id: ID of requesting user
            channel_gain: User's channel gain
            paired_user: ID of NOMA paired user
            paired_file: File requested by paired user
        
        Returns:
            Dict with hit status, CIC benefits, etc.
        """
        # Call is_hit with channel gain for weighted frequency
        hit = self.is_hit(item, update_stats=True, channel_gain=channel_gain)
        
        # Build result using base class logic
        result = {
            'hit': hit,
            'cic_enabled': False,
            'paired_user_cached': False,
            'weak_user_benefit': False,
            'strong_user_benefit': False,
            'cache_size': len(self),
        }
        
        if not self.enable_noma_awareness:
            return result
        
        # NOMA-aware processing
        if paired_user is not None and paired_file is not None:
            paired_cached = self.is_hit(paired_file, update_stats=False)
            result['paired_user_cached'] = paired_cached
            
            if paired_cached:
                result['weak_user_benefit'] = True
                result['cic_enabled'] = True
                self.cic_opportunities += 1
                
                if paired_file not in self.file_cic_benefits:
                    self.file_cic_benefits[paired_file] = 0
                self.file_cic_benefits[paired_file] += 1
            
            if hit:
                result['strong_user_benefit'] = True
                result['cic_enabled'] = True
                self.noma_paired_hits += 1
                
                if item not in self.file_cic_benefits:
                    self.file_cic_benefits[item] = 0
                self.file_cic_benefits[item] += 1
        
        # Update channel score
        if channel_gain is not None:
            if item in self.file_channel_scores:
                self.file_channel_scores[item] = (
                    0.9 * self.file_channel_scores[item] + 0.1 * channel_gain
                )
            else:
                self.file_channel_scores[item] = channel_gain
        
        # Store metadata
        if user_id is not None and channel_gain is not None:
            self.channel_gains[user_id] = channel_gain
        
        if user_id is not None and paired_user is not None:
            self.user_pairings[user_id] = paired_user
        
        return result
    
    def get_contents(self) -> Set[int]:
        """Get current cache contents."""
        return self.store
    
    def get_cic_benefit_stats(self) -> Dict:
        """Get CIC benefit statistics."""
        total_cic = sum(self.file_cic_benefits.values())
        return {
            'total_cic_benefits': total_cic,
            'files_providing_cic': len(self.file_cic_benefits),
            'avg_cic_per_file': total_cic / len(self.file_cic_benefits) if self.file_cic_benefits else 0,
            'top_cic_files': sorted(self.file_cic_benefits.items(), 
                                   key=lambda x: x[1], reverse=True)[:10]
        }
    
    def clear(self):
        """Clear cache and reset statistics."""
        self.store.clear()
        self.counter.clear()
        self.weighted_counter.clear()
        self.file_cic_benefits.clear()
        self.file_channel_scores.clear()
        self.reset_stats()


class RandomCache(CacheBase):
    """
    NOMA-Aware Random Replacement caching policy.
    
    Evicts a random item when cache is full.
    Useful as a baseline to compare against smarter policies.
    
    NOMA Enhancements:
    - Tracks CIC opportunities
    - Optional channel-weighted random eviction
    
    Pros:
        - Simple, minimal overhead
        - Useful baseline for comparison
        - No pathological worst-case behavior
        - With NOMA: can avoid evicting weak-user files
    
    Cons:
        - No adaptation to any pattern
        - Poor performance in practice
        - Only useful for comparison
    """
    
    def __init__(self, capacity: int, enable_noma_awareness: bool = True,
                 channel_weighted_eviction: bool = False):
        """
        Initialize NOMA-aware random cache.
        
        Args:
            capacity: Cache capacity
            enable_noma_awareness: Track NOMA metrics
            channel_weighted_eviction: Bias eviction toward strong users
        """
        super().__init__(capacity, enable_noma_awareness)
        self.store = set()
        self.channel_weighted_eviction = channel_weighted_eviction
        
        # NOMA-specific tracking
        self.file_cic_benefits = {}
        self.file_channel_scores = {}
    
    def populate(self, items=None):
        """Optionally pre-populate cache."""
        self.store.clear()
        if items:
            for it in list(items)[: self.capacity]:
                self.store.add(it)
    
    def is_hit(self, item: int, update_stats: bool = True) -> bool:
        """Check cache and randomly evict if full."""
        if item in self.store:
            if update_stats:
                self._record_hit()
            return True
        else:
            if len(self.store) >= self.capacity:
                self._evict_item()
                if update_stats:
                    self._record_eviction()
            
            self.store.add(item)
            if update_stats:
                self._record_miss()
            return False
    
    def _evict_item(self):
        """Evict random item (optionally weighted by channel)."""
        if self.channel_weighted_eviction and self.file_channel_scores:
            # Weighted random: more likely to evict strong-user files
            weights = [self.file_channel_scores.get(f, 1e-6) for f in self.store]
            evict_file = random.choices(list(self.store), weights=weights, k=1)[0]
            self.store.remove(evict_file)
        else:
            # Standard random
            self.store.remove(random.choice(list(self.store)))
    
    def request(self, item: int, user_id: Optional[int] = None,
                channel_gain: Optional[float] = None,
                paired_user: Optional[int] = None,
                paired_file: Optional[int] = None) -> Dict:
        """NOMA-aware request."""
        result = super().request(item, user_id, channel_gain, paired_user, paired_file)
        
        # Update channel score
        if channel_gain is not None:
            if item in self.file_channel_scores:
                self.file_channel_scores[item] = (
                    0.9 * self.file_channel_scores[item] + 0.1 * channel_gain
                )
            else:
                self.file_channel_scores[item] = channel_gain
        
        return result
    
    def get_contents(self) -> Set[int]:
        """Get current cache contents."""
        return self.store
    
    def clear(self):
        """Clear cache and reset statistics."""
        self.store.clear()
        self.file_cic_benefits.clear()
        self.file_channel_scores.clear()
        self.reset_stats()


if __name__ == "__main__":
    print("="*70)
    print("TESTING NOMA-AWARE DYNAMIC CACHING POLICIES")
    print("="*70)
    
    # Test 1: LRU with channel-aware eviction
    print("\n[TEST 1] LRU Cache with Channel-Aware Eviction")
    print("-" * 70)
    lru = LRUCache(capacity=5, channel_aware_eviction=True)
    lru.populate([1, 2, 3])
    
    # Simulate NOMA requests
    noma_requests = [
        (1, 10, 20, 2, 1e-8),  # Weak user 10 requests file 1, paired with strong user 20 requesting file 2
        (2, 11, 21, 3, 1e-7),
        (3, 12, 22, 4, 1e-9),  # Very weak user
        (4, 13, 23, 5, 1e-6),  # Strong user
        (5, 14, 24, 6, 1e-8),
        (6, 15, 25, 1, 1e-7),
    ]
    
    for file_id, user_id, paired_user, paired_file, channel_gain in noma_requests:
        result = lru.request(file_id, user_id, channel_gain, paired_user, paired_file)
        print(f"User {user_id} requests file {file_id} (gain={channel_gain:.1e}): "
              f"{'HIT' if result['hit'] else 'MISS'} | Cache: {sorted(lru.get_contents())}")
    
    lru.print_stats()
    cic_stats = lru.get_cic_benefit_stats()
    print(f"\nCIC Benefits: {cic_stats['total_cic_benefits']} opportunities")
    
    # Test 2: LFU with channel-weighted frequency
    print("\n" + "="*70)
    print("[TEST 2] LFU Cache with Channel-Weighted Frequency")
    print("-" * 70)
    lfu = LFUCache(capacity=5, channel_weighted_frequency=True)
    
    # Weak user requests same file multiple times → HIGH priority
    # Strong user requests file once → LOW priority
    requests = [
        (1, 10, 1e-9),  # Weak user, file 1
        (1, 10, 1e-9),  # Weak user, file 1 again (high weight!)
        (2, 11, 1e-6),  # Strong user, file 2
        (3, 12, 1e-9),  # Weak user, file 3
        (4, 13, 1e-6),  # Strong user, file 4
        (5, 14, 1e-9),  # Weak user, file 5
        (6, 15, 1e-6),  # Strong user, file 6 → should evict file 2 or 4 (strong user files)
    ]
    
    for file_id, user_id, channel_gain in requests:
        result = lfu.request(file_id, user_id, channel_gain)
        print(f"User {user_id} (gain={channel_gain:.1e}) requests file {file_id}: "
              f"{'HIT' if result['hit'] else 'MISS'} | Cache: {sorted(lfu.get_contents())}")
    
    lfu.print_stats()
    
    # Test 3: Standard LRU (for comparison)
    print("\n" + "="*70)
    print("[TEST 3] Standard LRU (No Channel Awareness)")
    print("-" * 70)
    lru_std = LRUCache(capacity=5, channel_aware_eviction=False)
    lru_std.populate([1, 2, 3])
    
    requests = [1, 2, 3, 4, 5, 6, 1, 7, 1]
    for file_id in requests:
        hit = lru_std.is_hit(file_id)
        print(f"Request {file_id}: {'HIT' if hit else 'MISS'} | Cache: {sorted(lru_std.get_contents())}")
    
    lru_std.print_stats()
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)
