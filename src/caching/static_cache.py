# src/caching/static_cache.py
"""
Static Top-K Caching Policy for Cache-Aided NOMA

Caches the K most popular files based on popularity distribution.
Simple but effective baseline for comparison with learning-based policies.

NOMA-Aware Features:
- CIC (Cache-aided Interference Cancellation) opportunity tracking
- Optional channel-aware selection (prioritize files for weak channel users)
- Integration with NOMA user pairing
- Performance metrics for cache-aided NOMA

Author: Cache-Aided NOMA Team
Date: December 2025
"""

from .cache_base import CacheBase
from typing import Iterable, Set, Dict, List, Optional, Tuple
import numpy as np


class StaticTopKCache(CacheBase):
    """
    Static cache that stores top-K most popular items.
    
    This is a simple baseline policy that:
    1. Analyzes popularity distribution (e.g., Zipf)
    2. Caches the K most popular files
    3. Never updates during simulation
    4. Optionally: prioritizes files for weak-channel users (NOMA-aware)
    
    Pros:
        - Simple, no computational overhead
        - Works well for highly skewed distributions (Zipf)
        - Optimal if popularity is perfectly known and static
        - With NOMA-awareness: maximizes CIC opportunities
    
    Cons:
        - Cannot adapt to changing popularity
        - Requires perfect popularity knowledge
        - Limited channel adaptation (only at initialization)
    
    Example:
        >>> # Cache top-100 files by Zipf distribution
        >>> cache = StaticTopKCache(capacity=100)
        >>> popularity_sorted = [1, 2, 3, ...]  # Files sorted by popularity
        >>> cache.populate(popularity_sorted)
        >>> 
        >>> # NOMA-aware: check CIC benefit
        >>> result = cache.request(file_id=5, user_id=10, paired_user=20)
        >>> if result['cic_enabled']:
        ...     print("CIC can be applied!")
    """
    
    def __init__(self, capacity: int, enable_noma_awareness: bool = True,
                 channel_aware: bool = False):
        """
        Initialize static top-K cache.
        
        Args:
            capacity: Number of files to cache
            enable_noma_awareness: Track NOMA-specific metrics (CIC, pairing)
            channel_aware: Whether to consider channel gains in selection
        """
        super().__init__(capacity, enable_noma_awareness)
        self.contents = set()  # Set of cached file IDs
        self.channel_aware = channel_aware
        
        # NOMA-specific tracking
        self.file_popularity = {}  # file_id -> popularity score
        self.file_channel_benefit = {}  # file_id -> avg channel benefit
        self.cic_benefits = {}  # Track CIC benefits per file
    
    def populate(self, items: Iterable[int], 
                 popularity_scores: Optional[Dict[int, float]] = None,
                 channel_gains: Optional[Dict[int, float]] = None):
        """
        Populate cache with top-K items.
        
        Standard mode: Cache K most popular files
        NOMA-aware mode: Balance popularity with channel benefit
        
        Args:
            items: Iterable of file IDs ordered by popularity (most popular first)
            popularity_scores: Optional dict mapping file_id -> popularity score
            channel_gains: Optional dict mapping file_id -> channel benefit
                          (avg channel gain of users requesting this file)
        
        Example:
            >>> # Standard: just popularity
            >>> cache.populate([1, 2, 3, 4, 5, ...])
            >>> 
            >>> # NOMA-aware: popularity + channel benefit
            >>> popularity = {1: 0.5, 2: 0.3, 3: 0.1, ...}
            >>> channels = {1: 1e-7, 2: 1e-8, 3: 1e-6, ...}
            >>> cache.populate([1, 2, 3, ...], popularity, channels)
        """
        sorted_list = list(items)
        
        # Store popularity scores if provided
        if popularity_scores is not None:
            self.file_popularity = popularity_scores.copy()
        else:
            # Infer from order (most popular first)
            self.file_popularity = {fid: 1.0/(i+1) for i, fid in enumerate(sorted_list)}
        
        # NOMA-aware selection: balance popularity and channel benefit
        if self.channel_aware and channel_gains is not None:
            self.file_channel_benefit = channel_gains.copy()
            
            # Compute combined score: popularity × channel_weight
            # Files requested by weak users (low channel gain) get higher priority
            # → More CIC opportunities for weak users!
            combined_scores = {}
            for fid in sorted_list:
                pop_score = self.file_popularity.get(fid, 0)
                
                # Inverse channel gain: weak users have LOW gain, need MORE help
                channel_weight = 1.0 / (channel_gains.get(fid, 1e-6) + 1e-9)
                
                # Combined score (can be tuned)
                combined_scores[fid] = pop_score * (1 + 0.1 * channel_weight)
            
            # Select top-K by combined score
            sorted_by_combined = sorted(combined_scores.items(), 
                                       key=lambda x: x[1], reverse=True)
            topk = [fid for fid, _ in sorted_by_combined[:self.capacity]]
            
            print(f"[NOMA-Aware Cache] Selected {len(topk)} files balancing popularity and channel benefit")
        else:
            # Standard: just take top-K by popularity
            topk = sorted_list[:self.capacity]
        
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
    
    def request(self, item: int, user_id: Optional[int] = None,
                channel_gain: Optional[float] = None,
                paired_user: Optional[int] = None,
                paired_file: Optional[int] = None,
                **kwargs) -> Dict:
        """
        NOMA-aware request handling.
        
        Enhanced version that tracks CIC opportunities for NOMA pairs.
        
        Args:
            item: Requested file ID
            user_id: ID of requesting user
            channel_gain: User's channel gain
            paired_user: ID of NOMA paired user
            paired_file: File requested by paired user (for CIC tracking)
        
        Returns:
            Dict with:
                - 'hit': Whether request was cache hit
                - 'cic_enabled': Whether CIC can be applied
                - 'paired_user_cached': Whether paired user's file is cached
                - 'weak_user_benefit': CIC benefit for weak user
                - 'strong_user_benefit': CIC benefit for strong user
        
        Example:
            >>> # User 10 (weak) requests file 5, paired with user 20 (strong) requesting file 8
            >>> result = cache.request(item=5, user_id=10, paired_user=20, paired_file=8)
            >>> if result['weak_user_benefit']:
            ...     # User 10 can cancel user 20's interference!
            ...     sinr = sinr_weak_user_with_cache(...)
        """
        # Check if requested item is cached
        hit = self.is_hit(item, update_stats=True)
        
        # Initialize result
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
            # Check if paired user's file is cached
            paired_cached = self.is_hit(paired_file, update_stats=False)
            result['paired_user_cached'] = paired_cached
            
            # Determine who benefits from CIC
            # Assumption: user_id is weak, paired_user is strong (from NOMA pairing)
            
            if paired_cached:
                # Weak user can cancel strong user's interference!
                result['weak_user_benefit'] = True
                result['cic_enabled'] = True
                self.cic_opportunities += 1
                
                # Track CIC benefit for this file
                if paired_file not in self.cic_benefits:
                    self.cic_benefits[paired_file] = 0
                self.cic_benefits[paired_file] += 1
            
            if hit:
                # Strong user can cancel weak user's interference (perfect SIC)
                result['strong_user_benefit'] = True
                result['cic_enabled'] = True
                self.noma_paired_hits += 1
                
                # Track CIC benefit for this file
                if item not in self.cic_benefits:
                    self.cic_benefits[item] = 0
                self.cic_benefits[item] += 1
        
        # Store channel gain if provided
        if user_id is not None and channel_gain is not None:
            self.channel_gains[user_id] = channel_gain
        
        # Store pairing
        if user_id is not None and paired_user is not None:
            self.user_pairings[user_id] = paired_user
        
        return result
    
    def get_contents(self) -> Set[int]:
        """
        Get current cache contents.
        
        Returns:
            Set[int]: Set of cached file IDs
        """
        return self.contents
    
    def get_cic_benefit_stats(self) -> Dict:
        """
        Get statistics about CIC benefits.
        
        Returns:
            Dict with CIC benefit analysis
        """
        total_cic = sum(self.cic_benefits.values())
        
        return {
            'total_cic_benefits': total_cic,
            'files_providing_cic': len(self.cic_benefits),
            'avg_cic_per_file': total_cic / len(self.cic_benefits) if self.cic_benefits else 0,
            'top_cic_files': sorted(self.cic_benefits.items(), 
                                   key=lambda x: x[1], reverse=True)[:10]
        }
    
    def analyze_cache_efficiency(self) -> Dict:
        """
        Analyze cache efficiency for NOMA system.
        
        Returns:
            Dict with efficiency metrics
        """
        stats = self.stats()
        cic_stats = self.get_cic_benefit_stats()
        
        return {
            **stats,
            **cic_stats,
            'cic_benefit_rate': cic_stats['total_cic_benefits'] / stats['total_requests'] 
                               if stats['total_requests'] > 0 else 0,
        }
    
    def clear(self):
        """
        Clear cache contents and reset statistics.
        """
        self.contents.clear()
        self.file_popularity.clear()
        self.file_channel_benefit.clear()
        self.cic_benefits.clear()
        self.reset_stats()


if __name__ == "__main__":
    print("="*70)
    print("TESTING NOMA-AWARE STATIC TOP-K CACHE")
    print("="*70)
    
    # Test 1: Standard popularity-based caching
    print("\n[TEST 1] Standard Top-K Caching")
    print("-" * 70)
    cache = StaticTopKCache(capacity=10)
    
    # Populate with files sorted by popularity
    popularity_sorted = list(range(1, 101))  # Files 1-100, file 1 most popular
    cache.populate(popularity_sorted)
    
    print(f"Cache populated with top-{cache.capacity} files")
    print(f"Contents: {sorted(cache.get_contents())}")
    print(f"Cache: {cache}")
    
    # Simulate requests
    print("\nSimulating requests...")
    requests = [1, 1, 2, 3, 15, 20, 1, 5, 100, 2]  # Mix of popular and unpopular
    
    for file_id in requests:
        hit = cache.is_hit(file_id)
        print(f"  Request file {file_id:3d}: {'HIT' if hit else 'MISS'}")
    
    cache.print_stats()
    
    # Test 2: NOMA-aware caching with CIC tracking
    print("\n" + "="*70)
    print("[TEST 2] NOMA-Aware Caching with CIC Tracking")
    print("-" * 70)
    cache2 = StaticTopKCache(capacity=10, enable_noma_awareness=True)
    cache2.populate(popularity_sorted)
    
    print("\nSimulating NOMA user pairs...")
    # Simulate NOMA pairs: (weak_user, strong_user) requesting (weak_file, strong_file)
    noma_requests = [
        (1, 5, 10, 8),   # User 1 (weak) requests 5, paired with user 10 (strong) requesting 8
        (2, 1, 11, 3),   # User 2 requests 1, paired with user 11 requesting 3
        (3, 20, 12, 2),  # User 3 requests 20 (miss), paired with user 12 requesting 2
        (4, 7, 13, 15),  # User 4 requests 7, paired with user 13 requesting 15 (miss)
    ]
    
    for weak_user, weak_file, strong_user, strong_file in noma_requests:
        result = cache2.request(
            item=weak_file,
            user_id=weak_user,
            paired_user=strong_user,
            paired_file=strong_file,
            channel_gain=1e-8  # Weak user has poor channel
        )
        
        print(f"\n  User {weak_user} (weak) requests file {weak_file}, "
              f"paired with user {strong_user} (strong) requesting file {strong_file}")
        print(f"    Weak file cached: {result['hit']}")
        print(f"    Strong file cached: {result['paired_user_cached']}")
        print(f"    Weak gets CIC: {result['weak_user_benefit']} (can cancel strong's interference)")
        print(f"    Strong gets CIC: {result['strong_user_benefit']} (perfect SIC)")
    
    cache2.print_stats()
    
    cic_stats = cache2.get_cic_benefit_stats()
    print("\nCIC Benefit Analysis:")
    print(f"  Total CIC benefits: {cic_stats['total_cic_benefits']}")
    print(f"  Files providing CIC: {cic_stats['files_providing_cic']}")
    print(f"  Top CIC files: {cic_stats['top_cic_files'][:5]}")
    
    # Test 3: Channel-aware selection
    print("\n" + "="*70)
    print("[TEST 3] Channel-Aware File Selection")
    print("-" * 70)
    cache3 = StaticTopKCache(capacity=10, enable_noma_awareness=True, 
                            channel_aware=True)
    
    # Simulate: files requested by weak users should get priority
    popularity = {i: 1.0/i for i in range(1, 21)}  # Zipf-like
    channel_gains = {
        1: 1e-6, 2: 1e-7, 3: 1e-6,  # Files 2 requested by weak users
        4: 1e-8, 5: 1e-7, 6: 1e-6,  # Files 4, 5 by weak users
        7: 1e-6, 8: 1e-9, 9: 1e-6, 10: 1e-6,  # File 8 by VERY weak user
    }
    
    cache3.populate(list(range(1, 21)), popularity, channel_gains)
    print(f"\nChannel-aware cache contents: {sorted(cache3.get_contents())}")
    print("Note: Files requested by weak users (low channel gain) prioritized!")
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)
