"""
src/caching/dynamic_cache.py

NOMA-Aware Dynamic Caching Policies for Cache-Aided NOMA

Bug Fix History:
- BUG-CACHE-1 (CRITICAL): LRU/LFU/Random is_hit() inserted the item on every
  miss regardless of update_stats flag. CIC peek calls silently mutated cache.
  FIX: Pure read-only when update_stats=False; insert+evict only on True.

- BUG-CACHE-2 (MEDIUM): LFUCache.get_contents() / RandomCache.get_contents()
  returned self.store directly (mutable internal set).
  FIX: return set(self.store) — defensive copy.

- BUG-DYN-1 (CRITICAL): LFUCache._evict_item() searched self.counter /
  self.weighted_counter which may contain stale keys for files no longer
  in self.store. min() picked a ghost key, store.discard() was a no-op,
  new file was added on top -> silent cache overflow (len > capacity).
  FIX: filter candidate dict to {k:v for k,v if k in self.store} before
  calling min().

Author: Cache-Aided NOMA Team
Date: December 2025 | Revised: March 2026
"""

import random
from collections import OrderedDict, Counter
from typing import Set, Dict, Optional
from .cache_base import CacheBase


class LRUCache(CacheBase):
    """
    NOMA-Aware Least Recently Used (LRU) caching policy.
    Uses OrderedDict for O(1) access and move-to-end operations.
    """

    def __init__(self, capacity: int, enable_noma_awareness: bool = True,
                 channel_aware_eviction: bool = False):
        super().__init__(capacity, enable_noma_awareness)
        self.cache = OrderedDict()
        self.channel_aware_eviction = channel_aware_eviction
        self.file_cic_benefits = {}
        self.file_channel_scores = {}
        self.file_request_count = Counter()

    def populate(self, items=None, channel_gains: Optional[Dict[int, float]] = None):
        self.cache.clear()
        if items:
            for it in list(items)[: self.capacity]:
                self.cache[it] = True
        if channel_gains:
            self.file_channel_scores = channel_gains.copy()

    def is_hit(self, item: int, update_stats: bool = True) -> bool:
        """
        BUG-CACHE-1 FIX: update_stats=False is a pure read-only peek.
        No insert/evict happens unless update_stats=True.
        """
        if item in self.cache:
            if update_stats:
                self.cache.move_to_end(item)
                self._record_hit()
            return True
        else:
            if not update_stats:
                return False
            if len(self.cache) >= self.capacity:
                self._evict_item()
                self._record_eviction()
            self.cache[item] = True
            self._record_miss()
            return False

    def _evict_item(self):
        if self.channel_aware_eviction and self.file_channel_scores:
            # Evict from the LRU end (front of OrderedDict), highest channel gain
            num_candidates = max(1, len(self.cache) // 3)
            candidates = list(self.cache.keys())[:num_candidates]
            evict_file = max(
                candidates,
                key=lambda f: self.file_channel_scores.get(f, 1e-6)
            )
            del self.cache[evict_file]
        else:
            self.cache.popitem(last=False)

    def request(self, item: int, user_id: Optional[int] = None,
                channel_gain: Optional[float] = None,
                paired_user: Optional[int] = None,
                paired_file: Optional[int] = None,
                **kwargs) -> Dict:
        result = super().request(item, user_id, channel_gain, paired_user, paired_file)
        if channel_gain is not None:
            self.file_channel_scores[item] = (
                0.9 * self.file_channel_scores.get(item, channel_gain)
                + 0.1 * channel_gain
            )
        if result.get('weak_user_benefit') or result.get('strong_user_benefit'):
            self.file_cic_benefits[item] = self.file_cic_benefits.get(item, 0) + 1
        self.file_request_count[item] += 1
        return result

    def get_contents(self) -> Set[int]:
        return set(self.cache.keys())

    def get_cic_benefit_stats(self) -> Dict:
        total = sum(self.file_cic_benefits.values())
        return {
            'total_cic_benefits':  total,
            'files_providing_cic': len(self.file_cic_benefits),
            'avg_cic_per_file':    total / len(self.file_cic_benefits) if self.file_cic_benefits else 0,
            'top_cic_files':       sorted(self.file_cic_benefits.items(),
                                          key=lambda x: x[1], reverse=True)[:10],
        }

    def clear(self):
        self.cache.clear()
        self.file_cic_benefits.clear()
        self.file_channel_scores.clear()
        self.file_request_count.clear()
        self.reset_stats()


class LFUCache(CacheBase):
    """
    NOMA-Aware Least Frequently Used (LFU) caching policy.

    BUG-DYN-1 FIX: _evict_item() now filters counter/weighted_counter
    to only live keys (present in self.store) before calling min().
    This prevents stale ghost keys from causing silent cache overflow.
    """

    def __init__(self, capacity: int, enable_noma_awareness: bool = True,
                 channel_weighted_frequency: bool = False):
        super().__init__(capacity, enable_noma_awareness)
        self.store = set()
        self.counter = Counter()
        self.channel_weighted_frequency = channel_weighted_frequency
        self.file_cic_benefits = {}
        self.file_channel_scores = {}
        self.weighted_counter = Counter()

    def populate(self, items=None, channel_gains: Optional[Dict[int, float]] = None):
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
        BUG-CACHE-1 FIX: update_stats=False is a pure read-only peek.
        """
        if item in self.store:
            if update_stats:
                self.counter[item] += 1
                if self.channel_weighted_frequency and channel_gain is not None:
                    self.weighted_counter[item] += 1.0 / (channel_gain + 1e-9)
                else:
                    self.weighted_counter[item] += 1.0
                self._record_hit()
            return True
        else:
            if not update_stats:
                return False
            if len(self.store) >= self.capacity:
                self._evict_item()
                self._record_eviction()
            self.store.add(item)
            self.counter[item] = 1
            self.weighted_counter[item] = (
                1.0 / (channel_gain + 1e-9)
                if (self.channel_weighted_frequency and channel_gain is not None)
                else 1.0
            )
            self._record_miss()
            return False

    def _evict_item(self):
        """
        BUG-DYN-1 FIX: only consider keys that exist in self.store.
        Stale keys in counter/weighted_counter (from prior episodes without
        a full clear()) could cause min() to pick a ghost, making
        store.discard() a no-op and growing len(store) beyond capacity.
        """
        if self.channel_weighted_frequency and self.weighted_counter:
            live = {k: v for k, v in self.weighted_counter.items() if k in self.store}
            if not live:
                # Fallback: evict arbitrary element
                lfu_item = next(iter(self.store))
            else:
                lfu_item = min(live, key=live.get)
        else:
            live = {k: v for k, v in self.counter.items() if k in self.store}
            if not live:
                lfu_item = next(iter(self.store))
            else:
                lfu_item = min(live, key=live.get)

        self.store.discard(lfu_item)
        self.counter.pop(lfu_item, None)
        self.weighted_counter.pop(lfu_item, None)

    def request(self, item: int, user_id: Optional[int] = None,
                channel_gain: Optional[float] = None,
                paired_user: Optional[int] = None,
                paired_file: Optional[int] = None,
                **kwargs) -> Dict:
        hit = self.is_hit(item, update_stats=True, channel_gain=channel_gain)
        result = {
            'hit':                 hit,
            'cic_enabled':         False,
            'paired_user_cached':  False,
            'weak_user_benefit':   False,
            'strong_user_benefit': False,
            'cache_size':          len(self),
        }
        if not self.enable_noma_awareness:
            return result
        if paired_user is not None and paired_file is not None:
            paired_cached = self.is_hit(paired_file, update_stats=False)
            result['paired_user_cached'] = paired_cached
            if paired_cached:
                result['weak_user_benefit'] = True
                result['cic_enabled']       = True
                self.cic_opportunities     += 1
                self.file_cic_benefits[paired_file] = self.file_cic_benefits.get(paired_file, 0) + 1
            if hit:
                result['strong_user_benefit'] = True
                result['cic_enabled']         = True
                self.noma_paired_hits        += 1
                self.file_cic_benefits[item]  = self.file_cic_benefits.get(item, 0) + 1
        if channel_gain is not None:
            self.file_channel_scores[item] = (
                0.9 * self.file_channel_scores.get(item, channel_gain)
                + 0.1 * channel_gain
            )
        if user_id is not None and channel_gain is not None:
            self.channel_gains[user_id] = channel_gain
        if user_id is not None and paired_user is not None:
            self.user_pairings[user_id] = paired_user
        return result

    def get_contents(self) -> Set[int]:
        """BUG-CACHE-2 FIX: defensive copy."""
        return set(self.store)

    def get_cic_benefit_stats(self) -> Dict:
        total = sum(self.file_cic_benefits.values())
        return {
            'total_cic_benefits':  total,
            'files_providing_cic': len(self.file_cic_benefits),
            'avg_cic_per_file':    total / len(self.file_cic_benefits) if self.file_cic_benefits else 0,
            'top_cic_files':       sorted(self.file_cic_benefits.items(),
                                          key=lambda x: x[1], reverse=True)[:10],
        }

    def clear(self):
        self.store.clear()
        self.counter.clear()
        self.weighted_counter.clear()
        self.file_cic_benefits.clear()
        self.file_channel_scores.clear()
        self.reset_stats()


class RandomCache(CacheBase):
    """
    NOMA-Aware Random Replacement caching policy.
    """

    def __init__(self, capacity: int, enable_noma_awareness: bool = True,
                 channel_weighted_eviction: bool = False):
        super().__init__(capacity, enable_noma_awareness)
        self.store = set()
        self.channel_weighted_eviction = channel_weighted_eviction
        self.file_cic_benefits = {}
        self.file_channel_scores = {}

    def populate(self, items=None):
        self.store.clear()
        if items:
            for it in list(items)[: self.capacity]:
                self.store.add(it)

    def is_hit(self, item: int, update_stats: bool = True) -> bool:
        """
        BUG-CACHE-1 FIX: update_stats=False is a pure read-only peek.
        """
        if item in self.store:
            if update_stats:
                self._record_hit()
            return True
        else:
            if not update_stats:
                return False
            if len(self.store) >= self.capacity:
                self._evict_item()
                self._record_eviction()
            self.store.add(item)
            self._record_miss()
            return False

    def _evict_item(self):
        if self.channel_weighted_eviction and self.file_channel_scores:
            weights = [self.file_channel_scores.get(f, 1e-6) for f in self.store]
            evict_file = random.choices(list(self.store), weights=weights, k=1)[0]
            self.store.remove(evict_file)
        else:
            self.store.remove(random.choice(list(self.store)))

    def request(self, item: int, user_id: Optional[int] = None,
                channel_gain: Optional[float] = None,
                paired_user: Optional[int] = None,
                paired_file: Optional[int] = None,
                **kwargs) -> Dict:
        result = super().request(item, user_id, channel_gain, paired_user, paired_file)
        if channel_gain is not None:
            self.file_channel_scores[item] = (
                0.9 * self.file_channel_scores.get(item, channel_gain)
                + 0.1 * channel_gain
            )
        return result

    def get_contents(self) -> Set[int]:
        """BUG-CACHE-2 FIX: defensive copy."""
        return set(self.store)

    def clear(self):
        self.store.clear()
        self.file_cic_benefits.clear()
        self.file_channel_scores.clear()
        self.reset_stats()


if __name__ == '__main__':
    print('=' * 70)
    print('TESTING NOMA-AWARE DYNAMIC CACHING POLICIES')
    print('=' * 70)

    print('\n[TEST 1] LRU peek must NOT insert')
    lru = LRUCache(capacity=3)
    lru.populate([1, 2, 3])
    before = lru.get_contents()
    lru.is_hit(99, update_stats=False)
    assert before == lru.get_contents(), 'LRU mutated on peek!'
    print(f'  PASS: {lru.get_contents()}')

    print('\n[TEST 2] LFU peek must NOT insert')
    lfu = LFUCache(capacity=3)
    lfu.populate([1, 2, 3])
    before = lfu.get_contents()
    lfu.is_hit(99, update_stats=False)
    assert before == lfu.get_contents(), 'LFU mutated on peek!'
    print(f'  PASS: {lfu.get_contents()}')

    print('\n[TEST 3] LFU _evict_item stale-key safety (BUG-DYN-1)')
    lfu2 = LFUCache(capacity=2, channel_weighted_frequency=True)
    lfu2.store = {10, 20}
    lfu2.counter = Counter({10: 5, 20: 3, 99: 100})  # 99 is stale
    lfu2.weighted_counter = Counter({10: 5.0, 20: 3.0, 99: 999.0})  # ghost
    lfu2._evict_item()
    assert 99 not in lfu2.store, 'Ghost eviction!'
    assert len(lfu2.store) == 1, f'Expected 1 item, got {len(lfu2.store)}'
    print(f'  PASS: evicted live item, store={lfu2.store}')

    print('\n[TEST 4] LFU get_contents() defensive copy')
    lfu3 = LFUCache(capacity=3)
    lfu3.populate([10, 20, 30])
    c = lfu3.get_contents()
    c.add(999)
    assert 999 not in lfu3.store, 'Internal store corrupted!'
    print(f'  PASS')

    print('\n[TEST 5] Random peek must NOT insert')
    rnd = RandomCache(capacity=3)
    rnd.populate([1, 2, 3])
    before = rnd.get_contents()
    rnd.is_hit(99, update_stats=False)
    assert before == rnd.get_contents(), 'Random mutated on peek!'
    print(f'  PASS: {rnd.get_contents()}')

    print('\n' + '=' * 70)
    print('ALL TESTS PASSED')
    print('=' * 70)
