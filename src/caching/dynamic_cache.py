# src/caching/dynamic_cache.py
import random
from collections import OrderedDict, Counter

class BaseCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.store = set()
        self.hits = 0
        self.requests = 0

    def __contains__(self, item):
        """Allow using `if item in cache` for membership check only (no stats)."""
        return item in self.store

    def is_hit(self, item):
        """Check + update hit statistics (used internally)."""
        self.requests += 1
        if item in self.store:
            self.hits += 1
            return True
        return False

    def hit_rate(self):
        return self.hits / self.requests if self.requests > 0 else 0


class LRUCache(BaseCache):
    def __init__(self, capacity):
        super().__init__(capacity)
        self.cache = OrderedDict()

    def request(self, item):
        """Insert/update item into cache (after miss)."""
        if item in self.cache:
            self.cache.move_to_end(item)  # mark as recently used
        else:
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)  # remove least recently used
            self.cache[item] = True
            self.store = set(self.cache.keys())


class LFUCache(BaseCache):
    def __init__(self, capacity):
        super().__init__(capacity)
        self.counter = Counter()

    def request(self, item):
        """Insert/update item into cache (after miss)."""
        if item in self.store:
            self.counter[item] += 1
        else:
            if len(self.store) >= self.capacity:
                # evict least frequently used
                lfu_item, _ = min(self.counter.items(), key=lambda x: x[1])
                self.store.remove(lfu_item)
                del self.counter[lfu_item]
            self.store.add(item)
            self.counter[item] = 1


class RandomCache(BaseCache):
    def __init__(self, capacity):
        super().__init__(capacity)

    def request(self, item):
        """Insert item into cache (after miss)."""
        if item not in self.store:
            if len(self.store) >= self.capacity:
                self.store.remove(random.choice(list(self.store)))
            self.store.add(item)

