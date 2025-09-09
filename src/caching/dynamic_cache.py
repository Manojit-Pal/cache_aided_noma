# src/caching/dynamic_cache.py
import random
from collections import OrderedDict, Counter
from .cache_base import CacheBase


class LRUCache(CacheBase):
    def __init__(self, capacity: int):
        super().__init__(capacity)
        self.cache = OrderedDict()

    def populate(self, items=None):
        self.cache.clear()
        if items:
            for it in items[:self.capacity]:
                self.cache[it] = True

    def is_hit(self, item: int) -> bool:
        if item in self.cache:
            self.cache.move_to_end(item)  # refresh recency
            return True
        else:
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
            self.cache[item] = True
            return False

    def clear(self):
        self.cache.clear()


class LFUCache(CacheBase):
    def __init__(self, capacity: int):
        super().__init__(capacity)
        self.store = set()
        self.counter = Counter()

    def populate(self, items=None):
        self.store.clear()
        self.counter.clear()
        if items:
            for it in items[:self.capacity]:
                self.store.add(it)
                self.counter[it] = 1

    def is_hit(self, item: int) -> bool:
        if item in self.store:
            self.counter[item] += 1
            return True
        else:
            if len(self.store) >= self.capacity:
                lfu_item, _ = min(self.counter.items(), key=lambda x: x[1])
                self.store.remove(lfu_item)
                del self.counter[lfu_item]
            self.store.add(item)
            self.counter[item] = 1
            return False

    def clear(self):
        self.store.clear()
        self.counter.clear()


class RandomCache(CacheBase):
    def __init__(self, capacity: int):
        super().__init__(capacity)
        self.store = set()

    def populate(self, items=None):
        self.store.clear()
        if items:
            for it in items[:self.capacity]:
                self.store.add(it)

    def is_hit(self, item: int) -> bool:
        if item in self.store:
            return True
        else:
            if len(self.store) >= self.capacity:
                self.store.remove(random.choice(list(self.store)))
            self.store.add(item)
            return False

    def clear(self):
        self.store.clear()
