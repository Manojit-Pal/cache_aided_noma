# src/caching/static_cache.py
from .cache_base import CacheBase
from typing import Iterable
import numpy as np

class StaticTopKCache(CacheBase):
    """
    Static cache that stores top-K most popular items according to observed
    popularity or a provided ranking list.
    """
    def __init__(self, capacity: int):
        super().__init__(capacity)
        self.contents = set()

    def populate(self, items: Iterable[int]):
        """
        items: iterable ordered by popularity (most popular first)
        We store up to capacity.
        """
        sorted_list = list(items)
        topk = sorted_list[: self.capacity]
        self.contents = set(topk)

    def is_hit(self, item: int) -> bool:
        return int(item) in self.contents

    def clear(self):
        self.contents.clear()

