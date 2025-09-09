# src/caching/cache_base.py
from abc import ABC, abstractmethod
from typing import Iterable

class CacheBase(ABC):
    def __init__(self, capacity: int):
        self.capacity = capacity

    @abstractmethod
    def populate(self, items: Iterable[int]):
        """Populate cache with items (called in placement phase)."""
        pass

    @abstractmethod
    def is_hit(self, item: int) -> bool:
        """Return True if item is in cache (during delivery)."""
        pass

    @abstractmethod
    def clear(self):
        """Clear cache contents."""
        pass

