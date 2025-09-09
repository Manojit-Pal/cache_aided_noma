# src/caching/predictive_cache.py
from .cache_base import CacheBase
import numpy as np

class EMAPredictiveCache(CacheBase):
    """
    Exponential Moving Average based predictor for file popularity.
    Maintain EMA of request counts and populate top-k accordingly.
    """
    def __init__(self, capacity: int, num_files: int, alpha: float = 0.3):
        super().__init__(capacity)
        self.capacity = capacity
        self.num_files = num_files
        self.alpha = alpha
        self.ema = np.zeros(num_files)  # EMA score per file
        self.contents = set()

    def observe_requests(self, requests):
        """
        requests: iterable of file indices observed in latest time window
        Update EMA counts using normalized frequency in this window.
        """
        counts = np.bincount(requests, minlength=self.num_files)
        # normalize to probability
        total = counts.sum()
        if total > 0:
            freq = counts / total
        else:
            freq = counts
        # EMA update: ema = alpha * freq + (1-alpha) * ema
        self.ema = self.alpha * freq + (1.0 - self.alpha) * self.ema

    def populate(self, items=None):
        """
        Populate cache according to current EMA ranking. 'items' ignored.
        """
        ranking = np.argsort(-self.ema)  # descending
        topk = ranking[: self.capacity]
        self.contents = set(int(x) for x in topk)

    def is_hit(self, item: int) -> bool:
        return int(item) in self.contents

    def clear(self):
        self.contents.clear()
        self.ema[:] = 0.0
