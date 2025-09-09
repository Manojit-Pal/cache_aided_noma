# src/caching/noma_aware_cache.py
"""
Novel Contribution: NOMA-Aware Predictive Caching (NAPC)

Key Innovation: Cache decisions consider both content popularity AND 
the expected NOMA transmission success probability for different user pairs.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque
from .cache_base import CacheBase

class NomaAwarePredictiveCache(CacheBase):
    """
    NOMA-Aware Predictive Cache (NAPC) - Novel Algorithm
    
    Key Ideas:
    1. Predict both content popularity AND user mobility/channel conditions
    2. Cache files that would be hardest to deliver via NOMA (poor channel users)  
    3. Use multi-objective optimization balancing hit rate vs outage reduction
    4. Adaptive learning from both cache hits/misses AND NOMA success/failure
    """
    
    def __init__(self, capacity: int, num_files: int, num_users: int, 
                 alpha_popularity: float = 0.3, alpha_channel: float = 0.2):
        super().__init__(capacity)
        self.num_files = num_files
        self.num_users = num_users
        self.alpha_pop = alpha_popularity    # EMA weight for popularity
        self.alpha_ch = alpha_channel        # EMA weight for channel conditions
        
        # State tracking
        self.contents = set()
        self.popularity_ema = np.ones(num_files) / num_files
        self.channel_quality_ema = np.ones(num_users) * 0.5  # 0=bad, 1=good
        self.user_file_affinity = defaultdict(lambda: defaultdict(float))
        
        # Learning history
        self.request_history = deque(maxlen=1000)
        self.noma_outcome_history = deque(maxlen=1000)
        self.file_user_requests = defaultdict(list)
        
        # Cache metrics for adaptive weighting
        self.cache_hit_value = 0.0  # Expected benefit of cache hit
        self.noma_success_rate = 0.5  # Global NOMA success rate
        
    def observe_request(self, user_id: int, file_id: int, cache_hit: bool, 
                       noma_success: Optional[bool] = None):
        """
        Learn from request outcome - this is the key innovation.
        We learn from BOTH cache performance AND NOMA performance.
        """
        # Update popularity
        freq = np.zeros(self.num_files)
        freq[file_id] = 1.0
        self.popularity_ema = self.alpha_pop * freq + (1 - self.alpha_pop) * self.popularity_ema
        
        # Track user-file patterns  
        self.user_file_affinity[user_id][file_id] = \
            0.1 * 1.0 + 0.9 * self.user_file_affinity[user_id][file_id]
            
        # Record request
        self.request_history.append((user_id, file_id, cache_hit))
        
        # Learn from NOMA outcome (if cache miss occurred)
        if not cache_hit and noma_success is not None:
            self.noma_outcome_history.append((user_id, file_id, noma_success))
            
            # Update user's channel quality estimate based on NOMA success
            current_quality = self.channel_quality_ema[user_id]
            success_indicator = 1.0 if noma_success else 0.0
            self.channel_quality_ema[user_id] = \
                self.alpha_ch * success_indicator + (1 - self.alpha_ch) * current_quality
                
            # Update global NOMA success rate
            recent_successes = [x[2] for x in list(self.noma_outcome_history)[-100:]]
            self.noma_success_rate = np.mean(recent_successes) if recent_successes else 0.5
            
        # Track file requests by user
        self.file_user_requests[file_id].append(user_id)
        if len(self.file_user_requests[file_id]) > 100:
            self.file_user_requests[file_id].pop(0)
    
    def compute_caching_priority(self, file_id: int) -> float:
        """
        Novel Priority Function: Combines popularity with NOMA difficulty.
        
        Intuition: Cache files that are:
        1. Popular (high request probability)  
        2. Requested by users with poor channels (high NOMA failure risk)
        3. Hard to deliver via NOMA (considering user pairing challenges)
        """
        
        # Component 1: Content popularity
        popularity_score = self.popularity_ema[file_id]
        
        # Component 2: Channel-aware scoring
        requesting_users = self.file_user_requests.get(file_id, [])
        if not requesting_users:
            channel_difficulty = 0.5  # Unknown
        else:
            # Average channel quality of users requesting this file
            user_qualities = [self.channel_quality_ema[u] for u in requesting_users[-20:]]
            avg_quality = np.mean(user_qualities)
            # Convert to difficulty: poor channel users = high caching priority
            channel_difficulty = 1.0 - avg_quality
        
        # Component 3: NOMA pairing difficulty
        pairing_difficulty = self._estimate_pairing_difficulty(file_id)
        
        # Component 4: Historical NOMA failure rate for this file
        file_failures = [(u, f, s) for u, f, s in self.noma_outcome_history if f == file_id]
        if file_failures:
            failure_rate = 1.0 - np.mean([s for _, _, s in file_failures])
        else:
            failure_rate = 1.0 - self.noma_success_rate  # Global average
        
        # Weighted combination - this is tunable based on system priorities
        w1, w2, w3, w4 = 0.4, 0.3, 0.2, 0.1  # Weights sum to 1.0
        
        priority = (w1 * popularity_score + 
                   w2 * channel_difficulty + 
                   w3 * pairing_difficulty + 
                   w4 * failure_rate)
        
        return priority
    
    def _estimate_pairing_difficulty(self, file_id: int) -> float:
        """
        Estimate how hard it would be to deliver this file via NOMA pairing.
        
        Considers: Are users requesting this file suitable for pairing?
        """
        requesting_users = self.file_user_requests.get(file_id, [])
        if len(requesting_users) < 2:
            return 0.8  # Hard to pair if only one type of user requests it
        
        # Check diversity of channel qualities among requesting users
        user_qualities = [self.channel_quality_ema[u] for u in requesting_users[-20:]]
        quality_std = np.std(user_qualities)
        
        # High diversity = easier pairing, low diversity = harder pairing
        difficulty = 1.0 / (1.0 + quality_std)  # Sigmoid-like mapping
        return difficulty
    
    def dynamic_capacity_adjustment(self) -> int:
        """
        Novel Feature: Dynamically adjust cache capacity based on NOMA performance.
        
        If NOMA is working well, we can cache less (save storage).
        If NOMA is failing often, cache more aggressively.
        """
        base_capacity = self.capacity
        
        # If NOMA success rate is high, we can reduce caching
        if self.noma_success_rate > 0.8:
            adjusted = int(base_capacity * 0.8)
        elif self.noma_success_rate < 0.3:  
            adjusted = int(base_capacity * 1.2)  # Cache more aggressively
        else:
            adjusted = base_capacity
            
        return min(adjusted, self.num_files)  # Don't exceed total files
    
    def populate(self, items=None):
        """
        Populate cache using NOMA-aware priorities.
        """
        # Compute priority for each file
        priorities = [(self.compute_caching_priority(f), f) for f in range(self.num_files)]
        
        # Sort by priority (descending)
        priorities.sort(reverse=True)
        
        # Use dynamic capacity
        effective_capacity = self.dynamic_capacity_adjustment()
        
        # Select top files
        self.contents = set(file_id for _, file_id in priorities[:effective_capacity])
    
    def is_hit(self, item: int) -> bool:
        return int(item) in self.contents
    
    def clear(self):
        self.contents.clear()
    
    def get_cache_stats(self) -> Dict:
        """Return detailed statistics for analysis."""
        return {
            "contents": list(self.contents),
            "popularity_top10": np.argsort(-self.popularity_ema)[:10].tolist(),
            "avg_channel_quality": np.mean(self.channel_quality_ema),
            "noma_success_rate": self.noma_success_rate,
            "request_history_len": len(self.request_history),
            "noma_history_len": len(self.noma_outcome_history)
        }


class MultiObjectiveCache(CacheBase):
    """
    Multi-Objective Optimization Cache using Pareto efficiency.
    
    Simultaneously optimizes:
    1. Cache hit rate (maximize)
    2. NOMA outage probability (minimize) 
    3. Energy consumption (minimize)
    4. Load balancing (fairness)
    """
    
    def __init__(self, capacity: int, num_files: int, objectives: List[str]):
        super().__init__(capacity)
        self.num_files = num_files
        self.objectives = objectives  # e.g., ['hit_rate', 'outage', 'energy']
        self.contents = set()
        self.pareto_solutions = []
        
    def evaluate_solution(self, cache_config: np.ndarray) -> Dict[str, float]:
        """Evaluate a cache configuration on all objectives."""
        metrics = {}
        
        if 'hit_rate' in self.objectives:
            metrics['hit_rate'] = self._estimate_hit_rate(cache_config)
        if 'outage' in self.objectives:
            metrics['outage'] = self._estimate_outage_rate(cache_config)
        if 'energy' in self.objectives:
            metrics['energy'] = self._estimate_energy_consumption(cache_config)
        if 'fairness' in self.objectives:
            metrics['fairness'] = self._estimate_fairness(cache_config)
            
        return metrics
    
    def find_pareto_optimal_caching(self) -> List[np.ndarray]:
        """Find Pareto-optimal cache configurations."""
        # For demonstration - in practice, use NSGA-II or similar
        solutions = []
        
        # Generate candidate solutions (simplified)
        for _ in range(100):
            config = np.zeros(self.num_files)
            selected = np.random.choice(self.num_files, self.capacity, replace=False)
            config[selected] = 1
            solutions.append(config)
        
        # Evaluate all solutions
        evaluated = [(config, self.evaluate_solution(config)) for config in solutions]
        
        # Find Pareto frontier (simplified)
        pareto_set = []
        for i, (config_i, metrics_i) in enumerate(evaluated):
            dominated = False
            for j, (config_j, metrics_j) in enumerate(evaluated):
                if i != j and self._dominates(metrics_j, metrics_i):
                    dominated = True
                    break
            if not dominated:
                pareto_set.append((config_i, metrics_i))
        
        return [config for config, _ in pareto_set]
    
    def _dominates(self, metrics_a: Dict, metrics_b: Dict) -> bool:
        """Check if solution A dominates solution B."""
        # A dominates B if A is at least as good in all objectives 
        # and strictly better in at least one
        better_in_at_least_one = False
        
        for obj in self.objectives:
            if obj in ['hit_rate', 'fairness']:  # Higher is better
                if metrics_a[obj] < metrics_b[obj]:
                    return False
                if metrics_a[obj] > metrics_b[obj]:
                    better_in_at_least_one = True
            else:  # Lower is better (outage, energy)
                if metrics_a[obj] > metrics_b[obj]:
                    return False  
                if metrics_a[obj] < metrics_b[obj]:
                    better_in_at_least_one = True
                    
        return better_in_at_least_one
    
    def _estimate_hit_rate(self, cache_config: np.ndarray) -> float:
        # Simplified estimation
        return np.sum(cache_config) / self.num_files
    
    def _estimate_outage_rate(self, cache_config: np.ndarray) -> float:
        # Would need channel information - simplified
        return 1.0 - np.sum(cache_config) / self.num_files
    
    def _estimate_energy_consumption(self, cache_config: np.ndarray) -> float:
        # Cache storage energy + transmission energy for misses
        storage_energy = np.sum(cache_config) * 0.1
        transmission_energy = (self.num_files - np.sum(cache_config)) * 1.0
        return storage_energy + transmission_energy
    
    def _estimate_fairness(self, cache_config: np.ndarray) -> float:
        # Simplified fairness metric
        return 1.0 - np.var(cache_config)  # Prefer balanced caching
    
    def populate(self, items=None):
        pareto_solutions = self.find_pareto_optimal_caching()
        if pareto_solutions:
            # Choose one solution (e.g., based on user preference)
            chosen_config = pareto_solutions[0]  # Simplified selection
            self.contents = set(i for i in range(self.num_files) if chosen_config[i] == 1)
    
    def is_hit(self, item: int) -> bool:
        return int(item) in self.contents
    
    def clear(self):
        self.contents.clear()