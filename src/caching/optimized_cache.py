# src/caching/optimized_cache.py
import numpy as np
from scipy.optimize import minimize
from .cache_base import CacheBase
from typing import Dict, List, Tuple

class JointOptimizationCache(CacheBase):
    """
    Cache that jointly optimizes content placement and NOMA transmission
    to minimize overall system outage probability.
    """
    
    def __init__(self, capacity: int, num_files: int, channel_gains: np.ndarray, cfg):
        super().__init__(capacity)
        self.num_files = num_files
        self.channel_gains = channel_gains
        self.cfg = cfg
        self.contents = set()
        self.popularity_weights = np.ones(num_files) / num_files
        
    def update_popularity(self, request_history: List[int], window_size: int = 1000):
        """Update popularity estimates from recent requests."""
        recent = request_history[-window_size:] if len(request_history) > window_size else request_history
        counts = np.bincount(recent, minlength=self.num_files)
        self.popularity_weights = counts / len(recent) if len(recent) > 0 else self.popularity_weights
    
    def expected_outage_cost(self, cache_config: np.ndarray) -> float:
        """
        Calculate expected outage probability for a given cache configuration.
        
        Args:
            cache_config: Binary array indicating which files to cache
        """
        total_cost = 0.0
        
        for file_idx in range(self.num_files):
            request_prob = self.popularity_weights[file_idx]
            
            if cache_config[file_idx] == 1:
                # File is cached - no transmission needed
                cost = 0.0
            else:
                # File not cached - need NOMA transmission
                # Estimate outage probability for this file request
                cost = self._estimate_noma_outage()
                
            total_cost += request_prob * cost
            
        return total_cost
    
    def _estimate_noma_outage(self) -> float:
        """Estimate average NOMA outage probability."""
        # Simplified model - in practice, this would be more sophisticated
        num_users = len(self.channel_gains)
        weak_gains = self.channel_gains[self.channel_gains < np.median(self.channel_gains)]
        strong_gains = self.channel_gains[self.channel_gains >= np.median(self.channel_gains)]
        
        # Estimate pairing success rate
        if len(weak_gains) == 0 or len(strong_gains) == 0:
            return 1.0  # High outage if can't pair
            
        avg_weak = np.mean(weak_gains)
        avg_strong = np.mean(strong_gains)
        
        # Simplified SINR calculation for average case
        sinr_th = 2 ** self.cfg.TARGET_RATE_BPS - 1
        p_w = self.cfg.POWER_COEFF_WEAK
        
        sinr_weak = (self.cfg.TX_POWER * p_w * avg_weak) / \
                   (self.cfg.TX_POWER * (1-p_w) * avg_weak + self.cfg.NOISE_POWER)
        
        outage_prob = 1.0 if sinr_weak < sinr_th else 0.1  # Simplified
        return outage_prob
    
    def optimize_cache_contents(self) -> np.ndarray:
        """
        Solve the cache optimization problem:
        min_x  sum_i p_i * (1-x_i) * outage_i
        s.t.   sum_i x_i <= capacity
               x_i ∈ {0,1}
        """
        
        # For large problems, use greedy approximation
        if self.num_files > 1000:
            return self._greedy_optimize()
        
        # For smaller problems, try integer programming approach
        return self._ip_optimize()
    
    def _greedy_optimize(self) -> np.ndarray:
        """Greedy algorithm: select files with highest benefit-to-cost ratio."""
        
        # Calculate benefit of caching each file
        benefits = []
        outage_cost = self._estimate_noma_outage()
        
        for i in range(self.num_files):
            # Benefit = probability of request * saved outage cost
            benefit = self.popularity_weights[i] * outage_cost
            benefits.append((benefit, i))
        
        # Sort by benefit (descending)
        benefits.sort(reverse=True)
        
        # Select top-k files
        cache_config = np.zeros(self.num_files)
        for j in range(min(self.capacity, len(benefits))):
            _, file_idx = benefits[j]
            cache_config[file_idx] = 1
            
        return cache_config
    
    def _ip_optimize(self) -> np.ndarray:
        """Integer programming approach for exact optimization."""
        from scipy.optimize import linprog
        
        # Relaxed LP version (then round)
        c = -self.popularity_weights * self._estimate_noma_outage()  # Minimize negative benefit
        
        # Constraint: sum(x_i) <= capacity
        A_ub = np.ones((1, self.num_files))
        b_ub = np.array([self.capacity])
        
        # Bounds: 0 <= x_i <= 1
        bounds = [(0, 1) for _ in range(self.num_files)]
        
        try:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if result.success:
                # Round to integer solution
                solution = result.x
                # Select top files based on LP solution
                indices = np.argsort(-solution)[:self.capacity]
                cache_config = np.zeros(self.num_files)
                cache_config[indices] = 1
                return cache_config
        except:
            pass
            
        # Fallback to greedy
        return self._greedy_optimize()
    
    def populate(self, items=None):
        """Populate cache using optimization."""
        cache_config = self.optimize_cache_contents()
        self.contents = set(i for i in range(self.num_files) if cache_config[i] == 1)
    
    def is_hit(self, item: int) -> bool:
        return int(item) in self.contents
    
    def clear(self):
        self.contents.clear()


class ReinforcementLearningCache(CacheBase):
    """
    Q-Learning based cache that learns optimal caching policy.
    """
    
    def __init__(self, capacity: int, num_files: int, learning_rate: float = 0.1):
        super().__init__(capacity)
        self.num_files = num_files
        self.lr = learning_rate
        self.epsilon = 0.1  # exploration rate
        self.contents = set()
        
        # Q-table: Q(state, action)
        # State: current cache contents (simplified)
        # Action: which file to cache/evict
        self.q_table = {}
        
    def get_state(self) -> str:
        """Get current state representation."""
        return str(sorted(list(self.contents)))
    
    def choose_action(self, available_actions: List[int]) -> int:
        """Choose action using epsilon-greedy policy."""
        state = self.get_state()
        
        if np.random.random() < self.epsilon or state not in self.q_table:
            # Exploration: random action
            return np.random.choice(available_actions)
        
        # Exploitation: best action
        q_values = [self.q_table[state].get(action, 0.0) for action in available_actions]
        best_action_idx = np.argmax(q_values)
        return available_actions[best_action_idx]
    
    def update_q_value(self, state: str, action: int, reward: float, next_state: str):
        """Update Q-table using Q-learning update rule."""
        if state not in self.q_table:
            self.q_table[state] = {}
        if next_state not in self.q_table:
            self.q_table[next_state] = {}
            
        old_q = self.q_table[state].get(action, 0.0)
        next_max_q = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0.0
        
        # Q-learning update: Q(s,a) = Q(s,a) + α[r + γ*max_Q(s',a') - Q(s,a)]
        new_q = old_q + self.lr * (reward + 0.9 * next_max_q - old_q)
        self.q_table[state][action] = new_q
    
    def request_with_learning(self, file_id: int, hit: bool, outage_occurred: bool):
        """Process request and learn from outcome."""
        state = self.get_state()
        
        if not hit:
            # Need to make caching decision
            if len(self.contents) < self.capacity:
                # Add to cache
                action = file_id
                self.contents.add(file_id)
            else:
                # Need to evict something
                evict_candidates = list(self.contents)
                action = self.choose_action(evict_candidates)
                self.contents.remove(action)
                self.contents.add(file_id)
        
            # Reward based on outcome
            reward = 1.0 if not outage_occurred else -1.0
            next_state = self.get_state()
            
            self.update_q_value(state, action, reward, next_state)
    
    def populate(self, items=None):
        """For RL cache, initial population is random or empty."""
        pass
    
    def is_hit(self, item: int) -> bool:
        return int(item) in self.contents
    
    def clear(self):
        self.contents.clear()
        # Don't clear Q-table - keep learned knowledge