# src/caching/rl_noma_cache.py
"""
Deep Q-Learning based cache for NOMA networks.
State space includes: cache contents, channel conditions, popularity trends, NOMA outcomes.
Action space: discrete actions for cache replacement decisions.
"""

import numpy as np
from collections import deque, defaultdict
import pickle
from typing import Dict, List, Tuple, Optional
from .cache_base import CacheBase


class DQNNomaCache(CacheBase):
    """
    Deep Q-Network based caching policy optimized for NOMA transmission.
    
    State Representation (Feature Vector):
    - Content popularity (EMA of request frequency)
    - Channel quality distribution (avg, std of user channel gains)
    - Cache status (which files are cached)
    - Recent NOMA performance (success rate, outage rate)
    - Request pattern features (temporal, spatial)
    
    Action Space:
    - For each request: cache the file or not (if cache full, which file to evict)
    
    Reward:
    - +10: Cache hit
    - -1: Cache miss with successful NOMA transmission
    - -10: Cache miss with NOMA failure (outage)
    - -5: Cache miss with high BER
    """
    
    def __init__(self, capacity: int, num_files: int, num_users: int,
                 learning_rate: float = 0.001, 
                 gamma: float = 0.95,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995):
        super().__init__(capacity)
        
        self.num_files = num_files
        self.num_users = num_users
        self.lr = learning_rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Cache contents
        self.contents = set()
        
        # State tracking
        self.popularity_ema = np.ones(num_files) / num_files
        self.alpha_pop = 0.1  # EMA weight for popularity
        
        self.channel_quality_history = deque(maxlen=100)
        self.noma_outcome_history = deque(maxlen=200)
        self.request_history = deque(maxlen=500)
        
        # Q-table with state hashing (for discrete states)
        # For continuous states, we'd use a neural network
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # Experience replay buffer
        self.replay_buffer = deque(maxlen=2000)
        self.batch_size = 32
        
        # File-specific metrics
        self.file_request_count = np.zeros(num_files)
        self.file_noma_success = defaultdict(list)  # Track NOMA success per file
        self.file_user_affinity = defaultdict(list)  # Which users request which files
        
        # Performance tracking
        self.episode_rewards = []
        self.cumulative_reward = 0
        
    def get_state_features(self) -> np.ndarray:
        """
        Extract state features for RL decision making.
        Returns a feature vector representing current state.
        """
        features = []
        
        # 1. Popularity features (top-k most popular files)
        top_k = 10
        popularity_sorted_idx = np.argsort(-self.popularity_ema)[:top_k]
        popularity_features = self.popularity_ema[popularity_sorted_idx]
        features.extend(popularity_features)
        
        # 2. Cache occupancy features
        cache_occupancy = len(self.contents) / self.capacity
        features.append(cache_occupancy)
        
        # 3. Channel quality features
        if len(self.channel_quality_history) > 0:
            recent_qualities = list(self.channel_quality_history)[-20:]
            avg_quality = np.mean(recent_qualities)
            std_quality = np.std(recent_qualities)
            features.extend([avg_quality, std_quality])
        else:
            features.extend([0.5, 0.1])  # Default values
        
        # 4. NOMA performance features
        if len(self.noma_outcome_history) > 0:
            recent_noma = list(self.noma_outcome_history)[-50:]
            success_rate = np.mean([x['success'] for x in recent_noma])
            avg_sinr_weak = np.mean([x['sinr_weak'] for x in recent_noma])
            avg_sinr_strong = np.mean([x['sinr_strong'] for x in recent_noma])
            features.extend([success_rate, avg_sinr_weak, avg_sinr_strong])
        else:
            features.extend([0.5, 0.0, 0.0])  # Default values
        
        # 5. Temporal features (request rate trends)
        if len(self.request_history) > 10:
            recent_requests = list(self.request_history)[-50:]
            request_rate = len(recent_requests) / 50.0
            features.append(request_rate)
        else:
            features.append(0.0)
        
        return np.array(features)
    
    def discretize_state(self, features: np.ndarray) -> str:
        """
        Convert continuous features to discrete state representation.
        Used for Q-table indexing.
        """
        # Discretize each feature into bins
        discretized = []
        
        # Popularity (10 features): 3 bins each
        for i in range(10):
            if i < len(features):
                val = features[i]
                if val < 0.05:
                    discretized.append('L')
                elif val < 0.15:
                    discretized.append('M')
                else:
                    discretized.append('H')
            else:
                discretized.append('L')
        
        # Cache occupancy: 3 bins
        if features[10] < 0.33:
            discretized.append('low')
        elif features[10] < 0.67:
            discretized.append('med')
        else:
            discretized.append('high')
        
        # Channel quality avg: 3 bins
        if features[11] < 0.3:
            discretized.append('poor')
        elif features[11] < 0.7:
            discretized.append('fair')
        else:
            discretized.append('good')
        
        # NOMA success rate: 3 bins
        if features[13] < 0.4:
            discretized.append('low_noma')
        elif features[13] < 0.7:
            discretized.append('med_noma')
        else:
            discretized.append('high_noma')
        
        return '|'.join(discretized)
    
    def get_action_space(self, file_id: int) -> List[str]:
        """
        Define possible actions for a given file request.
        Actions: 'cache', 'no_cache', or 'evict_X' (where X is a file in cache)
        """
        actions = []
        
        if file_id in self.contents:
            # File already cached - no action needed
            return ['keep']
        
        if len(self.contents) < self.capacity:
            # Cache has space - can directly cache
            actions.append('cache')
        else:
            # Cache is full - need to evict
            actions.append('no_cache')
            for cached_file in self.contents:
                actions.append(f'evict_{cached_file}')
        
        return actions
    
    def select_action(self, state_str: str, actions: List[str]) -> str:
        """
        Epsilon-greedy action selection.
        """
        if np.random.random() < self.epsilon:
            # Exploration: random action
            return np.random.choice(actions)
        else:
            # Exploitation: best action based on Q-values
            q_values = [self.q_table[state_str][action] for action in actions]
            max_q = max(q_values)
            best_actions = [actions[i] for i in range(len(actions)) if q_values[i] == max_q]
            return np.random.choice(best_actions)
    
    def update_q_value(self, state: str, action: str, reward: float, next_state: str, next_actions: List[str]):
        """
        Q-learning update: Q(s,a) = Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
        """
        current_q = self.q_table[state][action]
        
        # Find max Q-value for next state
        if next_actions:
            next_q_values = [self.q_table[next_state][a] for a in next_actions]
            max_next_q = max(next_q_values) if next_q_values else 0.0
        else:
            max_next_q = 0.0
        
        # Q-learning update
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q
    
    def compute_reward(self, cache_hit: bool, noma_success: Optional[bool] = None, 
                      ber: Optional[float] = None, outage: bool = False) -> float:
        """
        Reward function tailored for NOMA caching.
        """
        if cache_hit:
            # Cache hit is best outcome
            return 10.0
        
        # Cache miss - reward depends on NOMA performance
        if noma_success is None:
            return -1.0  # Unknown outcome
        
        if outage:
            # NOMA failed (outage) - worst case
            return -10.0
        
        if not noma_success:
            # NOMA transmission but user not satisfied
            return -5.0
        
        # NOMA succeeded but had to transmit
        reward = -1.0
        
        # Bonus for good BER
        if ber is not None and ber < 1e-4:
            reward += 2.0
        elif ber is not None and ber > 1e-2:
            reward -= 2.0
        
        return reward
    
    def observe_request(self, user_id: int, file_id: int, 
                       cache_hit: bool, 
                       noma_success: Optional[bool] = None,
                       channel_gain: Optional[float] = None,
                       sinr_weak: Optional[float] = None,
                       sinr_strong: Optional[float] = None,
                       ber: Optional[float] = None,
                       outage: bool = False):
        """
        Main learning function: observe request outcome and update Q-values.
        """
        # Update popularity
        self.file_request_count[file_id] += 1
        freq = np.zeros(self.num_files)
        freq[file_id] = 1.0
        self.popularity_ema = (self.alpha_pop * freq + 
                              (1 - self.alpha_pop) * self.popularity_ema)
        
        # Track channel quality
        if channel_gain is not None:
            self.channel_quality_history.append(channel_gain)
        
        # Track NOMA outcome
        if noma_success is not None:
            self.noma_outcome_history.append({
                'file_id': file_id,
                'user_id': user_id,
                'success': noma_success,
                'sinr_weak': sinr_weak or 0.0,
                'sinr_strong': sinr_strong or 0.0,
                'outage': outage
            })
            self.file_noma_success[file_id].append(noma_success)
        
        # Track user-file affinity
        self.file_user_affinity[file_id].append(user_id)
        
        # Record request
        self.request_history.append({
            'file_id': file_id,
            'user_id': user_id,
            'cache_hit': cache_hit,
            'timestamp': len(self.request_history)
        })
        
        # Get current state
        state_features = self.get_state_features()
        state_str = self.discretize_state(state_features)
        
        # Get possible actions
        actions = self.get_action_space(file_id)
        
        if len(actions) == 1 and actions[0] == 'keep':
            # File already cached, no learning needed
            return
        
        # Select action
        action = self.select_action(state_str, actions)
        
        # Execute action
        if action == 'cache':
            self.contents.add(file_id)
        elif action.startswith('evict_'):
            # Extract file to evict
            evict_file = int(action.split('_')[1])
            if evict_file in self.contents:
                self.contents.remove(evict_file)
                self.contents.add(file_id)
        # else: 'no_cache' - don't cache the file
        
        # Compute reward
        reward = self.compute_reward(cache_hit, noma_success, ber, outage)
        self.cumulative_reward += reward
        
        # Get next state
        next_state_features = self.get_state_features()
        next_state_str = self.discretize_state(next_state_features)
        
        # Next possible actions (for future request)
        # Simplified: assume same file might be requested
        next_actions = self.get_action_space(file_id)
        
        # Update Q-value
        self.update_q_value(state_str, action, reward, next_state_str, next_actions)
        
        # Store experience for potential replay
        self.replay_buffer.append({
            'state': state_str,
            'action': action,
            'reward': reward,
            'next_state': next_state_str,
            'next_actions': next_actions
        })
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def experience_replay(self, batch_size: Optional[int] = None):
        """
        Train on random batch from experience replay buffer.
        """
        if len(self.replay_buffer) < (batch_size or self.batch_size):
            return
        
        batch = np.random.choice(len(self.replay_buffer), 
                                size=batch_size or self.batch_size, 
                                replace=False)
        
        for idx in batch:
            exp = self.replay_buffer[idx]
            self.update_q_value(
                exp['state'], 
                exp['action'], 
                exp['reward'],
                exp['next_state'],
                exp['next_actions']
            )
    
    def populate(self, items=None):
        """
        Initial cache population based on learned policy.
        Use top files by predicted Q-value.
        """
        if len(self.q_table) == 0:
            # No learning yet - use popularity
            top_indices = np.argsort(-self.popularity_ema)[:self.capacity]
            self.contents = set(int(x) for x in top_indices)
        else:
            # Use learned policy
            state_features = self.get_state_features()
            state_str = self.discretize_state(state_features)
            
            # Evaluate each file's value
            file_values = []
            for file_id in range(self.num_files):
                action = 'cache'
                q_value = self.q_table[state_str].get(action, 0.0)
                popularity = self.popularity_ema[file_id]
                noma_success = (np.mean(self.file_noma_success[file_id]) 
                               if self.file_noma_success[file_id] else 0.5)
                
                # Combined score
                score = q_value + popularity * 10 + noma_success * 5
                file_values.append((score, file_id))
            
            # Select top files
            file_values.sort(reverse=True)
            self.contents = set(file_id for _, file_id in file_values[:self.capacity])
    
    def is_hit(self, item: int) -> bool:
        return int(item) in self.contents
    
    def clear(self):
        self.contents.clear()
    
    def save_model(self, filepath: str):
        """Save learned Q-table to disk."""
        model = {
            'q_table': dict(self.q_table),
            'popularity_ema': self.popularity_ema,
            'epsilon': self.epsilon,
            'cumulative_reward': self.cumulative_reward
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
    
    def load_model(self, filepath: str):
        """Load learned Q-table from disk."""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        self.q_table = defaultdict(lambda: defaultdict(float), model['q_table'])
        self.popularity_ema = model['popularity_ema']
        self.epsilon = model['epsilon']
        self.cumulative_reward = model['cumulative_reward']
    
    def get_stats(self) -> Dict:
        """Return learning statistics."""
        return {
            'q_table_size': len(self.q_table),
            'epsilon': self.epsilon,
            'cumulative_reward': self.cumulative_reward,
            'replay_buffer_size': len(self.replay_buffer),
            'cache_size': len(self.contents),
            'total_requests_observed': len(self.request_history),
            'noma_outcomes_observed': len(self.noma_outcome_history)
        }