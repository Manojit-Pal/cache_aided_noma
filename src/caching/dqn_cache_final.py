"""
src/caching/dqn_cache_final.py

NOMA-AWARE DEEP Q-NETWORK CACHE
===============================

Implementation based on research papers:
- IEEE DeepChunk (2019): Deep Q-Learning for Chunk-based Caching
- RLCaR: Reinforcement Learning Cache Replacement
- peihaowang/DRLCache: Deep RL-based Cache Replacement

Key NOMA Features:
- CIC (Cache-aided Interference Cancellation) tracking
- SIC (Successive Interference Cancellation) detection  
- Channel-aware state representation
- NOMA performance-based rewards
- User pairing integration

Author: Cache-Aided NOMA Team
Date: December 2025
"""

import numpy as np
import random
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional, Set, Iterable
import pickle

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not available - DQN will use Q-table fallback")

from .cache_base import CacheBase


# ====================================================================================
# DUELING DQN NETWORK (Following DeepMind Architecture)
# ============================================================================

class DuelingDQN(nn.Module):
    """
    Dueling DQN Architecture (Wang et al., ICML 2016).
    
    Separates Q-function into:
    - Value function V(s): How good is this state?
    - Advantage function A(s,a): How much better is action a?
    
    Q(s,a) = V(s) + [A(s,a) - mean(A(s,a))]
    
    This improves learning by allowing the network to learn which
    states are valuable independent of specific actions.
    """
    
    def __init__(self, state_dim: int, action_dim: int, 
                 hidden_dims: List[int] = [128, 64]):
        super(DuelingDQN, self).__init__()
        
        # Shared feature layers
        self.feature_layers = nn.ModuleList()
        prev_dim = state_dim
        
        for hdim in hidden_dims:
            self.feature_layers.append(nn.Linear(prev_dim, hdim))
            prev_dim = hdim
        
        # Value stream V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1] // 2, 1)
        )
        
        # Advantage stream A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1] // 2, action_dim)
        )
        
        # He initialization for better gradient flow
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with He/Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: state -> Q-values for all actions.
        
        Q(s,a) = V(s) + [A(s,a) - mean_a(A(s,a))]
        """
        # Shared features
        x = state
        for layer in self.feature_layers:
            x = F.relu(layer(x))
        
        # Separate value and advantage
        value = self.value_stream(x)  # (batch, 1)
        advantage = self.advantage_stream(x)  # (batch, action_dim)
        
        # Combine: subtract mean advantage for stability
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values


# ====================================================================================
# PRIORITIZED EXPERIENCE REPLAY (Schaul et al., ICLR 2016)
# ============================================================================

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer.
    
    Samples transitions with probability proportional to TD-error,
    allowing agent to learn more from surprising experiences.
    
    Based on: Schaul et al., "Prioritized Experience Replay", ICLR 2016
    """
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        """
        Args:
            capacity: Maximum buffer size
            alpha: Priority exponent (0=uniform, 1=full prioritization)
            beta: Importance sampling exponent (compensates for bias)
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0
    
    def add(self, experience: Dict):
        """Add experience with max priority (ensures at least one sample)."""
        self.buffer.append(experience)
        self.priorities.append(self.max_priority)
    
    def sample(self, batch_size: int) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
        """
        Sample batch with prioritized sampling.
        
        Returns:
            experiences: List of experience dicts
            weights: Importance sampling weights
            indices: Sampled indices (for priority updates)
        """
        if len(self.buffer) < batch_size:
            return None, None, None
        
        # Compute sampling probabilities
        priorities = np.array(self.priorities, dtype=np.float64)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, 
                                  p=probs, replace=False)
        
        # Compute importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize for stability
        
        experiences = [self.buffer[idx] for idx in indices]
        
        return experiences, weights, indices
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on absolute TD-errors."""
        for idx, error in zip(indices, td_errors):
            priority = float(abs(error) + 1e-6)
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = priority
                self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)


# ====================================================================================
# NOMA-AWARE DQN CACHE
# ============================================================================

class DQNCache(CacheBase):
    """
    Deep Q-Network Cache for NOMA Systems.
    
    **MDP Formulation:**
    
    State s_t:
        - LRU counters: timesteps since last access for each cached file
        - LFU counters: access frequency for each cached file
        - Requested file popularity
        - Channel quality metrics (mean, std)
        - NOMA performance (CIC rate, success rate)
        - Cache occupancy
    
    Action a_t:
        - Which cache slot to evict (0 to capacity-1)
        - Based on research: slot-based actions simplify learning
    
    Reward r_t:
        - +10: Cache hit (best outcome)
        - +2: Cache miss + CIC enabled (good outcome)
        - -1: Cache miss + NOMA success (acceptable)
        - -5: Cache miss + NOMA failure (bad outcome)
        - -10: Outage (worst outcome)
    
    **Research-Based Design Choices:**
    
    1. State = LRU + LFU heuristics (RLCaR paper)
    2. Action = Slot eviction (simplifies action space)
    3. Reward = Cache hit ratio optimization (standard)
    4. Network = Dueling DQN (better value estimation)
    5. Replay = Prioritized (learn from important transitions)
    
    **NOMA Integration:**
    
    - CIC tracking: Bonus reward when cache enables interference cancellation
    - SIC detection: Track when strong user gets perfect SIC
    - Channel-aware: State includes channel quality
    - Pairing-aware: Considers NOMA user pairs
    """
    
    def __init__(
        self,
        capacity: int,
        num_files: int,
        num_users: int,
        
        # DQN hyperparameters
        learning_rate: float = 0.0001,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay_steps: int = 25000,
        
        # Network architecture
        use_neural_network: bool = True,
        hidden_dims: List[int] = [128, 64],
        
        # Training parameters
        batch_size: int = 64,
        replay_buffer_size: int = 50000,
        target_update_freq: int = 1000,
        train_freq: int = 4,
        
        # Prioritized replay
        use_prioritized_replay: bool = True,
        priority_alpha: float = 0.6,
        priority_beta: float = 0.4,
        
        # Stability
        gradient_clip: float = 10.0,
        tau: float = 0.005,
        
        # NOMA awareness
        enable_noma_awareness: bool = True,
        
        seed: int = 2025
    ):
        super().__init__(capacity, enable_noma_awareness)
        
        # Environment parameters
        self.num_files = num_files
        self.num_users = num_users
        
        # Hyperparameters
        self.lr = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.train_freq = train_freq
        self.gradient_clip = gradient_clip
        self.tau = tau
        
        # Set seeds for reproducibility
        self._set_seeds(seed)
        
        # Epsilon-greedy exploration
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = (epsilon_start - epsilon_end) / max(1, epsilon_decay_steps)
        
        # Evaluation mode (no exploration)
        self.eval_mode = False
        self._eval_epsilon = 0.0
        
        # Cache state: list of file IDs in each slot
        self.cache_slots = [-1] * capacity  # -1 = empty slot
        self.file_to_slot = {}  # Reverse mapping: file_id -> slot
        
        # LRU/LFU counters (for state representation)
        self.lru_counters = np.zeros(capacity, dtype=np.int32)  # Steps since access
        self.lfu_counters = np.zeros(capacity, dtype=np.int32)  # Access frequency
        self.timestep = 0
        
        # Popularity tracking (EMA)
        self.popularity = np.ones(num_files, dtype=np.float32) / num_files
        self.popularity_decay = 0.9
        
        # NOMA-specific tracking
        self.channel_history = deque(maxlen=500)
        self.noma_history = deque(maxlen=500)
        self.cic_count = 0
        self.sic_count = 0
        
        # RL components
        self.use_nn = use_neural_network and TORCH_AVAILABLE
        self.training_step = 0
        self.target_update_freq = target_update_freq
        
        # Action space: evict slot i (0 to capacity-1)
        self.action_dim = capacity
        
        if self.use_nn:
            # State dimension
            self.state_dim = 2 * capacity + 6  # LRU + LFU + 6 global features
            
            # Device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Q-network and target network
            self.q_network = DuelingDQN(self.state_dim, self.action_dim, hidden_dims).to(self.device)
            self.target_network = DuelingDQN(self.state_dim, self.action_dim, hidden_dims).to(self.device)
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.target_network.eval()
            
            # Optimizer (Adam with weight decay for regularization)
            self.optimizer = optim.Adam(
                self.q_network.parameters(), 
                lr=self.lr, 
                weight_decay=1e-5
            )
            
            # Experience replay
            if use_prioritized_replay:
                self.replay_buffer = PrioritizedReplayBuffer(
                    replay_buffer_size, priority_alpha, priority_beta
                )
                self.use_prioritized = True
            else:
                self.replay_buffer = deque(maxlen=replay_buffer_size)
                self.use_prioritized = False
        else:
            # Q-table fallback
            self.q_table = defaultdict(lambda: np.zeros(self.action_dim))
            self.replay_buffer = None
            self.use_prioritized = False
        
        # Training metrics
        self.episode_rewards = []
        self.losses = []
        self.cumulative_reward = 0.0
        
        # For credit assignment
        self.last_state = None
        self.last_action = None
        
        print(f"✅ DQNCache initialized")
        print(f"   Mode: {'Neural Network (DQN)' if self.use_nn else 'Q-table'}")
        print(f"   State dim: {self.state_dim if self.use_nn else 'N/A'}")
        print(f"   Action dim: {self.action_dim}")
        print(f"   Device: {self.device if self.use_nn else 'CPU'}")
        print(f"   NOMA-aware: {self.enable_noma_awareness}")
    
    def _set_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        if TORCH_AVAILABLE:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
    
    # ========================================================================
    # STATE REPRESENTATION (Following RLCaR Paper)
    # ========================================================================
    
    def _get_state_vector(self, requested_file: int) -> np.ndarray:
        """
        Construct state vector from cache status.
        
        State components (following RLCaR paper):
        1. LRU counters (capacity values): timesteps since last access
        2. LFU counters (capacity values): access frequency
        3. Requested file popularity (1 value)
        4. Cache occupancy (1 value)
        5. Mean channel gain (1 value)
        6. Std channel gain (1 value)
        7. CIC success rate (1 value)
        8. NOMA success rate (1 value)
        
        Total: 2*capacity + 6 dimensions
        """
        state = []
        
        # 1. LRU counters (normalize by max timestep)
        max_lru = max(self.lru_counters.max(), 1)
        state.extend((self.lru_counters / max_lru).tolist())
        
        # 2. LFU counters (normalize by max frequency)
        max_lfu = max(self.lfu_counters.max(), 1)
        state.extend((self.lfu_counters / max_lfu).tolist())
        
        # 3. Requested file popularity
        state.append(float(self.popularity[requested_file]))
        
        # 4. Cache occupancy
        occupied = np.sum(self.cache_slots != -1)
        state.append(float(occupied / self.capacity))
        
        # 5-6. Channel quality
        if len(self.channel_history) > 0:
            recent_channels = list(self.channel_history)[-100:]
            state.append(float(np.mean(recent_channels)))
            state.append(float(np.std(recent_channels)))
        else:
            state.extend([0.5, 0.1])  # Default values
        
        # 7-8. NOMA performance (if NOMA-aware)
        if self.enable_noma_awareness and len(self.noma_history) > 0:
            recent_noma = list(self.noma_history)[-100:]
            cic_rate = sum(1 for x in recent_noma if x.get('cic', False)) / len(recent_noma)
            success_rate = sum(1 for x in recent_noma if x.get('success', False)) / len(recent_noma)
            state.append(float(cic_rate))
            state.append(float(success_rate))
        else:
            state.extend([0.0, 0.5])
        
        return np.array(state, dtype=np.float32)
    
    # ========================================================================
    # ACTION SELECTION (Epsilon-Greedy)
    # ========================================================================
    
    def _select_action(self, state: np.ndarray, file_id: int) -> int:
        """
        Epsilon-greedy action selection.
        
        Returns:
            action: slot index to evict (0 to capacity-1), or -1 for no action
        """
        # If file already cached, no action needed
        if file_id in self.file_to_slot:
            return -1  # Signal: no eviction needed
        
        # Find available slots
        empty_slots = [i for i, f in enumerate(self.cache_slots) if f == -1]
        
        # If cache not full, fill empty slot (no eviction)
        if empty_slots:
            return empty_slots[0]  # Use first empty slot
        
        # Cache is full: need to evict
        
        # Epsilon-greedy
        if random.random() < self.epsilon:
            # Exploration: random eviction
            return random.randint(0, self.capacity - 1)
        else:
            # Exploitation: best action from Q-network/table
            if self.use_nn:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    q_values = self.q_network(state_tensor).cpu().numpy()[0]
                    return int(np.argmax(q_values))
            else:
                # Q-table fallback
                state_key = self._discretize_state(state)
                q_values = self.q_table[state_key]
                return int(np.argmax(q_values))
    
    def _discretize_state(self, state: np.ndarray) -> str:
        """
        Discretize continuous state for Q-table (fallback mode).
        
        Maps continuous values to discrete bins: Low/Medium/High
        """
        # Use only first 10 features for simplicity
        key_features = state[:min(10, len(state))]
        
        bins = []
        for val in key_features:
            if val < 0.33:
                bins.append('L')
            elif val < 0.67:
                bins.append('M')
            else:
                bins.append('H')
        
        return ''.join(bins)
    
    # ========================================================================
    # REWARD FUNCTION (NOMA-Aware)
    # ========================================================================
    
    def _compute_reward(
        self,
        cache_hit: bool,
        cic_enabled: bool = False,
        noma_success: bool = True,
        outage: bool = False,
        ber: Optional[float] = None
    ) -> float:
        """
        NOMA-aware reward function.
        
        Reward structure:
        +10: Cache hit (best - no transmission needed)
        +2:  Cache miss + CIC enabled (good - interference cancelled)
        -1:  Cache miss + NOMA success (acceptable - delivered via NOMA)
        -5:  Cache miss + NOMA failure (bad - poor QoS)
        -10: Outage (worst - no communication)
        
        Additional modifiers:
        +1: Very low BER (< 1e-4)
        -2: High BER (> 1e-2)
        """
        if cache_hit:
            return 10.0
        
        # Cache miss cases
        if outage:
            return -10.0
        
        if not noma_success:
            return -5.0
        
        # NOMA succeeded
        if cic_enabled:
            reward = 2.0  # CIC helped!
        else:
            reward = -1.0  # Standard NOMA delivery
        
        # BER-based bonus/penalty
        if ber is not None:
            if ber < 1e-4:
                reward += 1.0  # Excellent quality
            elif ber > 1e-2:
                reward -= 2.0  # Poor quality
        
        return reward
    
    # ========================================================================
    # MAIN REQUEST INTERFACE (NOMA-Aware)
    # ========================================================================
    
    def request(
        self,
        item: int,
        user_id: Optional[int] = None,
        channel_gain: Optional[float] = None,
        paired_user: Optional[int] = None,
        paired_file: Optional[int] = None,
        noma_success: bool = True,
        outage: bool = False,
        ber: Optional[float] = None,
        sinr_weak: Optional[float] = None,
        sinr_strong: Optional[float] = None,
        episode_done: bool = False
    ) -> Dict:
        """
        NOMA-aware request handling with DQN learning.
        
        This is the main entry point for simulations.
        
        Args:
            item: Requested file ID
            user_id: Requesting user ID
            channel_gain: User's channel gain
            paired_user: NOMA paired user ID
            paired_file: File requested by paired user
            noma_success: Whether NOMA transmission succeeded
            outage: Whether outage occurred
            ber: Bit error rate
            sinr_weak: Weak user SINR
            sinr_strong: Strong user SINR
            episode_done: Whether this is the last request in episode
        
        Returns:
            Dict with cache hit status and NOMA benefits
        """
        # Update timestep
        self.timestep += 1
        
        # Check cache hit
        cache_hit = self.is_hit(item, update_stats=True)
        
        # Get NOMA information using base class method
        result = super().request(item, user_id, channel_gain, paired_user, paired_file)
        
        # Update tracking
        if channel_gain is not None:
            self.channel_history.append(float(channel_gain))
        
        if self.enable_noma_awareness:
            self.noma_history.append({
                'cic': result['cic_enabled'],
                'success': noma_success,
                'sinr_weak': sinr_weak,
                'sinr_strong': sinr_strong
            })
            
            if result['cic_enabled']:
                self.cic_count += 1
            if result['strong_user_benefit']:
                self.sic_count += 1
        
        # DQN learning
        self._learn_from_request(
            file_id=item,
            cache_hit=cache_hit,
            cic_enabled=result['cic_enabled'],
            noma_success=noma_success,
            outage=outage,
            ber=ber,
            episode_done=episode_done
        )
        
        # Update LRU/LFU counters
        self._update_counters(item, cache_hit)
        
        # Update popularity (EMA increment on access)
        self.popularity[item] = (
            self.popularity_decay * self.popularity[item]
            + (1.0 - self.popularity_decay) * 1.0
        )
        # Keep others decaying slightly towards 0
        self.popularity *= self.popularity_decay
        self.popularity[item] = max(self.popularity[item], 1e-6)
        self.popularity /= self.popularity.sum()
        
        return result
    
    def _learn_from_request(
        self,
        file_id: int,
        cache_hit: bool,
        cic_enabled: bool,
        noma_success: bool,
        outage: bool,
        ber: Optional[float],
        episode_done: bool
    ):
        """
        DQN learning loop with proper credit assignment.
        
        Flow:
        1. Get current state s_t
        2. Compute reward r_t for previous action a_{t-1}
        3. Store transition (s_{t-1}, a_{t-1}, r_t, s_t, done)
        4. Select new action a_t
        5. Execute action (update cache)
        6. Train network (if time)
        """
        # Get current state
        current_state = self._get_state_vector(file_id)
        
        # Compute reward
        reward = self._compute_reward(cache_hit, cic_enabled, noma_success, outage, ber)
        self.cumulative_reward += reward
        
        # Store experience from PREVIOUS action (only if a valid action was taken)
        if self.last_state is not None and self.last_action is not None and self.last_action >= 0:
            experience = {
                'state': self.last_state,
                'action': self.last_action,
                'reward': reward,
                'next_state': current_state,
                'done': episode_done
            }
            
            if self.use_nn and self.replay_buffer is not None:
                if self.use_prioritized:
                    self.replay_buffer.add(experience)
                else:
                    self.replay_buffer.append(experience)
            elif not self.use_nn:
                self._update_q_table(experience)
        
        # Select action for CURRENT request (only if miss)
        action = -1
        if not cache_hit:
            action = self._select_action(current_state, file_id)
            
            # Execute action (update cache) if valid
            if action >= 0:
                self._execute_action(action, file_id)
        
        # Save for next iteration ONLY if a valid action was taken
        if action >= 0:
            self.last_state = current_state
            self.last_action = action
        
        # Training
        if not self.eval_mode:
            self.training_step += 1
            
            # Train network periodically
            if self.use_nn and self.training_step % self.train_freq == 0:
                if len(self.replay_buffer) >= self.batch_size:
                    self._train_step()
            
            # Update target network
            if self.use_nn and self.training_step % self.target_update_freq == 0:
                self._soft_update_target()
            
            # Decay epsilon
            if self.epsilon > self.epsilon_end:
                self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)
        
        # Episode reset
        if episode_done:
            self.last_state = None
            self.last_action = None
            self.episode_rewards.append(self.cumulative_reward)
            self.cumulative_reward = 0.0
    
    def _execute_action(self, action: int, file_id: int):
        """
        Execute cache replacement action.
        
        Args:
            action: Slot index to replace
            file_id: New file to cache
        """
        if action < 0 or action >= self.capacity:
            return
        
        # Remove old file from slot
        old_file = self.cache_slots[action]
        if old_file != -1 and old_file in self.file_to_slot:
            del self.file_to_slot[old_file]
        
        # Add new file
        self.cache_slots[action] = file_id
        self.file_to_slot[file_id] = action
        
        # Initialize counters for this slot
        self.lru_counters[action] = 0
        self.lfu_counters[action] = 1
    
    def _update_counters(self, file_id: int, cache_hit: bool):
        """
        Update LRU/LFU counters.
        
        LRU: Increment all counters, reset accessed file on hit
        LFU: Increment accessed file counter on hit
        """
        # Increment all LRU counters
        self.lru_counters += 1
        
        # On cache hit, update the accessed file's counters
        if cache_hit and file_id in self.file_to_slot:
            slot = self.file_to_slot[file_id]
            self.lru_counters[slot] = 0  # Reset LRU
            self.lfu_counters[slot] += 1  # Increment LFU
    
    # ========================================================================
    # TRAINING (Double DQN with Prioritized Replay)
    # ========================================================================
    
    def _train_step(self):
        """
        Single DQN training step.
        
        Uses Double DQN to reduce overestimation:
        - Policy network selects best action
        - Target network evaluates that action
        """
        if not self.use_nn or len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch
        if self.use_prioritized:
            experiences, weights, indices = self.replay_buffer.sample(self.batch_size)
            if experiences is None:
                return
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            experiences = random.sample(self.replay_buffer, self.batch_size)
            weights = torch.ones(self.batch_size).to(self.device)
            indices = None
        
        # Prepare tensors
        states = torch.FloatTensor([e['state'] for e in experiences]).to(self.device)
        actions = torch.LongTensor([e['action'] for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e['reward'] for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e['next_state'] for e in experiences]).to(self.device)
        dones = torch.FloatTensor([e['done'] for e in experiences]).to(self.device)
        
        # Current Q-values: Q(s,a)
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN: select action with policy net, evaluate with target net
        with torch.no_grad():
            # Best actions according to policy network
            next_actions = self.q_network(next_states).argmax(1)
            
            # Q-values from target network
            next_q = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
            # Target: r + γ * Q_target(s', argmax_a Q_policy(s',a))
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss with importance sampling weights
        td_errors = target_q - current_q
        loss = (weights * F.smooth_l1_loss(current_q, target_q, reduction='none')).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.gradient_clip)
        
        self.optimizer.step()
        
        # Update priorities
        if self.use_prioritized and indices is not None:
            self.replay_buffer.update_priorities(
                indices, td_errors.detach().cpu().numpy()
            )
        
        self.losses.append(float(loss.item()))
    
    def _soft_update_target(self):
        """
        Soft update of target network.
        
        θ_target ← τ * θ_policy + (1 - τ) * θ_target
        
        Prevents target from changing too quickly (stabilizes training).
        """
        for target_param, param in zip(self.target_network.parameters(), 
                                      self.q_network.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
    
    def _update_q_table(self, experience: Dict):
        """
        Q-learning update for Q-table (fallback mode).
        
        Q(s,a) ← Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]
        """
        state_key = self._discretize_state(experience['state'])
        next_key = self._discretize_state(experience['next_state'])
        
        action = experience['action']
        reward = experience['reward']
        done = experience['done']
        
        # Current Q-value
        current_q = self.q_table[state_key][action]
        
        # Best next Q-value
        if done:
            next_max_q = 0.0
        else:
            next_max_q = np.max(self.q_table[next_key])
        
        # Q-learning update
        target_q = reward + self.gamma * next_max_q
        self.q_table[state_key][action] += self.lr * (target_q - current_q)
    
    # ========================================================================
    # CACHE INTERFACE METHODS
    # ========================================================================
    
    def populate(self, items: Optional[Iterable[int]] = None):
        """
        Initialize cache with most popular files.
        
        Args:
            items: Optional list of file IDs to cache (in priority order)
        """
        if items is None:
            # Use popularity
            top_files = np.argsort(-self.popularity)[:self.capacity]
        else:
            top_files = list(items)[:self.capacity]
        
        self.cache_slots = [-1] * self.capacity
        self.file_to_slot.clear()
        
        for slot, file_id in enumerate(top_files):
            self.cache_slots[slot] = int(file_id)
            self.file_to_slot[int(file_id)] = slot
            self.lfu_counters[slot] = 1
            self.lru_counters[slot] = 0
    
    def is_hit(self, item: int, update_stats: bool = True) -> bool:
        """
        Check if item is in cache.
        
        Args:
            item: File ID
            update_stats: Whether to update hit/miss statistics
        
        Returns:
            True if cache hit, False otherwise
        """
        hit = int(item) in self.file_to_slot
        
        if update_stats:
            if hit:
                self._record_hit()
            else:
                self._record_miss()
        
        return hit
    
    def get_contents(self) -> Set[int]:
        """Get current cache contents."""
        return set(f for f in self.cache_slots if f != -1)
    
    def clear(self):
        """Clear cache and reset state."""
        self.cache_slots = [-1] * self.capacity
        self.file_to_slot.clear()
        self.lru_counters = np.zeros(self.capacity, dtype=np.int32)
        self.lfu_counters = np.zeros(self.capacity, dtype=np.int32)
        self.last_state = None
        self.last_action = None
        self.reset_stats()
    
    def set_eval_mode(self, eval_mode: bool = True):
        """
        Set evaluation mode (no exploration, no training).
        
        Args:
            eval_mode: True for evaluation, False for training
        """
        self.eval_mode = eval_mode
        
        if eval_mode:
            self._eval_epsilon = self.epsilon
            self.epsilon = 0.0  # No exploration
            if self.use_nn:
                self.q_network.eval()
        else:
            self.epsilon = self._eval_epsilon
            if self.use_nn:
                self.q_network.train()
    
    # ========================================================================
    # MODEL PERSISTENCE
    # ========================================================================
    
    def save_model(self, filepath: str):
        """Save learned model to file."""
        if self.use_nn:
            torch.save({
                'q_network': self.q_network.state_dict(),
                'target_network': self.target_network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'training_step': self.training_step,
                'epsilon': self.epsilon,
                'popularity': self.popularity,
                'cache_slots': self.cache_slots,
                'file_to_slot': self.file_to_slot,
                'lru_counters': self.lru_counters,
                'lfu_counters': self.lfu_counters
            }, filepath)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'q_table': dict(self.q_table),
                    'training_step': self.training_step,
                    'epsilon': self.epsilon,
                    'popularity': self.popularity,
                    'cache_slots': self.cache_slots,
                    'file_to_slot': self.file_to_slot
                }, f)
        
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load learned model from file."""
        if self.use_nn:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network'])
            self.target_network.load_state_dict(checkpoint['target_network'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.training_step = checkpoint['training_step']
            self.epsilon = checkpoint['epsilon']
            self.popularity = checkpoint['popularity']
            self.cache_slots = checkpoint['cache_slots']
            self.file_to_slot = checkpoint['file_to_slot']
            self.lru_counters = checkpoint.get('lru_counters', self.lru_counters)
            self.lfu_counters = checkpoint.get('lfu_counters', self.lfu_counters)
        else:
            with open(filepath, 'rb') as f:
                checkpoint = pickle.load(f)
            self.q_table = defaultdict(lambda: np.zeros(self.action_dim), checkpoint['q_table'])
            self.training_step = checkpoint['training_step']
            self.epsilon = checkpoint['epsilon']
            self.popularity = checkpoint['popularity']
            self.cache_slots = checkpoint['cache_slots']
            self.file_to_slot = checkpoint['file_to_slot']
        
        print(f"✅ Model loaded from {filepath}")
    
    # ========================================================================
    # STATISTICS
    # ========================================================================
    
    def get_stats(self) -> Dict:
        """Get comprehensive statistics."""
        base_stats = super().stats()
        
        dqn_stats = {
            'training_step': self.training_step,
            'epsilon': self.epsilon,
            'eval_mode': self.eval_mode,
            'avg_episode_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0,
            'replay_buffer_size': len(self.replay_buffer) if self.replay_buffer else 0,
            'use_neural_network': self.use_nn,
            'cic_count': self.cic_count,
            'sic_count': self.sic_count
        }
        
        return {**base_stats, **dqn_stats}


# Alias for compatibility
StableDQNCache = DQNCache
