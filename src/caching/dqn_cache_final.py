"""
src/caching/dqn_cache_final.py

STABLE DEEP Q-NETWORK CACHE FOR NOMA SYSTEMS
==============================================
"""

import numpy as np
import random
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional
import pickle

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available - using Q-table fallback")

from src.caching.cache_base import CacheBase


# ============================================================================
# NEURAL NETWORK ARCHITECTURE
# ============================================================================

class DQNNetwork(nn.Module):
    """
    Dueling DQN Architecture with proper initialization.
    
    Architecture:
    - Input: state features
    - Shared layers: 2 hidden layers with ReLU + LayerNorm
    - Value stream: estimates state value V(s)
    - Advantage stream: estimates advantage A(s,a)
    - Output: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [128, 64]):
        super(DQNNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Shared feature extraction
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)  # Stabilizes training
            ])
            input_dim = hidden_dim
        
        self.feature_layer = nn.Sequential(*layers)
        
        # Dueling architecture
        mid_dim = hidden_dims[-1] // 2
        
        # Value stream: V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dims[-1], mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, 1)
        )
        
        # Advantage stream: A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dims[-1], mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, action_dim)
        )
        
        # Xavier initialization for stable gradients
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier/He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through dueling architecture.
        
        Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))
        """
        features = self.feature_layer(state)
        
        values = self.value_stream(features)  # (batch, 1)
        advantages = self.advantage_stream(features)  # (batch, actions)
        
        # Dueling aggregation: subtract mean advantage
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values


# ============================================================================
# PRIORITIZED REPLAY BUFFER
# ============================================================================

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer.
    
    Samples experiences based on TD-error priority, allowing the agent
    to learn more from surprising transitions.
    """
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta = beta  # Importance sampling exponent
        
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        
        self.max_priority = 1.0
        self.min_priority = 0.01
    
    def add(self, experience: Dict):
        """Add experience with maximum priority (optimistic initialization)."""
        self.buffer.append(experience)
        self.priorities.append(self.max_priority)
    
    def sample(self, batch_size: int) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
        """
        Sample batch with prioritized sampling.
        
        Returns:
            experiences: List of experience dictionaries
            weights: Importance sampling weights
            indices: Indices of sampled experiences
        """
        if len(self.buffer) < batch_size:
            return None, None, None
        
        # Convert priorities to probabilities
        priorities = np.array(self.priorities, dtype=np.float64)
        priorities = np.clip(priorities, self.min_priority, None)
        
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), size=batch_size, p=probs, replace=False)
        
        # Compute importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize
        
        experiences = [self.buffer[idx] for idx in indices]
        
        return experiences, weights, indices
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on TD errors."""
        for idx, error in zip(indices, td_errors):
            priority = abs(error) + 1e-6  # Small constant for numerical stability
            priority = float(np.clip(priority, self.min_priority, 100.0))
            
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = priority
                self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)


# ============================================================================
# MAIN DQN CACHE CLASS
# ============================================================================

class StableDQNCache(CacheBase):
    """
    Stable Deep Q-Network Cache for NOMA Systems.
    
    Key Features:
    - Proper credit assignment (fixed reward-action timing)
    - Balanced reward structure
    - Simplified state representation
    - Slot-based action space (replace file in slot i)
    - Prioritized experience replay
    - Double DQN for stability
    - Gradient clipping
    - Evaluation mode
    
    State: [popularity features, cache occupancy, channel quality, NOMA metrics, cache content]
    Action: Select which slot to replace (0 to capacity-1) or do nothing (capacity)
    Reward: Balanced reward based on cache hit/miss and NOMA performance
    """
    
    def __init__(
        self,
        capacity: int,
        num_files: int,
        num_users: int,
        
        # Learning parameters
        learning_rate: float = 0.0001,
        gamma: float = 0.95,
        
        # Exploration parameters
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay_steps: int = 50000,
        
        # Network parameters
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
        
        # Stability parameters
        gradient_clip: float = 10.0,
        tau: float = 0.005,  # Soft update parameter
        
        seed: int = 2025
    ):
        super().__init__(capacity)
        
        # Validate inputs
        assert 0 < capacity <= num_files
        assert 0 < gamma <= 1
        assert 0 < learning_rate < 1
        assert epsilon_start >= epsilon_end > 0
        
        self.num_files = num_files
        self.num_users = num_users
        self.lr = learning_rate
        self.gamma = gamma
        self.seed = seed
        
        # Set random seeds
        self._set_seeds(seed)
        
        # Epsilon schedule
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_decay_rate = (epsilon_start - epsilon_end) / max(1, epsilon_decay_steps)
        
        # Evaluation mode
        self.eval_mode = False
        self._stored_epsilon = self.epsilon
        
        # Cache structure: list of file IDs in each slot
        self.cache_slots = [-1] * capacity  # -1 means empty
        self.file_to_slot = {}  # Reverse mapping
        
        # State tracking
        self.popularity_ema = np.ones(num_files, dtype=np.float32) / num_files
        self.popularity_alpha = 0.1
        
        self.request_history = deque(maxlen=1000)
        self.channel_history = deque(maxlen=200)
        self.noma_history = deque(maxlen=200)
        
        # Training setup
        self.use_nn = use_neural_network and TORCH_AVAILABLE
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.train_freq = train_freq
        self.gradient_clip = gradient_clip
        self.tau = tau
        
        self.training_step = 0
        self.update_counter = 0
        
        # Action space: 0 to capacity-1 = replace slot i, capacity = do nothing
        self.action_dim = capacity + 1
        
        # RL components
        if self.use_nn:
            self.state_dim = self._compute_state_dim()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Q-network and target network
            self.q_network = DQNNetwork(self.state_dim, self.action_dim, hidden_dims).to(self.device)
            self.target_network = DQNNetwork(self.state_dim, self.action_dim, hidden_dims).to(self.device)
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.target_network.eval()
            
            # Optimizer with weight decay for regularization
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr, weight_decay=1e-5)
            
            # Replay buffer
            if use_prioritized_replay:
                self.replay_buffer = PrioritizedReplayBuffer(replay_buffer_size, priority_alpha, priority_beta)
            else:
                self.replay_buffer = deque(maxlen=replay_buffer_size)
                self.use_prioritized = False
        else:
            # Q-table fallback
            self.q_table = defaultdict(lambda: np.zeros(self.action_dim))
            self.replay_buffer = None
        
        # Metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_losses = []
        self.cumulative_reward = 0.0
        
        # For proper credit assignment
        self.last_state = None
        self.last_action = None
        
        print(f"✅ StableDQNCache initialized:")
        print(f"   Neural Network: {self.use_nn}")
        print(f"   State Dim: {self.state_dim if self.use_nn else 'N/A'}")
        print(f"   Action Dim: {self.action_dim}")
        print(f"   Device: {self.device if self.use_nn else 'CPU'}")
    
    def _set_seeds(self, seed: int):
        """Set all random seeds."""
        random.seed(seed)
        np.random.seed(seed)
        if TORCH_AVAILABLE:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
    
    def _compute_state_dim(self) -> int:
        """
        Compute state dimension.
        
        State components:
        - 5 popularity features (top-5 file popularities)
        - 1 cache occupancy
        - 2 channel quality (mean, std)
        - 2 NOMA performance (success rate, avg SINR)
        - capacity slot features (popularity of cached files)
        
        Total: 10 + capacity
        """
        return 10 + self.capacity
    
    def set_eval_mode(self, eval_mode: bool = True):
        """
        Set evaluation mode (no exploration).
        Call before testing to get true policy performance.
        """
        self.eval_mode = eval_mode
        if eval_mode:
            self._stored_epsilon = self.epsilon
            self.epsilon = 0.0
            if self.use_nn:
                self.q_network.eval()
        else:
            self.epsilon = self._stored_epsilon
            if self.use_nn:
                self.q_network.train()
    
    def _get_state_vector(self, file_id: int) -> np.ndarray:
        """
        Extract state features.
        
        Returns normalized feature vector representing current state.
        """
        features = []
        
        # 1. Top-5 file popularities (5 features)
        top_5_idx = np.argsort(-self.popularity_ema)[:5]
        features.extend(self.popularity_ema[top_5_idx])
        
        # 2. Cache occupancy (1 feature)
        occupied = sum(1 for x in self.cache_slots if x != -1)
        features.append(occupied / self.capacity)
        
        # 3. Channel quality (2 features)
        if len(self.channel_history) > 0:
            recent_channels = list(self.channel_history)[-50:]
            features.append(np.mean(recent_channels))
            features.append(np.std(recent_channels))
        else:
            features.extend([0.5, 0.1])
        
        # 4. NOMA performance (2 features)
        if len(self.noma_history) > 0:
            recent_noma = list(self.noma_history)[-50:]
            success_rate = np.mean([x['success'] for x in recent_noma])
            avg_sinr = np.mean([x['sinr'] for x in recent_noma])
            features.append(success_rate)
            features.append(avg_sinr / 10.0)  # Normalize SINR
        else:
            features.extend([0.5, 0.0])
        
        # 5. Cache slot popularities (capacity features)
        slot_popularities = [
            self.popularity_ema[f] if f != -1 else 0.0
            for f in self.cache_slots
        ]
        features.extend(slot_popularities)
        
        return np.array(features, dtype=np.float32)
    
    def _compute_reward(
        self,
        cache_hit: bool,
        noma_success: Optional[bool] = None,
        ber: Optional[float] = None,
        outage: bool = False
    ) -> float:
        """
        BALANCED reward function.
        
        Reward structure:
        - Cache hit: +10 (good outcome, no transmission needed)
        - Cache miss + NOMA success: -1 (acceptable, content delivered)
        - Cache miss + NOMA failure: -5 (bad, outage occurred)
        - Cache miss + high BER: -3 (moderate penalty for poor quality)
        
        This creates a clear learning signal without huge imbalances.
        """
        if cache_hit:
            return 10.0
        
        # Cache miss cases
        if outage or (noma_success is not None and not noma_success):
            return -5.0
        
        if noma_success:
            reward = -1.0
            
            # Additional penalty for high BER
            if ber is not None:
                if ber > 0.01:  # High BER
                    reward -= 2.0
                elif ber < 0.0001:  # Very good BER
                    reward += 1.0
            
            return reward
        
        # Unknown outcome (shouldn't happen in normal operation)
        return -1.0
    
    def _select_action(self, state: np.ndarray, file_id: int) -> int:
        """
        Epsilon-greedy action selection.
        
        Returns:
            action: 0 to capacity-1 = replace slot i
                    capacity = do nothing (don't cache this file)
        """
        # Check if file already cached
        if file_id in self.file_to_slot:
            return self.capacity  # Do nothing, already cached
        
        # Epsilon-greedy
        if random.random() < self.epsilon:
            # Exploration: random action
            # Bias towards replacing occupied slots
            empty_slots = [i for i, f in enumerate(self.cache_slots) if f == -1]
            occupied_slots = [i for i, f in enumerate(self.cache_slots) if f != -1]
            
            if empty_slots and random.random() < 0.7:  # Prefer empty slots
                return random.choice(empty_slots)
            elif occupied_slots:
                return random.choice(occupied_slots)
            else:
                return self.capacity  # Do nothing
        else:
            # Exploitation: best action from Q-network/table
            if self.use_nn:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    q_values = self.q_network(state_tensor).cpu().numpy()[0]
                    
                    # Mask invalid actions (trying to replace empty slot when better options exist)
                    # This helps learning by focusing on meaningful actions
                    return int(np.argmax(q_values))
            else:
                # Q-table
                state_key = self._discretize_state(state)
                q_values = self.q_table[state_key]
                return int(np.argmax(q_values))
    
    def _execute_action(self, action: int, file_id: int):
        """
        Execute the selected action.
        
        Actions:
        - 0 to capacity-1: Replace file in slot i with file_id
        - capacity: Do nothing
        """
        if action == self.capacity:
            return  # Do nothing
        
        if action < 0 or action >= self.capacity:
            return  # Invalid action
        
        slot = action
        
        # Remove old file from slot
        old_file = self.cache_slots[slot]
        if old_file != -1 and old_file in self.file_to_slot:
            del self.file_to_slot[old_file]
        
        # Add new file
        self.cache_slots[slot] = file_id
        self.file_to_slot[file_id] = slot
    
    def observe_request(
        self,
        user_id: int,
        file_id: int,
        cache_hit: bool,
        noma_success: Optional[bool] = None,
        channel_gain: Optional[float] = None,
        sinr_weak: Optional[float] = None,
        sinr_strong: Optional[float] = None,
        ber: Optional[float] = None,
        outage: bool = False,
        episode_done: bool = False
    ):
        """
        Main learning loop with PROPER credit assignment.
        
        Correct flow:
        1. Observe outcome of PREVIOUS action (compute reward)
        2. Store transition (s_t-1, a_t-1, r_t, s_t, done)
        3. Get current state s_t
        4. Select and execute action a_t
        5. Update last_state and last_action for next iteration
        """
        # Update statistics
        self.popularity_ema[file_id] = (
            self.popularity_alpha + (1 - self.popularity_alpha) * self.popularity_ema[file_id]
        )
        
        if channel_gain is not None:
            self.channel_history.append(float(channel_gain))
        
        if noma_success is not None and sinr_weak is not None:
            self.noma_history.append({
                'success': bool(noma_success),
                'sinr': float(sinr_weak if sinr_weak is not None else 0.0)
            })
        
        self.request_history.append({
            'file_id': file_id,
            'user_id': user_id,
            'cache_hit': cache_hit
        })
        
        # STEP 1: Get current state
        current_state = self._get_state_vector(file_id)
        
        # STEP 2: Compute reward for PREVIOUS action
        reward = self._compute_reward(cache_hit, noma_success, ber, outage)
        self.cumulative_reward += reward
        
        # STEP 3: Store transition for PREVIOUS action
        if self.last_state is not None and self.last_action is not None:
            experience = {
                'state': self.last_state,
                'action': self.last_action,
                'reward': reward,
                'next_state': current_state,
                'done': episode_done
            }
            
            if self.use_nn and self.replay_buffer is not None:
                if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
                    self.replay_buffer.add(experience)
                else:
                    self.replay_buffer.append(experience)
            elif not self.use_nn:
                # Q-table update
                self._update_q_table(experience)
        
        # STEP 4: Select action for CURRENT request
        action = self._select_action(current_state, file_id)
        
        # STEP 5: Execute action
        if not cache_hit:  # Only cache on miss
            self._execute_action(action, file_id)
        
        # STEP 6: Save for next iteration
        self.last_state = current_state
        self.last_action = action
        
        # Training step
        self.training_step += 1
        
        if self.use_nn and not self.eval_mode:
            # Train periodically
            if self.training_step % self.train_freq == 0 and len(self.replay_buffer) >= self.batch_size:
                self._train_step()
            
            # Update target network
            if self.training_step % self.target_update_freq == 0:
                self._soft_update_target()
        
        # Decay epsilon
        if not self.eval_mode and self.epsilon > self.epsilon_end:
            self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay_rate)
        
        # Episode end handling
        if episode_done:
            self.last_state = None
            self.last_action = None
    
    def _train_step(self):
        """Perform one training step."""
        if not self.use_nn or len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch
        if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
            experiences, weights, indices = self.replay_buffer.sample(self.batch_size)
            if experiences is None:
                return
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            experiences = random.sample(self.replay_buffer, self.batch_size)
            weights = torch.ones(self.batch_size).to(self.device)
            indices = None
        
        # Prepare batch
        states = torch.FloatTensor([exp['state'] for exp in experiences]).to(self.device)
        actions = torch.LongTensor([exp['action'] for exp in experiences]).to(self.device)
        rewards = torch.FloatTensor([exp['reward'] for exp in experiences]).to(self.device)
        next_states = torch.FloatTensor([exp['next_state'] for exp in experiences]).to(self.device)
        dones = torch.FloatTensor([exp['done'] for exp in experiences]).to(self.device)
        
        # Current Q-values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN: select action with policy network, evaluate with target network
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
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
        if isinstance(self.replay_buffer, PrioritizedReplayBuffer) and indices is not None:
            self.replay_buffer.update_priorities(indices, td_errors.detach().cpu().numpy())
        
        self.training_losses.append(float(loss.item()))
    
    def _soft_update_target(self):
        """Soft update target network: θ' ← τθ + (1-τ)θ'"""
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def _update_q_table(self, experience: Dict):
        """Q-table update (fallback when no neural network)."""
        state_key = self._discretize_state(experience['state'])
        next_state_key = self._discretize_state(experience['next_state'])
        
        action = experience['action']
        reward = experience['reward']
        
        current_q = self.q_table[state_key][action]
        next_max_q = np.max(self.q_table[next_state_key])
        
        # Q-learning update
        new_q = current_q + self.lr * (reward + self.gamma * next_max_q - current_q)
        self.q_table[state_key][action] = new_q
    
    def _discretize_state(self, state: np.ndarray) -> str:
        """Discretize continuous state for Q-table (fallback)."""
        discretized = []
        for val in state[:10]:  # Only use first 10 features
            if val < 0.33:
                discretized.append('L')
            elif val < 0.67:
                discretized.append('M')
            else:
                discretized.append('H')
        return ''.join(discretized)
    
    # ========================================================================
    # CACHE INTERFACE METHODS
    # ========================================================================
    
    def populate(self, items=None):
        """Initial cache population based on popularity."""
        if items is None:
            top_indices = np.argsort(-self.popularity_ema)[:self.capacity]
        else:
            top_indices = items[:self.capacity]
        
        self.cache_slots = [-1] * self.capacity
        self.file_to_slot.clear()
        
        for slot, file_id in enumerate(top_indices):
            self.cache_slots[slot] = int(file_id)
            self.file_to_slot[int(file_id)] = slot
    
    def is_hit(self, item: int) -> bool:
        """Check if item is in cache."""
        return int(item) in self.file_to_slot
    
    def clear(self):
        """Clear cache."""
        self.cache_slots = [-1] * self.capacity
        self.file_to_slot.clear()
        self.last_state = None
        self.last_action = None
    
    def get_stats(self) -> Dict:
        """Return learning statistics."""
        return {
            'use_neural_network': self.use_nn,
            'training_step': self.training_step,
            'epsilon': self.epsilon,
            'cumulative_reward': self.cumulative_reward,
            'replay_buffer_size': len(self.replay_buffer) if self.replay_buffer else 0,
            'cache_occupancy': sum(1 for x in self.cache_slots if x != -1),
            'avg_loss': np.mean(self.training_losses[-100:]) if self.training_losses else 0.0,
            'eval_mode': self.eval_mode
        }
    
    def save_model(self, filepath: str):
        """Save learned model."""
        if self.use_nn:
            torch.save({
                'q_network': self.q_network.state_dict(),
                'target_network': self.target_network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'training_step': self.training_step,
                'epsilon': self.epsilon,
                'cumulative_reward': self.cumulative_reward,
                'cache_slots': self.cache_slots,
                'file_to_slot': self.file_to_slot
            }, filepath)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'q_table': dict(self.q_table),
                    'training_step': self.training_step,
                    'epsilon': self.epsilon,
                    'cumulative_reward': self.cumulative_reward,
                    'cache_slots': self.cache_slots,
                    'file_to_slot': self.file_to_slot
                }, f)
    
    def load_model(self, filepath: str):
        """Load learned model."""
        if self.use_nn:
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
            self.q_network.load_state_dict(checkpoint['q_network'])
            self.target_network.load_state_dict(checkpoint['target_network'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.training_step = checkpoint.get('training_step', 0)
            self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
            self.cumulative_reward = checkpoint.get('cumulative_reward', 0.0)
            self.cache_slots = checkpoint.get('cache_slots', self.cache_slots)
            self.file_to_slot = checkpoint.get('file_to_slot', {})
        else:
            with open(filepath, 'rb') as f:
                checkpoint = pickle.load(f)
            self.q_table = defaultdict(lambda: np.zeros(self.action_dim), checkpoint['q_table'])
            self.training_step = checkpoint.get('training_step', 0)
            self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
            self.cumulative_reward = checkpoint.get('cumulative_reward', 0.0)
            self.cache_slots = checkpoint.get('cache_slots', self.cache_slots)
            self.file_to_slot = checkpoint.get('file_to_slot', {})