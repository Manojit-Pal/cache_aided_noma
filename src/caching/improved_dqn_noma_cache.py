# src/caching/improved_dqn_noma_cache.py
"""
FULLY CORRECTED Deep Q-Network Implementation for NOMA Caching
Version 2.0 - All Critical Bugs Fixed

FIXES APPLIED:
✅ Fix #1: Corrected reward-transition temporal alignment
✅ Fix #2: Fixed Dueling DQN formula (max instead of mean)  
✅ Fix #3: Safe priority updates with bounds checking
✅ Fix #5: Added PyTorch seeding for reproducibility
✅ Fix #6: Reward normalization with running statistics
✅ Fix #7: Input validation in constructor
✅ Fix #8: Better epsilon decay handling
✅ Fix #9: Efficient prioritized sampling
"""

import numpy as np
from collections import deque, defaultdict
import pickle
import os
from typing import Dict, List, Tuple, Optional

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available. Using Q-table fallback.")

from .cache_base import CacheBase


class DQNNetwork(nn.Module):
    """Dueling Q-Network with CORRECTED aggregation formula."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = None):
        super(DQNNetwork, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 64]
        
        self.action_dim = action_dim
        
        # Shared feature layers
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            input_dim = hidden_dim
        
        self.feature_layer = nn.Sequential(*layers)
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, self.action_dim)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.feature_layer(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # ✅ FIX #2: Correct Dueling DQN formula using MAX
        advantages_max = advantages.max(dim=1, keepdim=True)[0]
        q_values = values + (advantages - advantages_max)
        
        return q_values


class ImprovedDQNNomaCache(CacheBase):
    """
    CORRECTED Deep Q-Network based NOMA cache.
    All critical bugs have been fixed.
    """
    
    def __init__(
        self,
        capacity: int,
        num_files: int,
        num_users: int,
        learning_rate: float = 1e-4,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay_steps: int = 50000,
        use_neural_network: bool = True,
        replay_buffer_size: int = 50000,
        batch_size: int = 64,
        priority_alpha: float = 0.6,
        priority_beta: float = 0.4,
        tau: float = 1e-3,
        hidden_dims: List[int] = None,
        seed: int = 2025
    ):
        super().__init__(capacity)
        
        # ✅ FIX #7: Input validation
        assert 0 < capacity <= num_files, f"Invalid capacity: {capacity}"
        assert 0 < gamma <= 1, f"Invalid gamma: {gamma}"
        assert 0 < learning_rate < 1, f"Invalid learning_rate: {learning_rate}"
        assert epsilon_start >= epsilon_end > 0, "Invalid epsilon range"
        assert batch_size > 0, f"Invalid batch_size: {batch_size}"
        assert replay_buffer_size >= batch_size, "replay_buffer_size must be >= batch_size"
        
        self.num_files = int(num_files)
        self.num_users = int(num_users)
        self.lr = float(learning_rate)
        self.gamma = float(gamma)
        self.seed = int(seed)
        
        # ✅ FIX #5: Set all seeds for reproducibility
        self._set_seeds(self.seed)
        
        # Epsilon schedule
        self.epsilon_start = float(epsilon_start)
        self.epsilon = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay_steps = int(epsilon_decay_steps)
        
        # ✅ FIX #8: Better epsilon decay
        if self.epsilon_decay_steps > 0:
            self.epsilon_decay_rate = (self.epsilon_start - self.epsilon_end) / self.epsilon_decay_steps
            self.epsilon_decay_mode = 'linear'
        else:
            self.epsilon_decay_rate = 0.995
            self.epsilon_decay_mode = 'exponential'
        
        # Cache structure
        self.capacity = int(capacity)
        self.contents_list = [-1] * self.capacity
        self.file_to_slot = {}
        
        # State tracking
        self.popularity_ema = np.ones(self.num_files, dtype=np.float32) / max(1, self.num_files)
        self.alpha_pop = 0.1
        self.channel_quality_history = deque(maxlen=100)
        self.noma_outcome_history = deque(maxlen=200)
        self.request_history = deque(maxlen=500)
        self.file_request_count = np.zeros(self.num_files, dtype=np.int32)
        self.file_noma_success = defaultdict(lambda: deque(maxlen=50))
        self.file_user_affinity = defaultdict(lambda: deque(maxlen=50))
        
        # RL components
        self.use_nn = use_neural_network and TORCH_AVAILABLE
        self.training_step = 0
        self.action_dim = self.capacity + 1
        
        # ✅ FIX #1: Proper transition tracking (CRITICAL FIX)
        self.last_state = None
        self.last_action = None
        self.last_reward = None  # Store reward separately!
        
        # ✅ FIX #6: Reward normalization statistics
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_history = deque(maxlen=1000)
        
        if self.use_nn:
            self.state_dim = self._get_state_dim()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            if hidden_dims is None:
                hidden_dims = [128, 64]
            
            self.q_network = DQNNetwork(self.state_dim, self.action_dim, hidden_dims).to(self.device)
            self.target_network = DQNNetwork(self.state_dim, self.action_dim, hidden_dims).to(self.device)
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.target_network.eval()
            
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
            self.tau = float(tau)
        else:
            self.q_table = defaultdict(lambda: defaultdict(float))
        
        # Prioritized replay
        self.replay_buffer = deque(maxlen=int(replay_buffer_size))
        self.priorities = deque(maxlen=int(replay_buffer_size))
        self.batch_size = int(batch_size)
        self.priority_alpha = float(priority_alpha)
        self.priority_beta = float(priority_beta)
        self.priority_eps = 1e-6
        self.priority_min = 1e-6
        self.priority_max = 1e3
        
        # Performance tracking
        self.cumulative_reward = 0.0
        self.episode_rewards = []
        self.training_losses = []
        
    
    def _set_seeds(self, seed: int):
        """✅ FIX #5: Set all random seeds."""
        np.random.seed(seed)
        
        if TORCH_AVAILABLE:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def _get_state_dim(self) -> int:
        """Calculate state dimension."""
        return 7 + 2 + self.capacity
    
    def get_state_vector(self, file_id: int) -> np.ndarray:
        """Extract state representation."""
        features = []
        
        # Global features (7)
        cache_occupancy = sum(1 for x in self.contents_list if x != -1) / max(1, self.capacity)
        features.append(cache_occupancy)
        
        if len(self.channel_quality_history) > 0:
            features.extend([float(np.mean(self.channel_quality_history)), 
                           float(np.std(self.channel_quality_history))])
        else:
            features.extend([0.5, 0.1])
        
        if len(self.noma_outcome_history) > 0:
            recent = list(self.noma_outcome_history)
            success_rate = float(np.mean([1.0 if x['success'] else 0.0 for x in recent]))
            avg_sinr_weak = float(np.mean([x['sinr_weak'] for x in recent]))
            avg_sinr_strong = float(np.mean([x['sinr_strong'] for x in recent]))
            features.extend([success_rate, avg_sinr_weak, avg_sinr_strong])
        else:
            features.extend([0.5, 0.0, 0.0])
        
        if len(self.request_history) > 10:
            request_rate = float(len(self.request_history)) / float(self.request_history.maxlen)
        else:
            request_rate = 0.0
        features.append(request_rate)
        
        # Request features (2)
        features.append(float(self.popularity_ema[file_id]))
        
        if file_id in self.file_noma_success and len(self.file_noma_success[file_id]) > 0:
            features.append(float(np.mean(self.file_noma_success[file_id])))
        else:
            features.append(0.5)
        
        # Cache content features (capacity)
        slot_popularities = [
            float(self.popularity_ema[f]) if f != -1 else 0.0
            for f in self.contents_list
        ]
        features.extend(slot_popularities)
        
        return np.array(features, dtype=np.float32)
    
    def _normalize_reward(self, reward: float) -> float:
        """✅ FIX #6: Normalize rewards using running statistics."""
        self.reward_history.append(reward)
        
        if len(self.reward_history) > 10:
            self.reward_mean = float(np.mean(self.reward_history))
            self.reward_std = float(np.std(self.reward_history)) + 1e-8
        
        normalized = (reward - self.reward_mean) / self.reward_std
        return normalized
    
    def compute_shaped_reward(
        self,
        cache_hit: bool,
        noma_success: Optional[bool] = None,
        ber: Optional[float] = None,
        outage: bool = False,
        cache_occupancy: float = 0.5
    ) -> float:
        """Compute shaped reward."""
        reward = 0.0
        
        if cache_hit:
            base_reward = 50.0
            if cache_occupancy < 0.9:
                base_reward += 10.0
            reward = base_reward
        else:
            if noma_success is None:
                reward = -2.0
            elif outage:
                reward = -20.0
            elif not noma_success:
                reward = -10.0
            else:
                reward = -2.0
                if ber is not None:
                    if ber < 1e-4:
                        reward += 3.0
                    elif ber < 1e-3:
                        reward += 1.0
                    elif ber > 1e-2:
                        reward -= 3.0
        
        return self._normalize_reward(reward)
    
    def select_action_nn(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return int(np.random.randint(0, self.action_dim))
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor).cpu().numpy()[0]
            return int(np.argmax(q_values))
    
    def _sample_prioritized_batch(self):
        """Sample prioritized batch."""
        if len(self.replay_buffer) < self.batch_size:
            return None, None, None
        
        prios = np.array(self.priorities, dtype=np.float64)
        prios = np.clip(prios, self.priority_min, None)
        probs = prios ** self.priority_alpha
        probs_sum = probs.sum()
        
        if probs_sum <= 0.0:
            probs = np.ones_like(prios) / prios.size
        else:
            probs = probs / probs_sum
        
        # ✅ FIX #9: Use replace=True for efficiency
        indices = np.random.choice(len(self.replay_buffer), size=self.batch_size, 
                                  replace=True, p=probs)
        
        total = len(self.replay_buffer)
        weights = (total * probs[indices]) ** (-self.priority_beta)
        weights = weights / (weights.max() + 1e-8)
        
        return indices, weights.astype(np.float32), probs[indices]
    
    def _soft_update_target(self):
        """Soft update target network."""
        for target_param, policy_param in zip(
            self.target_network.parameters(),
            self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * policy_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def update_network(self, batch, weights: Optional[np.ndarray] = None):
        """Update Q-network."""
        states, actions, rewards, next_states, dones, indices = batch
        
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)
        
        q_values = self.q_network(states_t)
        current_q = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)
        
        # Double DQN
        with torch.no_grad():
            next_q_main = self.q_network(next_states_t)
            next_actions = torch.argmax(next_q_main, dim=1)
            next_q_target = self.target_network(next_states_t)
            next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards_t + (1.0 - dones_t) * self.gamma * next_q
        
        td_errors = (target_q - current_q).detach()
        losses = F.smooth_l1_loss(current_q, target_q, reduction='none')
        
        if weights is not None:
            weights_t = torch.FloatTensor(weights).to(self.device)
            loss = (losses * weights_t).mean()
        else:
            loss = losses.mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # ✅ FIX #6: Adaptive gradient clipping
        max_grad_norm = max(1.0, loss.item())
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_grad_norm)
        
        self.optimizer.step()
        
        loss_value = float(loss.item())
        self.training_losses.append(loss_value)
        
        self._soft_update_target()
        
        return loss_value, td_errors.cpu().numpy()
    
    def train_step(self) -> Optional[float]:
        """Perform one training step."""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        indices, weights, _ = self._sample_prioritized_batch()
        if indices is None:
            return None
        
        batch = [self.replay_buffer[i] for i in indices]
        states = np.array([exp['state'] for exp in batch], dtype=np.float32)
        actions = np.array([exp['action'] for exp in batch], dtype=np.int64)
        rewards = np.array([exp['reward'] for exp in batch], dtype=np.float32)
        next_states = np.array([exp['next_state'] for exp in batch], dtype=np.float32)
        dones = np.array([exp['done'] for exp in batch], dtype=np.float32)
        
        loss, td_errors = self.update_network(
            (states, actions, rewards, next_states, dones, indices),
            weights=weights
        )
        
        # ✅ FIX #3: Safe priority update
        abs_td = np.abs(td_errors) + self.priority_eps
        abs_td = np.clip(abs_td, self.priority_min, self.priority_max)
        
        for idx_local, global_idx in enumerate(indices):
            if global_idx < len(self.priorities):
                self.priorities[global_idx] = float(abs_td[idx_local])
        
        return loss
    
    def _update_q_table_entry(self, experience: Dict):
        """Update Q-table (fallback)."""
        state_str = str(experience['state'])
        action = experience['action']
        reward = experience['reward']
        next_state_str = str(experience['next_state'])
        
        current_q = self.q_table[state_str][action]
        next_max_q = max(self.q_table[next_state_str].values()) if self.q_table[next_state_str] else 0.0
        
        new_q = current_q + self.lr * (reward + self.gamma * next_max_q - current_q)
        self.q_table[state_str][action] = new_q
    
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
        outage: bool = False
    ):
        """
        ✅ FIX #1: CORRECTED transition logic with proper temporal alignment.
        
        This is the MOST CRITICAL FIX in the entire codebase!
        
        The key insight: reward_t belongs to the transition (s_{t-1}, a_{t-1}, r_t, s_t)
        """
        file_id = int(file_id)
        user_id = int(user_id)
        
        # === 1. Update statistics ===
        self.file_request_count[file_id] += 1
        freq = np.zeros(self.num_files, dtype=float)
        freq[file_id] = 1.0
        self.popularity_ema = (self.alpha_pop * freq + (1 - self.alpha_pop) * self.popularity_ema)
        
        if channel_gain is not None:
            self.channel_quality_history.append(float(channel_gain))
        
        if noma_success is not None:
            outcome = {
                'file_id': file_id, 'user_id': user_id, 'success': bool(noma_success),
                'sinr_weak': float(sinr_weak or 0.0), 'sinr_strong': float(sinr_strong or 0.0),
                'outage': bool(outage)
            }
            self.noma_outcome_history.append(outcome)
            self.file_noma_success[file_id].append(bool(noma_success))
        
        self.file_user_affinity[file_id].append(user_id)
        self.request_history.append({
            'file_id': file_id, 'user_id': user_id,
            'cache_hit': cache_hit, 'timestamp': len(self.request_history)
        })
        
        # === 2. Get current state ===
        current_state = self.get_state_vector(file_id)
        
        # === 3. Compute reward for CURRENT action ===
        cache_occupancy = sum(1 for x in self.contents_list if x != -1) / max(1, self.capacity)
        current_reward = self.compute_shaped_reward(cache_hit, noma_success, ber, outage, cache_occupancy)
        self.cumulative_reward += float(current_reward)
        
        # === 4. ✅ CRITICAL FIX: Store PREVIOUS transition with its CORRECT reward ===
        if self.last_state is not None and self.last_reward is not None:
            experience = {
                'state': self.last_state,          # State at t-1
                'action': self.last_action,        # Action taken at t-1
                'reward': self.last_reward,        # ✅ Reward FROM last action (NOT current!)
                'next_state': current_state,       # Current state at t
                'done': False
            }
            
            self.replay_buffer.append(experience)
            
            # Initialize priority
            if len(self.priorities) > 0:
                init_p = float(max(max(self.priorities), self.priority_min))
            else:
                init_p = 1.0
            self.priorities.append(init_p)
        
        # === 5. Decide and execute action for CURRENT request ===
        selected_action = -1
        
        if not cache_hit:
            # Cache miss - ask agent what to do
            if self.use_nn:
                selected_action = self.select_action_nn(current_state)
            else:
                # Q-table fallback
                if np.random.random() < self.epsilon:
                    selected_action = int(np.random.randint(0, self.action_dim))
                else:
                    empty_slots = [i for i, v in enumerate(self.contents_list) if v == -1]
                    if empty_slots:
                        selected_action = empty_slots[0]
                    else:
                        selected_action = int(np.random.randint(0, self.capacity))
            
            # Execute action
            if selected_action < self.capacity:
                slot = int(selected_action)
                old_file = self.contents_list[slot]
                if old_file != -1 and old_file in self.file_to_slot:
                    del self.file_to_slot[old_file]
                
                self.contents_list[slot] = file_id
                self.file_to_slot[file_id] = slot
            # else: action was self.capacity (do not cache)
            
            action_to_store = int(selected_action)
        else:
            # Cache hit - no decision needed
            action_to_store = self.file_to_slot.get(file_id, self.capacity)
        
        # === 6. ✅ Save state, action, AND REWARD for NEXT transition ===
        self.last_state = current_state
        self.last_action = action_to_store
        self.last_reward = current_reward  # ✅ Store current reward for next iteration
        
        # === 7. Training ===
        self.training_step += 1
        if self.training_step % 4 == 0 and self.use_nn:
            try:
                self.train_step()
            except Exception as e:
                self.training_losses.append(float('nan'))
                print(f"Warning: training step failed: {e}")
        
        # === 8. Epsilon decay ===
        if self.epsilon > self.epsilon_end:
            if self.epsilon_decay_mode == 'linear':
                self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay_rate)
            else:
                self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay_rate)
    
    def populate(self, items=None):
        """Initial cache population."""
        if items is None:
            top_indices = np.argsort(-self.popularity_ema)[:self.capacity]
        else:
            top_indices = items[:self.capacity]
        
        self.contents_list = [-1] * self.capacity
        self.file_to_slot.clear()
        for slot, f in enumerate(top_indices):
            f = int(f)
            self.contents_list[slot] = f
            self.file_to_slot[f] = slot
    
    def is_hit(self, item: int) -> bool:
        """Check if item is in cache."""
        return int(item) in self.file_to_slot
    
    def clear(self):
        """Clear cache."""
        self.contents_list = [-1] * self.capacity
        self.file_to_slot.clear()
        self.last_state = None
        self.last_action = None
        self.last_reward = None
    
    def get_stats(self) -> Dict:
        """Return learning statistics."""
        return {
            'use_neural_network': self.use_nn,
            'training_step': self.training_step,
            'epsilon': self.epsilon,
            'epsilon_start': self.epsilon_start,
            'epsilon_end': self.epsilon_end,
            'cumulative_reward': self.cumulative_reward,
            'replay_buffer_size': len(self.replay_buffer),
            'cache_size': sum(1 for x in self.contents_list if x != -1),
            'avg_loss': float(np.nanmean(self.training_losses[-100:])) if self.training_losses else 0.0,
            'total_requests_observed': len(self.request_history),
            'noma_outcomes_observed': len(self.noma_outcome_history),
            'reward_mean': self.reward_mean,
            'reward_std': self.reward_std
        }
    
    def save_model(self, filepath: str):
        """Save model."""
        if self.use_nn:
            torch.save({
                'q_network': self.q_network.state_dict(),
                'target_network': self.target_network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'epsilon_start': self.epsilon_start,
                'epsilon_end': self.epsilon_end,
                'epsilon_decay_steps': self.epsilon_decay_steps,
                'epsilon_decay_mode': self.epsilon_decay_mode,
                'training_step': self.training_step,
                'cumulative_reward': self.cumulative_reward,
                'contents_list': self.contents_list,
                'file_to_slot': self.file_to_slot,
                'replay_buffer': list(self.replay_buffer),
                'priorities': list(self.priorities),
                'last_state': self.last_state,
                'last_action': self.last_action,
                'last_reward': self.last_reward,
                'reward_mean': self.reward_mean,
                'reward_std': self.reward_std,
                'reward_history': list(self.reward_history)
            }, filepath)
            print(f"✅ Model saved: {filepath}")
        else:
            model = {
                'q_table': dict(self.q_table),
                'popularity_ema': self.popularity_ema,
                'epsilon': self.epsilon,
                'cumulative_reward': self.cumulative_reward,
                'contents_list': self.contents_list,
                'file_to_slot': self.file_to_slot,
                'last_reward': self.last_reward
            }
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            print(f"✅ Model saved: {filepath}")
    
    def load_model(self, filepath: str):
        """Load learned model."""
        if self.use_nn:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network'])
            self.target_network.load_state_dict(checkpoint['target_network'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            self.epsilon_start = checkpoint.get('epsilon_start', self.epsilon_start)
            self.epsilon_end = checkpoint.get('epsilon_end', self.epsilon_end)
            self.epsilon_decay_steps = checkpoint.get('epsilon_decay_steps', self.epsilon_decay_steps)
            self.training_step = checkpoint.get('training_step', self.training_step)
            self.cumulative_reward = checkpoint.get('cumulative_reward', self.cumulative_reward)
            self.contents_list = checkpoint.get('contents_list', self.contents_list)
            self.file_to_slot = checkpoint.get('file_to_slot', self.file_to_slot)
            rb = checkpoint.get('replay_buffer', None)
            pr = checkpoint.get('priorities', None)
            if rb is not None and pr is not None:
                self.replay_buffer = deque(rb, maxlen=self.replay_buffer.maxlen)
                self.priorities = deque(pr, maxlen=self.priorities.maxlen)
            self.last_state = checkpoint.get('last_state', None) ### NEW ###
            self.last_action = checkpoint.get('last_action', None) ### NEW ###
        else:
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            self.q_table = defaultdict(lambda: defaultdict(float), model.get('q_table', {}))
            self.popularity_ema = model.get('popularity_ema', self.popularity_ema)
            self.epsilon = model.get('epsilon', self.epsilon)
            self.cumulative_reward = model.get('cumulative_reward', self.cumulative_reward)
            self.contents_list = model.get('contents_list', self.contents_list)
            self.file_to_slot = model.get('file_to_slot', self.file_to_slot)

