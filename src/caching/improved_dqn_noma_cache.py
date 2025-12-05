# src/caching/improved_dqn_noma_cache.py
"""
TRUE Deep Q-Network implementation for NOMA caching using neural network.
Reworked to fix stability and semantics:
 - meaningful action space: choose cache slot to evict OR choose not to cache
 - Double DQN targets
 - Huber loss (smooth L1)
 - stable prioritized replay (clipped priorities, importance-sampling weights)
 - reward scaling / normalization to avoid huge TDs
 - proper cache slot bookkeeping (deterministic mapping)

MODIFICATIONS:
 - State Vector: Now includes features for the requested file and cache contents.
 - Dueling DQN: Network architecture changed to DuelingDQN.
 - Soft Target Updates: Replaced hard updates with Polyak averaging (tau).
 - observe_request: Updated to store (s, a, r, s') transitions correctly.
 - Reward Function: Re-balanced to strongly prioritize cache hits.
 - Defaults: Increased default replay_buffer_size and epsilon_decay_steps.
"""

import numpy as np
from collections import deque, defaultdict
import pickle
from typing import Dict, List, Tuple, Optional

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Using tabular Q-learning fallback.")

from .cache_base import CacheBase


### MODIFIED: Replaced DQNNetwork with DuelingDQNNetwork ###
class DQNNetwork(nn.Module):
    """
    Dueling Q-Network implementation.
    Splits the network into a Value stream and an Advantage stream.
    Q(s, a) = V(s) + (A(s, a) - mean(A(s, a')))
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims=[128, 64]):
        super(DQNNetwork, self).__init__()
        self.action_dim = action_dim
        
        # Shared feature layer
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
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
    
    def forward(self, state):
        features = self.feature_layer(state)
        
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine V(s) and A(s, a)
        # Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum(A(s,a')))
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values


class ImprovedDQNNomaCache(CacheBase):
    """
    Reworked DQN-based NOMA cache:
     - action space: cache_slot index [0..capacity-1] to evict and replace with requested file,
       plus special action = capacity meaning "do NOT cache".
     - Dueling Double DQN (policy network selects next action; target network evaluates).
     - Prioritized replay with clipped priorities + importance sampling.
     - Huber (smooth L1) loss.
     - Reward scaling to keep magnitudes small.
     - State vector includes request and cache content features.
     - Soft target updates (Polyak averaging).
    """
    
    def __init__(self, capacity: int, num_files: int, num_users: int,
                 learning_rate: float = 1e-4,
                 gamma: float = 0.95,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 ### MODIFIED: Increased default for longer exploration ###
                 epsilon_decay_steps: int = 25000,
                 use_neural_network: bool = True,
                 ### MODIFIED: Increased default for larger memory ###
                 replay_buffer_size: int = 50000,
                 batch_size: int = 64,
                 priority_alpha: float = 0.6,
                 priority_beta: float = 0.4,
                 tau: float = 1e-3): ### MODIFIED: Replaced target_update_freq with tau ###
        super().__init__(capacity)
        
        self.num_files = int(num_files)
        self.num_users = int(num_users)
        self.lr = float(learning_rate)
        self.gamma = float(gamma)

        # Epsilon schedule
        self.epsilon_start = float(epsilon_start)
        self.epsilon = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay_steps = int(epsilon_decay_steps)
        if self.epsilon_decay_steps > 0:
            self.epsilon_decay_rate = (self.epsilon_start - self.epsilon_end) / self.epsilon_decay_steps
        else:
            self.epsilon_decay_rate = 0.0
        
        # Cache bookkeeping: deterministic indexed slots (0..capacity-1)
        # Use -1 to denote empty slot
        self.capacity = int(capacity)
        self.contents_list = [-1] * self.capacity
        self.file_to_slot = {}  # file_id -> slot index
        
        # State tracking
        self.popularity_ema = np.ones(self.num_files) / max(1, self.num_files)
        self.alpha_pop = 0.1
        self.channel_quality_history = deque(maxlen=100)
        self.noma_outcome_history = deque(maxlen=200)
        self.request_history = deque(maxlen=500)
        self.file_request_count = np.zeros(self.num_files, dtype=np.int32)
        self.file_noma_success = defaultdict(lambda: deque(maxlen=50)) ### MODIFIED: Use deque for efficiency
        self.file_user_affinity = defaultdict(lambda: deque(maxlen=50)) ### MODIFIED: Use deque for efficiency
        
        # RL components
        self.use_nn = use_neural_network and TORCH_AVAILABLE
        self.training_step = 0
        
        # Action dimension: capacity slots + 1 (do-not-cache)
        self.action_dim = self.capacity + 1
        
        ### NEW: Track last state/action for (s, a, r, s') transitions ###
        self.last_state = None
        self.last_action = None
        
        if self.use_nn:
            # Neural network setup
            self.state_dim = self._get_state_dim()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Main Q-network & target
            self.q_network = DQNNetwork(self.state_dim, self.action_dim).to(self.device)
            self.target_network = DQNNetwork(self.state_dim, self.action_dim).to(self.device)
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.target_network.eval()
            
            # Optimizer
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
            
            ### MODIFIED: Store tau for soft updates ###
            self.tau = float(tau)
            
        else:
            # Fallback to Q-table (keeps previous signature; rarely used)
            self.q_table = defaultdict(lambda: defaultdict(float))
        
        # Prioritized replay (simple but stable implementation)
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
    
    ### MODIFIED: State dim now includes cache contents + request features ###
    def _get_state_dim(self):
        """Calculate state dimension for neural network."""
        # 1. Global features: cache occupancy, channel (mean, std), NOMA (rate, weak, strong), req rate
        global_features_dim = 1 + 2 + 3 + 1
        
        # 2. Requested file features: popularity, NOMA success rate
        request_features_dim = 2
        
        # 3. Cache content features: popularity of file in each slot
        cache_content_features_dim = self.capacity
        
        return global_features_dim + request_features_dim + cache_content_features_dim
    
    ### MODIFIED: Renamed and now takes file_id ###
    def get_state_vector(self, file_id: int) -> np.ndarray:
        """
        Get state as continuous vector (for neural network).
        State = [Global Features, Requested File Features, Cache Content Features]
        """
        features = []
        
        # === 1. Global Features ===
        
        # 1.1 Cache occupancy
        cache_occupancy = sum(1 for x in self.contents_list if x != -1) / max(1, self.capacity)
        features.append(cache_occupancy)
        
        # 1.2 Channel quality
        if len(self.channel_quality_history) > 0:
            # Use the deque directly, it's already the "recent" history
            features.extend([float(np.mean(self.channel_quality_history)), 
                             float(np.std(self.channel_quality_history))])
        else:
            features.extend([0.5, 0.1])
        
        # 1.3 NOMA performance
        if len(self.noma_outcome_history) > 0:
            recent = list(self.noma_outcome_history) # Already capped by maxlen
            success_rate = float(np.mean([1.0 if x['success'] else 0.0 for x in recent]))
            avg_sinr_weak = float(np.mean([x['sinr_weak'] for x in recent]))
            avg_sinr_strong = float(np.mean([x['sinr_strong'] for x in recent]))
            features.extend([success_rate, avg_sinr_weak, avg_sinr_strong])
        else:
            features.extend([0.5, 0.0, 0.0])
        
        # 1.4 Request rate
        if len(self.request_history) > 10:
            request_rate = float(len(self.request_history) / self.request_history.maxlen)
        else:
            request_rate = 0.0
        features.append(request_rate)
        
        # === 2. Requested File Features ===
        
        # 2.1 Popularity
        features.append(self.popularity_ema[file_id])
        
        # 2.2 NOMA success rate for this file
        if file_id in self.file_noma_success and len(self.file_noma_success[file_id]) > 0:
            features.append(float(np.mean(self.file_noma_success[file_id])))
        else:
            features.append(0.5) # Default assumption
            
        # === 3. Cache Content Features ===
        
        # 3.1 Popularity of each file in cache slots
        slot_popularities = [
            self.popularity_ema[file_in_slot] if file_in_slot != -1 else 0.0
            for file_in_slot in self.contents_list
        ]
        features.extend(slot_popularities)
        
        return np.array(features, dtype=np.float32)
    
    def _scale_reward(self, reward: float) -> float:
        """
        Scale rewards to small magnitudes to avoid exploding TD errors.
        This is simple linear scaling; you may prefer to normalize online.
        """
        # scale factor chosen empirically
        return float(reward) / 10.0
    
    ### MODIFIED: Re-balanced reward function ###
    def compute_shaped_reward(self, cache_hit: bool, noma_success: Optional[bool] = None,
                              ber: Optional[float] = None, outage: bool = False,
                              cache_occupancy: float = 0.5) -> float:
        """
        Reward function with shaping for faster learning.
        Returns the *scaled* reward (small magnitude).
        """
        reward = 0.0
        
        if cache_hit:
            ### MODIFICATION: Make hits much more valuable ###
            base_reward = 50.0  # Was 10.0
            if cache_occupancy < 0.9:
                base_reward += 10.0 # Was 2.0
            reward = base_reward
        else:
            # Cache miss - evaluate transmission
            if noma_success is None:
                reward = -2.0 # Was -1.0 (penalize misses more)
            elif outage:
                reward = -20.0 # Was -10.0 (penalize outage more)
            elif not noma_success:
                reward = -10.0 # Was -5.0 (penalize NOMA failure more)
            else:
                # NOMA was successful, but it's still a miss
                reward = -2.0 # Was -1.0
                if ber is not None:
                    # Keep these shaping rewards small, they just fine-tune
                    if ber < 1e-4:
                        reward += 3.0
                    elif ber < 1e-3:
                        reward += 1.0
                    elif ber > 1e-2:
                        reward -= 3.0
        
        return self._scale_reward(reward)
    
    def select_action_nn(self, state: np.ndarray) -> int:
        """
        Epsilon-greedy action selection over action_dim:
         - actions 0..capacity-1 => eviction slot index
         - action == capacity => do not cache
        """
        if np.random.random() < self.epsilon:
            return int(np.random.randint(0, self.action_dim))
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor).cpu().numpy()[0]
            # choose best action
            return int(np.argmax(q_values))
    
    def _sample_prioritized_batch(self):
        """Sample indices with priority-based probabilities and return IS weights."""
        if len(self.replay_buffer) < self.batch_size:
            return None, None, None
        
        prios = np.array(self.priorities, dtype=np.float64)
        # avoid zero and negative
        prios = np.clip(prios, self.priority_min, None)
        probs = prios ** self.priority_alpha
        probs_sum = probs.sum()
        if probs_sum <= 0.0:
            probs = np.ones_like(probs) / probs.size
        else:
            probs = probs / probs_sum
        
        indices = np.random.choice(len(self.replay_buffer), size=self.batch_size, replace=False, p=probs)
        
        # importance-sampling weights
        total = len(self.replay_buffer)
        weights = (total * probs[indices]) ** (-self.priority_beta)
        # normalize weights (divide by max to stabilize)
        weights = weights / (weights.max() + 1e-8)
        return indices, weights.astype(np.float32), probs[indices]
    
    ### NEW: Soft target update function ###
    def _soft_update_target(self):
        """Soft update model parameters. θ_target = τ*θ_policy + (1 - τ)*θ_target"""
        for target_param, policy_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)

    def update_network(self, batch, weights: Optional[np.ndarray] = None):
        """Update neural network using batch of experiences. Uses Double DQN and Huber loss."""
        states, actions, rewards, next_states, dones, indices = batch
        
        # convert to tensors
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)
        
        # current Q
        q_values = self.q_network(states_t)
        current_q = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)
        
        # Double DQN target:
        with torch.no_grad():
            # actions chosen by main network
            next_q_main = self.q_network(next_states_t)
            next_actions = torch.argmax(next_q_main, dim=1)
            # evaluate with target network
            next_q_target = self.target_network(next_states_t)
            next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards_t + (1.0 - dones_t) * self.gamma * next_q
        
        # elementwise Huber loss
        td_errors = (target_q - current_q).detach()
        losses = F.smooth_l1_loss(current_q, target_q, reduction='none')
        
        if weights is not None:
            weights_t = torch.FloatTensor(weights).to(self.device)
            loss = (losses * weights_t).mean()
        else:
            loss = losses.mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # record loss and update priorities (return td_errors for outer update)
        loss_value = float(loss.item())
        self.training_losses.append(loss_value)
        
        ### MODIFIED: Call soft update instead of periodic hard update ###
        self._soft_update_target()
        
        return loss_value, td_errors.cpu().numpy()
    
    def train_step(self):
        """Perform one training step with experience replay and priority updates."""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        indices, weights, sampled_probs = self._sample_prioritized_batch()
        if indices is None:
            return None
        
        batch = [self.replay_buffer[i] for i in indices]
        states = np.array([exp['state'] for exp in batch], dtype=np.float32)
        actions = np.array([exp['action'] for exp in batch], dtype=np.int64)
        rewards = np.array([exp['reward'] for exp in batch], dtype=np.float32)
        next_states = np.array([exp['next_state'] for exp in batch], dtype=np.float32)
        dones = np.array([exp['done'] for exp in batch], dtype=np.float32)
        
        loss, td_errors = self.update_network(((states, actions, rewards, next_states, dones, indices)),
                                              weights=weights)
        
        # update priorities: abs(td) + eps, clipped
        abs_td = np.abs(td_errors) + self.priority_eps
        abs_td = np.clip(abs_td, self.priority_min, self.priority_max)
        for idx_local, global_idx in enumerate(indices):
            self.priorities[global_idx] = float(abs_td[idx_local])
        
        return loss
    
    def _update_q_table_entry(self, experience):
        """Update Q-table (fallback method)."""
        state_str = str(experience['state'])
        action = experience['action']
        reward = experience['reward']
        next_state_str = str(experience['next_state'])
        
        current_q = self.q_table[state_str][action]
        next_max_q = max(self.q_table[next_state_str].values()) if self.q_table[next_state_str] else 0.0
        
        new_q = current_q + self.lr * (reward + self.gamma * next_max_q - current_q)
        self.q_table[state_str][action] = new_q
    
    ### MODIFIED: Complete rework of observe_request for (s, a, r, s') logic ###
    def observe_request(self, user_id: int, file_id: int,
                        cache_hit: bool,
                        noma_success: Optional[bool] = None,
                        channel_gain: Optional[float] = None,
                        sinr_weak: Optional[float] = None,
                        sinr_strong: Optional[float] = None,
                        ber: Optional[float] = None,
                        outage: bool = False):
        """
        Main learning function.
        Stores the (s_t-1, a_t-1, r_t, s_t) transition.
        The state s_t is [global_features, request_features(file_id), cache_features].
        """
        file_id = int(file_id)
        user_id = int(user_id)
        
        # === 1. Update internal statistics (popularity, NOMA, etc.) ===
        
        # Update popularity
        self.file_request_count[file_id] += 1
        freq = np.zeros(self.num_files, dtype=float)
        freq[file_id] = 1.0
        self.popularity_ema = (self.alpha_pop * freq +
                               (1 - self.alpha_pop) * self.popularity_ema)
        
        # Track channel and NOMA
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
        
        # === 2. Get current state and compute reward ===
        
        # Get state vector for the *current* request
        current_state = self.get_state_vector(file_id)
        
        # Compute shaped reward (already scaled small)
        cache_occupancy = sum(1 for x in self.contents_list if x != -1) / max(1, self.capacity)
        # Use the NEW re-balanced reward function
        reward = self.compute_shaped_reward(cache_hit, noma_success, ber, outage, cache_occupancy)
        self.cumulative_reward += float(reward)
        
        # === 3. Store the (s_t-1, a_t-1, r_t, s_t) transition ===
        # This is the transition from the *previous* step, which is now complete
        if self.last_state is not None:
            experience = {
                'state': self.last_state,
                'action': self.last_action,
                'reward': float(reward),      # Reward is received *after* the action
                'next_state': current_state,
                'done': False
            }
            
            # Append to replay & priorities
            self.replay_buffer.append(experience)
            if len(self.priorities) > 0:
                init_p = float(max(max(self.priorities), self.priority_min))
            else:
                init_p = float(1.0)
            self.priorities.append(init_p)

        # === 4. Decide and execute action for the *current* request ===
        
        selected_action = -1 # Default (will be updated)
        
        if not cache_hit:
            # On a miss, ask the agent what to do
            # The state vector (current_state) already has all info
            if self.use_nn:
                selected_action = self.select_action_nn(current_state)
            else:
                # Fallback
                if np.random.random() < self.epsilon:
                    selected_action = int(np.random.randint(0, self.action_dim))
                else:
                    # heuristic: fill empty slot or evict random
                    empty_slots = [i for i, v in enumerate(self.contents_list) if v == -1]
                    if empty_slots:
                        selected_action = empty_slots[0]
                    else:
                        selected_action = int(np.random.randint(0, self.capacity))
            
            # Execute the action
            if selected_action < self.capacity:
                slot = int(selected_action)
                old_file = self.contents_list[slot]
                if old_file != -1 and old_file in self.file_to_slot:
                    del self.file_to_slot[old_file]
                
                self.contents_list[slot] = file_id
                self.file_to_slot[file_id] = slot
            else:
                # Action was self.capacity (do not cache)
                pass
            
            action_to_store = int(selected_action)
            
        else:
            # On a hit, no cache decision is made.
            # Store the "dummy" action: either the slot it was in or "do-not-cache"
            action_to_store = self.file_to_slot.get(file_id, self.capacity)
        
        # === 5. Save state and action for the *next* transition ===
        self.last_state = current_state
        self.last_action = action_to_store
        
        # === 6. Perform training step ===
        self.training_step += 1
        if self.training_step % 4 == 0 and self.use_nn:
            try:
                self.train_step()
            except Exception as e:
                self.training_losses.append(float('nan'))
                print("Warning: training step failed:", e)
        
        # === 7. Decay epsilon ===
        if self.epsilon > self.epsilon_end:
            self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay_rate)
    
    def populate(self, items=None):
        """Initial cache population (fill slots deterministically)."""
        if items is None:
            # use popularity TOP-K to fill initial cache
            top_indices = np.argsort(-self.popularity_ema)[:self.capacity]
        else:
            top_indices = items[:self.capacity]
        
        # fill slots 0..k-1
        self.contents_list = [-1] * self.capacity
        self.file_to_slot.clear()
        for slot, f in enumerate(top_indices):
            f = int(f)
            self.contents_list[slot] = f
            self.file_to_slot[f] = slot
    
    def is_hit(self, item: int) -> bool:
        return int(item) in self.file_to_slot
    
    def clear(self):
        self.contents_list = [-1] * self.capacity
        self.file_to_slot.clear()
        ### NEW: Clear last_state/action ###
        self.last_state = None
        self.last_action = None
    
    def get_stats(self) -> Dict:
        """Return comprehensive learning statistics."""
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
            'noma_outcomes_observed': len(self.noma_outcome_history)
        }
    
    def save_model(self, filepath: str):
        """Save learned model and schedule params."""
        if self.use_nn:
            torch.save({
                'q_network': self.q_network.state_dict(),
                'target_network': self.target_network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'epsilon_start': self.epsilon_start,
                'epsilon_end': self.epsilon_end,
                'epsilon_decay_steps': self.epsilon_decay_steps,
                'training_step': self.training_step,
                'cumulative_reward': self.cumulative_reward,
                'contents_list': self.contents_list,
                'file_to_slot': self.file_to_slot,
                'replay_buffer': list(self.replay_buffer),
                'priorities': list(self.priorities),
                'last_state': self.last_state, ### NEW ###
                'last_action': self.last_action ### NEW ###
            }, filepath)
        else:
            model = {
                'q_table': dict(self.q_table),
                'popularity_ema': self.popularity_ema,
                'epsilon': self.epsilon,
                'cumulative_reward': self.cumulative_reward,
                'contents_list': self.contents_list,
                'file_to_slot': self.file_to_slot
            }
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
    
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