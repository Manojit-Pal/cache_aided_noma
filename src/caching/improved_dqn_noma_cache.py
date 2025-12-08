# src/caching/improved_dqn_noma_cache.py

import random
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
    """Dueling Q-Network with CORRECT mean-based aggregation."""

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
            nn.Linear(input_dim, max(1, input_dim // 2)),
            nn.ReLU(),
            nn.Linear(max(1, input_dim // 2), 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(input_dim, max(1, input_dim // 2)),
            nn.ReLU(),
            nn.Linear(max(1, input_dim // 2), self.action_dim)
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

        # FIXED: Use MEAN for standard dueling aggregation
        advantages_mean = advantages.mean(dim=1, keepdim=True)
        q_values = values + (advantages - advantages_mean)

        return q_values


class ImprovedDQNNomaCache(CacheBase):
    """
    FIXED Deep Q-Network based NOMA cache with proper credit assignment.
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

        # Input validation
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

        # Set seeds for reproducibility
        self._set_seeds(self.seed)

        # Epsilon schedule
        self.epsilon_start = float(epsilon_start)
        self.epsilon = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay_steps = int(epsilon_decay_steps)

        if self.epsilon_decay_steps > 0:
            self.epsilon_decay_rate = (self.epsilon_start - self.epsilon_end) / self.epsilon_decay_steps
            self.epsilon_decay_mode = 'linear'
        else:
            self.epsilon_decay_rate = 0.995
            self.epsilon_decay_mode = 'exponential'

        # NEW: Evaluation mode flag
        self.eval_mode = False
        self._stored_epsilon = self.epsilon

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

        # Transition tracking - SIMPLIFIED
        self.last_state = None
        self.last_action = None

        # Reward normalization stats - IMPROVED STABILITY
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_history = deque(maxlen=2000)  # Larger window

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

        # NEW: Training warmup
        self.min_buffer_size = max(1000, self.batch_size * 10)

        # Performance tracking
        self.cumulative_reward = 0.0
        self.episode_rewards = []
        self.training_losses = []

    def _set_seeds(self, seed: int):
        """Set all random seeds for reproducibility."""
        np.random.seed(seed)
        random.seed(seed)

        if TORCH_AVAILABLE:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def set_eval_mode(self, eval_mode: bool = True):
        """
        NEW: Set evaluation mode (no exploration).
        Call this before testing to get true policy performance.
        """
        self.eval_mode = eval_mode
        if eval_mode:
            self._stored_epsilon = self.epsilon
            self.epsilon = 0.0  # No exploration during evaluation
            print("🔍 Evaluation mode: ON (epsilon=0)")
        else:
            self.epsilon = self._stored_epsilon
            print("🎓 Training mode: ON (epsilon={:.4f})".format(self.epsilon))

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
        """Normalize rewards using running statistics - IMPROVED STABILITY."""
        self.reward_history.append(reward)

        # Wait for sufficient samples before normalizing
        if len(self.reward_history) >= 100:
            self.reward_mean = float(np.mean(self.reward_history))
            self.reward_std = max(1.0, float(np.std(self.reward_history)))  # Prevent division by tiny values

        normalized = (reward - self.reward_mean) / self.reward_std
        normalized = float(np.clip(normalized, -10.0, 10.0))
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
        """Epsilon-greedy action selection - respects eval_mode."""
        # In eval mode, epsilon is 0, so this always takes greedy action
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

        replace_flag = len(self.replay_buffer) < self.batch_size
        indices = np.random.choice(len(self.replay_buffer), size=self.batch_size,
                                   replace=replace_flag, p=probs)

        total = len(self.replay_buffer)
        eps = 1e-8
        sample_probs = np.clip(probs[indices], eps, None)
        weights = (total * sample_probs) ** (-self.priority_beta)
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

        # Gradient clipping
        fixed_max = 10.0
        adaptive = 0.0
        try:
            if torch.isfinite(loss):
                adaptive = min(5.0, max(0.0, float(loss.item())))
        except Exception:
            adaptive = 0.0
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), fixed_max + adaptive)

        # Check for NaN/Inf
        nan_grad = False
        for p in self.q_network.parameters():
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all():
                nan_grad = True
                break
        
        if not nan_grad and torch.isfinite(loss).item():
            self.optimizer.step()
        else:
            print("⚠️ NaN/Inf detected - skipping optimizer step")
            self.optimizer.zero_grad()

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

        # IMPROVED: Priority update using percentile-based clipping
        abs_td = np.abs(td_errors) + self.priority_eps
        abs_td = np.clip(abs_td, self.priority_min, self.priority_max)

        for idx_local, global_idx in enumerate(indices):
            if 0 <= global_idx < len(self.priorities):
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
        outage: bool = False,
        episode_done: bool = False
    ):
        """
        Correct reward-action timing for proper credit assignment.

        NEW FLOW (corrected):
        1. Observe current state (S_t)
        2. Compute reward observed at time t (R_t) using PRE-ACTION environment
        3. Store (S_{t-1}, A_{t-1}, R_t, S_t)
        4. Select and execute action A_t for the current request
        5. Save current state/action for next iteration
        """
        file_id = int(file_id)
        user_id = int(user_id)

        if not (0 <= file_id < self.num_files):
            raise ValueError(f"file_id {file_id} out of range [0, {self.num_files - 1}]")
        if not (isinstance(user_id, int) and user_id >= 0):
            raise ValueError(f"user_id must be a non-negative int (got {user_id})")

        # Update statistics
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

        # STEP 1: Observe current state S_t (before taking any new action)
        current_state = self.get_state_vector(file_id)

        # IMPORTANT: compute reward observed at time t using PRE-ACTION environment
        cache_occupancy_pre = sum(1 for x in self.contents_list if x != -1) / max(1, self.capacity)
        current_reward = self.compute_shaped_reward(cache_hit, noma_success, ber, outage, cache_occupancy_pre)
        self.cumulative_reward += float(current_reward)

        # STEP 2: Store transition for PREVIOUS action: (S_{t-1}, A_{t-1}, R_t, S_t)
        if self.last_state is not None and self.last_action is not None:
            experience = {
                'state': self.last_state,
                'action': self.last_action,
                'reward': current_reward,   # reward observed now belongs to last action
                'next_state': current_state,
                'done': bool(episode_done)
            }

            self.replay_buffer.append(experience)

            # Priority initialization using percentile
            if len(self.priorities) > 0:
                try:
                    if self.priorities:
                        init_p = float(max(self.priorities))
                    else:
                        init_p = 1.0
                except Exception:
                    init_p = 1.0
            else:
                init_p = 1.0
            self.priorities.append(init_p)

            # Update Q-table if not using NN
            if not self.use_nn:
                self._update_q_table_entry(experience)

            if episode_done:
                self.last_state = None
                self.last_action = None

        # STEP 3: SELECT and EXECUTE action for CURRENT request (now)
        selected_action = None

        if cache_hit:
            # Cache hit: no caching decision needed; mark as maintain/no-op
            selected_action = self.file_to_slot.get(file_id, self.capacity)
        else:
            # Cache miss: agent decides whether/where to cache
            if self.use_nn:
                selected_action = self.select_action_nn(current_state)
            else:
                # Q-table fallback
                if np.random.random() < self.epsilon:
                    selected_action = int(np.random.randint(0, self.action_dim))
                else:
                    state_str = str(current_state.tolist() if isinstance(current_state, np.ndarray) else current_state)
                    action_values = self.q_table.get(state_str, None)
                    if action_values:
                        best_action = max(action_values.items(), key=lambda kv: kv[1])[0]
                        selected_action = int(best_action)
                    else:
                        empty_slots = [i for i, v in enumerate(self.contents_list) if v == -1]
                        if empty_slots:
                            selected_action = empty_slots[0]
                        else:
                            selected_action = int(np.random.randint(0, self.capacity))


            # Execute caching action (apply selection)
            if selected_action is not None and selected_action < self.capacity:
                slot = int(selected_action)

                old_file = self.contents_list[slot]
                if old_file != -1 and old_file in self.file_to_slot:
                    try:
                        del self.file_to_slot[old_file]
                    except KeyError:
                        pass

                prev_slot = self.file_to_slot.get(file_id)
                if prev_slot is not None and prev_slot != slot:
                    self.contents_list[prev_slot] = -1
                    try:
                        del self.file_to_slot[file_id]
                    except KeyError:
                            pass

                self.contents_list[slot] = file_id
                self.file_to_slot[file_id] = slot

        # STEP 4: Save current state/action for next iteration
        self.last_state = current_state
        self.last_action = selected_action
        self.last_reward = current_reward  # available for external inspection/debugging

        # Training step with warmup (unchanged)
        self.training_step += 1
        if (self.training_step % 4 == 0 and
            self.use_nn and
            len(self.replay_buffer) >= self.min_buffer_size and
            not self.eval_mode):  # Don't train in eval mode
            try:
                self.train_step()
            except Exception as e:
                self.training_losses.append(float('nan'))
                print(f"⚠️ Training step failed: {e}")

        # Epsilon decay (only in training mode)
        if not self.eval_mode and self.epsilon > self.epsilon_end:
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

    def reset_training_state(self):
        """
        Reset training state for fresh training run.
        Keeps learned networks but resets epsilon, replay buffer, etc.
        """
        self.epsilon = self.epsilon_start
        self.training_step = 0
        self.cumulative_reward = 0.0
        self.replay_buffer.clear()
        self.priorities.clear()
        self.last_state = None
        self.last_action = None
        self.reward_history.clear()
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.training_losses.clear()
        self.episode_rewards.clear()
        print(f"🔄 Training state reset: epsilon={self.epsilon:.4f}, buffer cleared")

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
            'reward_std': self.reward_std,
            'eval_mode': self.eval_mode
        }

    def save_model(self, filepath: str):
        """Save model."""
        if self.use_nn:
            # Convert numpy arrays to lists for safer serialization
            replay_buffer_serializable = []
            for exp in self.replay_buffer:
                exp_copy = exp.copy()
                if isinstance(exp.get('state'), np.ndarray):
                    exp_copy['state'] = exp['state'].tolist()
                if isinstance(exp.get('next_state'), np.ndarray):
                    exp_copy['next_state'] = exp['next_state'].tolist()
                replay_buffer_serializable.append(exp_copy)
            
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
                'replay_buffer': replay_buffer_serializable,
                'priorities': list(self.priorities),
                'last_state': self.last_state.tolist() if self.last_state is not None else None,
                'last_action': self.last_action,
                'reward_mean': float(self.reward_mean),
                'reward_std': float(self.reward_std),
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
                'file_to_slot': self.file_to_slot
            }
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            print(f"✅ Model saved: {filepath}")

    def load_model(self, filepath: str):
        """Load learned model."""
        if self.use_nn:
            # Fix for PyTorch 2.6+ weights_only warning
            # Safe to use weights_only=False for our own checkpoints
            try:
                checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
            except TypeError:
                # Fallback for older PyTorch versions
                checkpoint = torch.load(filepath, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network'])
            self.target_network.load_state_dict(checkpoint['target_network'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            
            # Move optimizer state to device
            try:
                for state in self.optimizer.state.values():
                    for k, v in list(state.items()):
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)
            except Exception as e:
                print(f"⚠️ Warning: failed to move optimizer state: {e}")

            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            self.epsilon_start = checkpoint.get('epsilon_start', self.epsilon_start)
            self.epsilon_end = checkpoint.get('epsilon_end', self.epsilon_end)
            self.epsilon_decay_steps = checkpoint.get('epsilon_decay_steps', self.epsilon_decay_steps)
            self.epsilon_decay_mode = checkpoint.get('epsilon_decay_mode', 'linear')
            self.training_step = checkpoint.get('training_step', self.training_step)
            self.cumulative_reward = checkpoint.get('cumulative_reward', self.cumulative_reward)
            self.contents_list = checkpoint.get('contents_list', self.contents_list)
            self.file_to_slot = checkpoint.get('file_to_slot', self.file_to_slot)

            rb = checkpoint.get('replay_buffer', None)
            pr = checkpoint.get('priorities', None)
            if rb is not None and pr is not None:
                # Convert numpy arrays in experiences back from lists
                converted_rb = []
                for exp in rb:
                    converted_exp = exp.copy()
                    if isinstance(exp.get('state'), list):
                        converted_exp['state'] = np.array(exp['state'], dtype=np.float32)
                    if isinstance(exp.get('next_state'), list):
                        converted_exp['next_state'] = np.array(exp['next_state'], dtype=np.float32)
                    converted_rb.append(converted_exp)
                
                self.replay_buffer = deque(converted_rb, maxlen=self.replay_buffer.maxlen)
                self.priorities = deque(pr, maxlen=self.priorities.maxlen)

            self.last_state = checkpoint.get('last_state', None)
            if self.last_state is not None and isinstance(self.last_state, list):
                self.last_state = np.array(self.last_state, dtype=np.float32)
            self.last_action = checkpoint.get('last_action', None)
            self.reward_mean = checkpoint.get('reward_mean', self.reward_mean)
            self.reward_std = checkpoint.get('reward_std', self.reward_std)
            
            reward_hist = checkpoint.get('reward_history', None)
            if reward_hist is not None:
                self.reward_history = deque(reward_hist, maxlen=self.reward_history.maxlen)

            print(f"📌 Model loaded: {filepath}")
        else:
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            self.q_table = defaultdict(lambda: defaultdict(float), model.get('q_table', {}))
            self.popularity_ema = model.get('popularity_ema', self.popularity_ema)
            self.epsilon = model.get('epsilon', self.epsilon)
            self.cumulative_reward = model.get('cumulative_reward', self.cumulative_reward)
            self.contents_list = model.get('contents_list', self.contents_list)
            self.file_to_slot = model.get('file_to_slot', self.file_to_slot)
            print(f"📄 Q-table model loaded: {filepath}")
