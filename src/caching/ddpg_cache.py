"""
src/caching/ddpg_cache.py

DDPG (Deep Deterministic Policy Gradient) Cache Agent
=====================================================

Actor-Critic based caching policy for comparison against DQN.
Uses the SAME 14-dim state space, SAME reward function, and SAME
cache interface as DQNCache v2 for fair, apples-to-apples comparison.

Architecture (from Lillicrap et al., 2016):
  - Actor:  state -> continuous action in [0,1] (cache probability)
  - Critic: (state, action) -> Q-value
  - Ornstein-Uhlenbeck noise for exploration
  - Soft target updates (tau)

The continuous action is thresholded at 0.5:
  action >= 0.5 -> cache the file
  action <  0.5 -> skip

Reference: Lillicrap et al., "Continuous control with deep reinforcement
learning", ICLR 2016.

Author: Cache-Aided NOMA Team
Date: April 2026
"""

import numpy as np
import random
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .cache_base import CacheBase


# ============================================================================
# ACTOR NETWORK: state -> continuous action [0, 1]
# ============================================================================

class DDPGActor(nn.Module):
    """Actor network that maps state to a continuous action."""

    def __init__(self, state_dim: int, hidden_dims: List[int] = [64, 32]):
        super().__init__()
        layers = []
        prev_dim = state_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.ReLU())
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())  # Output in [0, 1]
        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # Final layer: small init for stable initial outputs near 0.5
        final = list(self.modules())[-2]  # Last Linear before Sigmoid
        if isinstance(final, nn.Linear):
            nn.init.uniform_(final.weight, -3e-3, 3e-3)
            nn.init.uniform_(final.bias, -3e-3, 3e-3)

    def forward(self, state: 'torch.Tensor') -> 'torch.Tensor':
        return self.network(state)


# ============================================================================
# CRITIC NETWORK: (state, action) -> Q-value
# ============================================================================

class DDPGCritic(nn.Module):
    """Critic network that evaluates (state, action) pairs."""

    def __init__(self, state_dim: int, hidden_dims: List[int] = [64, 32]):
        super().__init__()
        # State pathway
        self.state_layer = nn.Linear(state_dim, hidden_dims[0])
        # Action injected after first layer
        self.combined_layers = nn.ModuleList()
        prev_dim = hidden_dims[0] + 1  # +1 for action
        for hdim in hidden_dims[1:]:
            self.combined_layers.append(nn.Linear(prev_dim, hdim))
            prev_dim = hdim
        self.output_layer = nn.Linear(prev_dim, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.uniform_(self.output_layer.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.output_layer.bias, -3e-3, 3e-3)

    def forward(self, state: 'torch.Tensor', action: 'torch.Tensor') -> 'torch.Tensor':
        x = F.relu(self.state_layer(state))
        x = torch.cat([x, action], dim=1)
        for layer in self.combined_layers:
            x = F.relu(layer(x))
        return self.output_layer(x)


# ============================================================================
# ORNSTEIN-UHLENBECK NOISE
# ============================================================================

class OUNoise:
    """Ornstein-Uhlenbeck process for temporally correlated exploration."""

    def __init__(self, size: int = 1, mu: float = 0.0,
                 theta: float = 0.15, sigma: float = 0.2,
                 sigma_min: float = 0.01, sigma_decay: float = 0.9999):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.sigma_min = sigma_min
        self.sigma_decay = sigma_decay
        self.state = self.mu.copy()

    def reset(self):
        self.state = self.mu.copy()

    def sample(self) -> np.ndarray:
        dx = self.theta * (self.mu - self.state) + \
             self.sigma * np.random.randn(len(self.mu))
        self.state += dx
        self.sigma = max(self.sigma_min, self.sigma * self.sigma_decay)
        return self.state


# ============================================================================
# REPLAY BUFFER
# ============================================================================

class ReplayBuffer:
    """Simple replay buffer for DDPG."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition: Dict):
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Dict]:
        replace = len(self.buffer) < batch_size
        indices = np.random.choice(len(self.buffer), batch_size, replace=replace)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)


# ============================================================================
# DDPG CACHE AGENT
# ============================================================================

class DDPGCache(CacheBase):
    """
    DDPG-based caching policy.

    Uses the SAME state space (14 dims), reward function, and cache
    interface as DQNCache v2 for fair comparison.

    Key differences from DQN:
    - Actor-Critic architecture (vs. Q-network only)
    - Continuous action space (vs. discrete {0,1})
    - Ornstein-Uhlenbeck noise (vs. epsilon-greedy)
    - Deterministic policy gradient (vs. value-based gradient)
    """

    RECENT_WINDOW = 500

    def __init__(
        self, capacity: int, num_files: int = 2000, num_users: int = 200,
        learning_rate: float = 0.001, gamma: float = 0.99,
        hidden_dims: List[int] = [64, 32],
        batch_size: int = 64, replay_buffer_size: int = 50000,
        train_freq: int = 4, warm_up_steps: int = 2000,
        gradient_clip: float = 10.0, tau: float = 0.005,
        ou_sigma: float = 0.2, ou_theta: float = 0.15,
        ou_sigma_min: float = 0.02, ou_sigma_decay: float = 0.9999,
        enable_noma_awareness: bool = True, seed: int = 2025,
        **kwargs  # Accept extra kwargs silently
    ):
        super().__init__(capacity, enable_noma_awareness)
        self.num_files = num_files
        self.num_users = num_users
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.train_freq = train_freq
        self.gradient_clip = gradient_clip
        self.tau = tau
        self.warm_up_steps = warm_up_steps
        self.eval_mode = False
        self._set_seeds(seed)

        # Cache storage (same as DQN)
        self.cache_set = set()
        self.file_to_slot = {}
        self.cache_slots = [-1] * capacity
        self.timestep = 0

        # Popularity tracking (same as DQN)
        self.request_counts = np.zeros(num_files, dtype=np.float64)
        self.recent_requests = deque(maxlen=self.RECENT_WINDOW)
        self.popularity_rank = np.arange(num_files)

        # NOMA tracking (same as DQN)
        self.channel_history = deque(maxlen=500)
        self.noma_history = deque(maxlen=500)
        self.cic_count = 0
        self.sic_count = 0

        # State/action dims (same state as DQN)
        self.state_dim = 14
        self.action_dim = 1  # Continuous [0, 1]

        # Networks
        if TORCH_AVAILABLE:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # Actor (policy) networks
            self.actor = DDPGActor(self.state_dim, hidden_dims).to(self.device)
            self.actor_target = DDPGActor(self.state_dim, hidden_dims).to(self.device)
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.actor_target.eval()

            # Critic (value) networks
            self.critic = DDPGCritic(self.state_dim, hidden_dims).to(self.device)
            self.critic_target = DDPGCritic(self.state_dim, hidden_dims).to(self.device)
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.critic_target.eval()

            # Optimizers
            self.actor_optimizer = optim.Adam(
                self.actor.parameters(), lr=learning_rate, weight_decay=1e-5)
            self.critic_optimizer = optim.Adam(
                self.critic.parameters(), lr=learning_rate, weight_decay=1e-5)

            # Replay buffer
            self.replay_buffer = ReplayBuffer(replay_buffer_size)

            # Exploration noise
            self.noise = OUNoise(
                size=1, theta=ou_theta, sigma=ou_sigma,
                sigma_min=ou_sigma_min, sigma_decay=ou_sigma_decay)
        else:
            self.replay_buffer = None

        self.training_step = 0
        self.episode_rewards = []
        self.losses = []
        self.cumulative_reward = 0.0

        print('DDPGCache initialized (actor-critic, continuous action)')
        print(f'  State dim  : {self.state_dim}')
        print(f'  Action     : continuous [0,1] -> threshold 0.5')
        print(f'  Network    : {hidden_dims}')
        print(f'  Warm-up    : {self.warm_up_steps} steps')

    def _set_seeds(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        if TORCH_AVAILABLE:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    # =========================================================================
    # POPULARITY TRACKING (identical to DQN)
    # =========================================================================

    def _update_popularity(self, file_id: int):
        self.request_counts[file_id] += 1
        self.recent_requests.append(file_id)

    def _get_file_popularity_rank(self, file_id: int) -> float:
        rank = np.sum(self.request_counts > self.request_counts[file_id])
        return float(rank) / max(self.num_files - 1, 1)

    def _get_file_recent_freq(self, file_id: int) -> float:
        if len(self.recent_requests) == 0:
            return 0.0
        count = sum(1 for f in self.recent_requests if f == file_id)
        return float(count) / len(self.recent_requests)

    def _get_min_popularity_slot(self) -> Tuple[int, int]:
        worst_slot, worst_file, worst_count = -1, -1, float('inf')
        for slot_idx in range(self.capacity):
            fid = self.cache_slots[slot_idx]
            if fid == -1:
                continue
            if self.request_counts[fid] < worst_count:
                worst_count = self.request_counts[fid]
                worst_slot = slot_idx
                worst_file = fid
        return worst_slot, worst_file

    # =========================================================================
    # STATE VECTOR (identical to DQN — 14 dims)
    # =========================================================================

    def _get_state_vector(self, requested_file: int) -> np.ndarray:
        state = []

        # 1. Popularity rank
        state.append(self._get_file_popularity_rank(requested_file))
        # 2. Recent frequency
        state.append(self._get_file_recent_freq(requested_file))
        # 3. Is cached
        state.append(1.0 if requested_file in self.cache_set else 0.0)
        # 4. Cache occupancy
        state.append(len(self.cache_set) / self.capacity)

        # 5-7. Cache popularity stats
        if self.cache_set:
            cached_counts = np.array([self.request_counts[f] for f in self.cache_set if f >= 0])
            total_counts = max(self.request_counts.sum(), 1.0)
            if len(cached_counts) > 0:
                cached_pops = cached_counts / total_counts
                state.extend([float(cached_pops.mean()), float(cached_pops.min()), float(cached_pops.max())])
            else:
                state.extend([0.0, 0.0, 0.0])
        else:
            state.extend([0.0, 0.0, 0.0])

        # 8. Would improve
        if self.cache_set:
            _, worst_file = self._get_min_popularity_slot()
            would_improve = 1.0 if (worst_file >= 0 and self.request_counts[requested_file] > self.request_counts[worst_file]) else (1.0 if worst_file < 0 else 0.0)
        else:
            would_improve = 1.0
        state.append(would_improve)

        # 9. Improve ratio
        if self.cache_set and len(self.cache_set) >= self.capacity:
            _, worst_file = self._get_min_popularity_slot()
            if worst_file >= 0 and self.request_counts[worst_file] > 0:
                ratio = float(self.request_counts[requested_file] / self.request_counts[worst_file])
                state.append(min(ratio, 5.0) / 5.0)
            else:
                state.append(1.0)
        else:
            state.append(1.0)

        # 10. CIC rate
        if self.noma_history:
            nh = list(self.noma_history)[-100:]
            state.append(float(sum(1 for x in nh if x.get('cic', False)) / len(nh)))
        else:
            state.append(0.0)

        # 11-12. Channel stats
        if self.channel_history:
            ch = list(self.channel_history)[-100:]
            state.extend([float(np.mean(ch)), float(np.std(ch))])
        else:
            state.extend([0.5, 0.1])

        # 13. NOMA success rate
        if self.noma_history:
            nh = list(self.noma_history)[-100:]
            state.append(float(sum(1 for x in nh if x.get('success', False)) / len(nh)))
        else:
            state.append(0.5)

        # 14. Unique ratio
        if self.recent_requests:
            state.append(len(set(self.recent_requests)) / len(self.recent_requests))
        else:
            state.append(1.0)

        return np.array(state, dtype=np.float32)

    # =========================================================================
    # ACTION SELECTION (continuous -> threshold)
    # =========================================================================

    def _select_action(self, state: np.ndarray, file_id: int) -> Tuple[float, int]:
        """
        Select continuous action and convert to binary decision.
        Returns (continuous_action, binary_decision).
        """
        if file_id in self.cache_set:
            return 0.0, 0  # Already cached

        if TORCH_AVAILABLE:
            with torch.no_grad():
                t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action_value = self.actor(t).cpu().numpy()[0, 0]

            if not self.eval_mode:
                # Add OU noise for exploration
                noise = self.noise.sample()[0]
                action_value = np.clip(action_value + noise, 0.0, 1.0)

            binary = 1 if action_value >= 0.5 else 0
            return float(action_value), binary
        else:
            return 0.5, random.randint(0, 1)

    # =========================================================================
    # REWARD (identical to DQN)
    # =========================================================================

    def _compute_reward(self, action: int, file_id: int, cache_hit: bool,
                        cic_enabled: bool = False,
                        evicted_file: Optional[int] = None) -> float:
        if cache_hit:
            return 2.0

        pop_rank = self._get_file_popularity_rank(file_id)

        if action == 1:
            if pop_rank < 0.1:
                reward = 1.5
            elif pop_rank < 0.3:
                reward = 0.8
            elif pop_rank < 0.5:
                reward = 0.2
            else:
                reward = -0.5

            if evicted_file is not None:
                evicted_rank = self._get_file_popularity_rank(evicted_file)
                if evicted_rank > pop_rank:
                    reward += 0.5
                else:
                    reward -= 0.5

            if cic_enabled:
                reward += 1.5

            return reward
        else:
            if pop_rank < 0.2 and len(self.cache_set) < self.capacity:
                return -0.5
            elif pop_rank < 0.2:
                _, worst_file = self._get_min_popularity_slot()
                if worst_file >= 0:
                    worst_rank = self._get_file_popularity_rank(worst_file)
                    if worst_rank > pop_rank + 0.2:
                        return -0.3
                return 0.0
            else:
                return 0.1

    # =========================================================================
    # CACHE EXECUTION (identical to DQN)
    # =========================================================================

    def _execute_cache(self, file_id: int) -> Optional[int]:
        if file_id in self.cache_set:
            return None

        evicted_file = None
        if len(self.cache_set) >= self.capacity:
            worst_slot, worst_file = self._get_min_popularity_slot()
            if worst_slot >= 0 and worst_file >= 0:
                evicted_file = worst_file
                self.cache_set.discard(worst_file)
                if worst_file in self.file_to_slot:
                    del self.file_to_slot[worst_file]
                self.cache_slots[worst_slot] = file_id
                self.file_to_slot[file_id] = worst_slot
                self.cache_set.add(file_id)
                self._record_eviction()
        else:
            for slot_idx in range(self.capacity):
                if self.cache_slots[slot_idx] == -1:
                    self.cache_slots[slot_idx] = file_id
                    self.file_to_slot[file_id] = slot_idx
                    self.cache_set.add(file_id)
                    break

        return evicted_file

    # =========================================================================
    # MAIN REQUEST (same interface as DQN)
    # =========================================================================

    def request(
        self, item: int, user_id: Optional[int] = None,
        channel_gain: Optional[float] = None,
        paired_user: Optional[int] = None, paired_file: Optional[int] = None,
        noma_success: bool = True, outage: bool = False,
        ber: Optional[float] = None,
        sinr_weak: Optional[float] = None, sinr_strong: Optional[float] = None,
        episode_done: bool = False
    ) -> Dict:
        self.timestep += 1
        self._update_popularity(item)

        cache_hit = self.is_hit(item, update_stats=True)
        paired_cached = (
            self.is_hit(paired_file, update_stats=False)
            if paired_file is not None else False
        )

        result = {
            'hit': cache_hit, 'cic_enabled': False,
            'paired_user_cached': paired_cached,
            'weak_user_benefit': False, 'strong_user_benefit': False,
            'cache_size': len(self),
        }
        if self.enable_noma_awareness and paired_file is not None:
            if paired_cached:
                result['weak_user_benefit'] = True
                result['cic_enabled'] = True
                self.cic_opportunities += 1
            if cache_hit:
                result['strong_user_benefit'] = True
                result['cic_enabled'] = True
                self.noma_paired_hits += 1

        if user_id is not None and channel_gain is not None:
            self.channel_gains[user_id] = channel_gain
        if user_id is not None and paired_user is not None:
            self.user_pairings[user_id] = paired_user
        if channel_gain is not None:
            self.channel_history.append(float(channel_gain))
        if self.enable_noma_awareness:
            self.noma_history.append({
                'cic': result['cic_enabled'], 'success': noma_success,
                'sinr_weak': sinr_weak, 'sinr_strong': sinr_strong
            })
            if result['cic_enabled']:
                self.cic_count += 1
            if result['strong_user_benefit']:
                self.sic_count += 1

        # ── DDPG Learning ──
        if not self.eval_mode:
            state = self._get_state_vector(item)
            cont_action, binary_action = self._select_action(state, item)

            evicted_file = None
            if cache_hit:
                binary_action = 0
                cont_action = 0.0
            elif binary_action == 1:
                evicted_file = self._execute_cache(item)
                if paired_file is not None and item in self.cache_set:
                    result['cic_enabled'] = True

            reward = self._compute_reward(
                action=binary_action, file_id=item, cache_hit=cache_hit,
                cic_enabled=result['cic_enabled'], evicted_file=evicted_file
            )
            self.cumulative_reward += reward

            next_state = self._get_state_vector(item)

            if TORCH_AVAILABLE and self.replay_buffer is not None:
                self.replay_buffer.push({
                    'state': state,
                    'action': np.array([cont_action], dtype=np.float32),
                    'reward': reward,
                    'next_state': next_state,
                    'done': episode_done,
                })

            self.training_step += 1
            buf_len = len(self.replay_buffer) if self.replay_buffer is not None else 0
            if (TORCH_AVAILABLE
                    and buf_len >= self.warm_up_steps
                    and self.training_step % self.train_freq == 0
                    and buf_len >= self.batch_size):
                self._train_step()

            if episode_done:
                self.episode_rewards.append(self.cumulative_reward)
                self.cumulative_reward = 0.0
                self.noise.reset()

        else:
            # Eval mode
            if not cache_hit:
                state = self._get_state_vector(item)
                _, binary_action = self._select_action(state, item)
                if binary_action == 1:
                    self._execute_cache(item)

        return result

    # =========================================================================
    # TRAINING STEP (DDPG actor-critic update)
    # =========================================================================

    def _train_step(self):
        if not TORCH_AVAILABLE or self.replay_buffer is None:
            return
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(np.array([e['state'] for e in batch])).to(self.device)
        actions = torch.FloatTensor(np.array([e['action'] for e in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([e['reward'] for e in batch])).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array([e['next_state'] for e in batch])).to(self.device)
        dones = torch.FloatTensor(np.array([float(e['done']) for e in batch])).unsqueeze(1).to(self.device)

        # ── Critic update ──
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            q_target = rewards + self.gamma * (1 - dones) * self.critic_target(next_states, next_actions)

        q_current = self.critic(states, actions)
        critic_loss = F.mse_loss(q_current, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradient_clip)
        self.critic_optimizer.step()

        # ── Actor update ──
        predicted_actions = self.actor(states)
        actor_loss = -self.critic(states, predicted_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_clip)
        self.actor_optimizer.step()

        # ── Soft target update ──
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.losses.append(float(critic_loss.item()))

    # =========================================================================
    # ABSTRACT METHOD IMPLEMENTATIONS (required by CacheBase)
    # =========================================================================

    def populate(self, items=None):
        """Pre-load cache with items."""
        if items is None:
            top = np.argsort(-self.request_counts)[:self.capacity]
        else:
            top = list(items)[:self.capacity]
        self.cache_slots = [-1] * self.capacity
        self.cache_set.clear()
        self.file_to_slot.clear()
        for slot, fid in enumerate(top):
            fid = int(fid)
            self.cache_slots[slot] = fid
            self.file_to_slot[fid] = slot
            self.cache_set.add(fid)

    def is_hit(self, item: int, update_stats: bool = True) -> bool:
        hit = int(item) in self.cache_set
        if update_stats:
            self._record_hit() if hit else self._record_miss()
        return hit

    def get_contents(self):
        return set(self.cache_set)

    # =========================================================================
    # INTERFACE METHODS
    # =========================================================================

    def set_eval_mode(self, mode: bool):
        self.eval_mode = mode

    def clear(self):
        self.cache_set.clear()
        self.file_to_slot.clear()
        self.cache_slots = [-1] * self.capacity
        self.timestep = 0
        # Keep weights, reset cache contents only
        super().clear()

    def reset_popularity(self):
        self.request_counts = np.zeros(self.num_files, dtype=np.float64)
        self.recent_requests.clear()

    def get_stats(self) -> Dict:
        avg_loss = float(np.mean(self.losses[-100:])) if self.losses else 0.0
        return {
            'epsilon': float(self.noise.sigma) if TORCH_AVAILABLE else 0.0,
            'avg_loss': avg_loss,
            'training_steps': self.training_step,
            'buffer_size': len(self.replay_buffer) if self.replay_buffer else 0,
            'beta': 0.0,  # No PER in DDPG (for compat)
            'algorithm': 'DDPG',
        }

    def save_model(self, filepath: str):
        if TORCH_AVAILABLE:
            torch.save({
                'actor': self.actor.state_dict(),
                'critic': self.critic.state_dict(),
                'actor_target': self.actor_target.state_dict(),
                'critic_target': self.critic_target.state_dict(),
                'actor_optimizer': self.actor_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict(),
            }, filepath)

    def load_model(self, filepath: str):
        if TORCH_AVAILABLE:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.actor_target.load_state_dict(checkpoint['actor_target'])
            self.critic_target.load_state_dict(checkpoint['critic_target'])
