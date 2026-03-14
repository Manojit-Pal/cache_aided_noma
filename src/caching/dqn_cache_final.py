"""
src/caching/dqn_cache_final.py

NOMA-AWARE DEEP Q-NETWORK CACHE
===============================

Implementation based on:
- IEEE DeepChunk (2019): Deep Q-Learning for Chunk-based Caching
- RLCaR: Reinforcement Learning Cache Replacement
- Wang et al. (2016): Dueling DQN architecture
- Schaul et al. (2016): Prioritized Experience Replay

2026 Bug Fix Log:
- BUG-1  (CRITICAL): Replaced last_state/last_action pattern with
  pending_transitions dict for correct deferred reward assignment.
- BUG-3  (CRITICAL): EMA popularity decays ALL files every step.
- BUG-4  (MODERATE): O(N^2) _get_state_vector fixed to O(N).
- BUG-9  (MODERATE): populate() clears pending_transitions.
- BUG-12 (MODERATE): save/load persists pending_transitions.
- FIX-3  (HIGH): Reward redesign — outage removed; CIC +2->+3;
  miss+success neutral (0.0).
- BUG-DQN-2 (CRITICAL): popularity NaN guard on float32 underflow.
- BUG-DQN-4 (MODERATE): empty-slot fill uses neutral reward 0.0.
- DQN-A  (CRITICAL): _update_counters() moved BEFORE
  _learn_from_request() so all state vectors see fresh LRU/LFU
  counters instead of one-step-lagged values.
- DQN-J  (MODERATE): popularity bump moved BEFORE
  _learn_from_request() so state features reflect current request.
- DQN-I  (MODERATE): clear() now resets cumulative_reward to 0.0
  to prevent reward accumulator carry-over between eval runs.

Author: Cache-Aided NOMA Team
Date: December 2025 | Revised: March 2026
"""

import numpy as np
import random
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional, Set, Iterable
import pickle

try:
    from .. import config as cfg_module
except ImportError:
    cfg_module = None

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print('Warning: PyTorch not available — DQN will use Q-table fallback')

from .cache_base import CacheBase


# ============================================================================
# DUELING DQN NETWORK
# ============================================================================

class DuelingDQN(nn.Module):
    """
    Dueling DQN (Wang et al., ICML 2016).
    Q(s,a) = V(s) + [A(s,a) - mean_a A(s,a)]
    """
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: List[int] = [128, 64]):
        super().__init__()
        self.feature_layers = nn.ModuleList()
        prev_dim = state_dim
        for hdim in hidden_dims:
            self.feature_layers.append(nn.Linear(prev_dim, hdim))
            prev_dim = hdim
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2), nn.ReLU(),
            nn.Linear(hidden_dims[-1] // 2, 1))
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2), nn.ReLU(),
            nn.Linear(hidden_dims[-1] // 2, action_dim))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, state: 'torch.Tensor') -> 'torch.Tensor':
        x = state
        for layer in self.feature_layers:
            x = F.relu(layer(x))
        value     = self.value_stream(x)
        advantage = self.advantage_stream(x)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


# ============================================================================
# PRIORITIZED EXPERIENCE REPLAY
# ============================================================================

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay with beta annealing (Schaul et al., 2016)."""

    def __init__(self, capacity: int, alpha: float = 0.6,
                 beta_start: float = 0.4, beta_end: float = 1.0,
                 beta_frames: int = 100000):
        self.capacity     = capacity
        self.alpha        = alpha
        self.beta         = beta_start
        self.beta_start   = beta_start
        self.beta_end     = beta_end
        self.beta_frames  = beta_frames
        self.frame_idx    = 0
        self.buffer       = deque(maxlen=capacity)
        self.priorities   = deque(maxlen=capacity)
        self.max_priority = 1.0

    def add(self, experience: Dict):
        self.buffer.append(experience)
        self.priorities.append(self.max_priority)

    def sample(self, batch_size: int) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
        if len(self.buffer) < batch_size:
            return None, None, None
        self.frame_idx += 1
        self.beta = min(
            self.beta_end,
            self.beta_start + (self.beta_end - self.beta_start)
            * (self.frame_idx / self.beta_frames))
        priorities = np.array(self.priorities, dtype=np.float64)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        use_replace = len(self.buffer) < 3 * batch_size
        try:
            indices = np.random.choice(len(self.buffer), batch_size,
                                       p=probs, replace=use_replace)
        except ValueError:
            indices = np.random.choice(len(self.buffer), batch_size,
                                       replace=use_replace)
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        return [self.buffer[i] for i in indices], weights, indices

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        for idx, err in zip(indices, td_errors):
            p = float(abs(err) + 1e-6)
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = p
                self.max_priority    = max(self.max_priority, p)

    def get_beta(self) -> float:
        return self.beta

    def __len__(self):
        return len(self.buffer)


# ============================================================================
# NOMA-AWARE DQN CACHE
# ============================================================================

class DQNCache(CacheBase):
    """
    Deep Q-Network Cache for Cache-Aided NOMA.

    MDP:
      State  : LRU + LFU counters (per slot) + 6 global features
      Action : which slot to evict (0 .. capacity-1)
      Reward : deferred via pending_transitions (see _learn_from_request)

    Reward hierarchy (FIX-3):
      +10  cache hit on evicted file
      + 3  miss + CIC enabled
      +  0  miss + NOMA success (neutral)
      - 5  miss + NOMA failure

    Call ordering in request() (critical for correct state vectors):
      1. is_hit()                  — check cache, record stat
      2. build result dict         — NOMA flags
      3. update channel/NOMA logs  — history deques
      4. _update_counters()        — MUST be before step 5 & 6
      5. popularity update         — MUST be before step 6
      6. _learn_from_request()     — sees fresh counters + popularity
    """

    def __init__(
        self, capacity: int, num_files: int, num_users: int,
        learning_rate: float = 0.0001, gamma: float = 0.99,
        epsilon_start: float = 1.0, epsilon_end: float = 0.01,
        epsilon_decay_steps: int = 200000,
        use_neural_network: bool = True,
        hidden_dims: List[int] = [128, 128],
        batch_size: int = 64, replay_buffer_size: int = 50000,
        train_freq: int = 4, warm_up_steps: Optional[int] = None,
        use_prioritized_replay: bool = True,
        priority_alpha: float = 0.6, priority_beta_start: float = 0.4,
        priority_beta_end: float = 1.0, priority_beta_frames: int = 100000,
        gradient_clip: float = 10.0, tau: float = 0.005,
        enable_noma_awareness: bool = True, seed: int = 2025
    ):
        super().__init__(capacity, enable_noma_awareness)
        self.num_files     = num_files
        self.num_users     = num_users
        self.lr            = learning_rate
        self.gamma         = gamma
        self.batch_size    = batch_size
        self.train_freq    = train_freq
        self.gradient_clip = gradient_clip
        self.tau           = tau
        self.warm_up_steps = max(10 * batch_size, 1000) if warm_up_steps is None else warm_up_steps
        self._set_seeds(seed)

        self.epsilon       = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon_decay = (epsilon_start - epsilon_end) / max(1, epsilon_decay_steps)
        self.eval_mode     = False

        self.cache_slots  = [-1] * capacity
        self.file_to_slot = {}
        self.lru_counters = np.zeros(capacity, dtype=np.int32)
        self.lfu_counters = np.zeros(capacity, dtype=np.int32)
        self.timestep     = 0

        self.popularity       = np.ones(num_files, dtype=np.float32) / num_files
        self.popularity_decay = 0.9

        self.channel_history = deque(maxlen=500)
        self.noma_history    = deque(maxlen=500)
        self.cic_count       = 0
        self.sic_count       = 0

        self.use_nn        = use_neural_network and TORCH_AVAILABLE
        self.training_step = 0
        self.action_dim    = capacity

        if self.use_nn:
            self.state_dim      = 2 * capacity + 6
            self.device         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.q_network      = DuelingDQN(self.state_dim, self.action_dim, hidden_dims).to(self.device)
            self.target_network = DuelingDQN(self.state_dim, self.action_dim, hidden_dims).to(self.device)
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.target_network.eval()
            self.optimizer = optim.Adam(self.q_network.parameters(),
                                        lr=self.lr, weight_decay=1e-5)
            if use_prioritized_replay:
                self.replay_buffer   = PrioritizedReplayBuffer(
                    replay_buffer_size, priority_alpha,
                    priority_beta_start, priority_beta_end, priority_beta_frames)
                self.use_prioritized = True
            else:
                self.replay_buffer   = deque(maxlen=replay_buffer_size)
                self.use_prioritized = False
        else:
            self.q_table         = defaultdict(lambda: np.zeros(self.action_dim))
            self.replay_buffer   = None
            self.use_prioritized = False

        self.episode_rewards   = []
        self.losses            = []
        self.cumulative_reward = 0.0
        self.pending_transitions: Dict[int, Dict] = {}

        print('DQNCache initialized')
        print(f'  Mode      : {"Neural Network (DQN)" if self.use_nn else "Q-table"}')
        print(f'  Warm-up   : {self.warm_up_steps} steps')
        print(f'  Fixes     : DQN-A/J (call order), DQN-I (clear reset), '
              f'BUG-DQN-2 (NaN guard), BUG-DQN-4 (neutral fill reward)')

    def _set_seeds(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        if TORCH_AVAILABLE:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    # -------------------------------------------------------------------------
    # STATE (O(N))
    # -------------------------------------------------------------------------
    def _get_state_vector(self, requested_file: int) -> np.ndarray:
        state    = []
        occupied = np.array([s != -1 for s in self.cache_slots], dtype=bool)
        has_occ  = occupied.any()
        max_lru  = max(int(self.lru_counters[occupied].max()), 1) if has_occ else 1
        max_lfu  = max(int(self.lfu_counters[occupied].max()), 1) if has_occ else 1

        for i in range(self.capacity):
            state.append(-1.0 if not occupied[i] else float(self.lru_counters[i] / max_lru))
        for i in range(self.capacity):
            state.append( 0.0 if not occupied[i] else float(self.lfu_counters[i] / max_lfu))

        state.append(float(self.popularity[requested_file]))
        state.append(float(occupied.sum() / self.capacity))

        if self.channel_history:
            ch = list(self.channel_history)[-100:]
            state.extend([float(np.mean(ch)), float(np.std(ch))])
        else:
            state.extend([0.5, 0.1])

        if self.enable_noma_awareness and self.noma_history:
            nh = list(self.noma_history)[-100:]
            state.append(float(sum(1 for x in nh if x.get('cic',     False)) / len(nh)))
            state.append(float(sum(1 for x in nh if x.get('success', False)) / len(nh)))
        else:
            state.extend([0.0, 0.5])

        return np.array(state, dtype=np.float32)

    # -------------------------------------------------------------------------
    # ACTION SELECTION
    # -------------------------------------------------------------------------
    def _select_action(self, state: np.ndarray, file_id: int) -> int:
        if file_id in self.file_to_slot:
            return -1
        empty = [i for i, f in enumerate(self.cache_slots) if f == -1]
        if empty:
            return empty[0]
        if self.eval_mode or random.random() >= self.epsilon:
            if self.use_nn:
                with torch.no_grad():
                    t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    return int(self.q_network(t).cpu().numpy()[0].argmax())
            else:
                return int(self.q_table[self._discretize_state(state)].argmax())
        return random.randint(0, self.capacity - 1)

    def _discretize_state(self, state: np.ndarray) -> str:
        bins = []
        for v in state[:min(10, len(state))]:
            bins.append('E' if v < 0 else 'L' if v < 0.33 else 'M' if v < 0.67 else 'H')
        return ''.join(bins)

    # -------------------------------------------------------------------------
    # REWARD (FIX-3)
    # -------------------------------------------------------------------------
    def _compute_reward(
        self, cache_hit: bool, cic_enabled: bool = False,
        noma_success: bool = True, outage: bool = False,
        ber: Optional[float] = None
    ) -> float:
        """
        Deferred reward for eviction decisions.
        Outage excluded (channel quality independent of caching choice).
        """
        if cache_hit:        return 10.0
        if not noma_success: return -5.0
        reward = 3.0 if cic_enabled else 0.0
        if ber is not None:
            if   ber < 1e-4: reward += 1.0
            elif ber > 1e-2: reward -= 2.0
        return reward

    # -------------------------------------------------------------------------
    # MAIN REQUEST
    # -------------------------------------------------------------------------
    def request(
        self, item: int, user_id: Optional[int] = None,
        channel_gain: Optional[float] = None,
        paired_user: Optional[int] = None, paired_file: Optional[int] = None,
        noma_success: bool = True, outage: bool = False,
        ber: Optional[float] = None,
        sinr_weak: Optional[float] = None, sinr_strong: Optional[float] = None,
        episode_done: bool = False
    ) -> Dict:
        """
        Handle one file request with NOMA-aware DQN learning.

        Critical call ordering (DQN-A / DQN-J fix):
          _update_counters() and popularity bump happen BEFORE
          _learn_from_request() so all _get_state_vector() calls
          inside the learning step see fresh, up-to-date cache state.
        """
        self.timestep += 1
        cache_hit     = self.is_hit(item, update_stats=True)
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
                result['cic_enabled']       = True
                self.cic_opportunities     += 1
            if cache_hit:
                result['strong_user_benefit'] = True
                result['cic_enabled']         = True
                self.noma_paired_hits        += 1

        if user_id is not None and channel_gain is not None:
            self.channel_gains[user_id] = channel_gain
        if user_id is not None and paired_user is not None:
            self.user_pairings[user_id] = paired_user
        if channel_gain is not None:
            self.channel_history.append(float(channel_gain))
        if self.enable_noma_awareness:
            self.noma_history.append({'cic': result['cic_enabled'],
                                      'success': noma_success,
                                      'sinr_weak': sinr_weak,
                                      'sinr_strong': sinr_strong})
            if result['cic_enabled']:         self.cic_count += 1
            if result['strong_user_benefit']: self.sic_count += 1

        # ── DQN-A FIX: update LRU/LFU counters BEFORE _learn_from_request()
        # so every _get_state_vector() call inside learning sees the
        # post-request cache state (hit-slot LRU reset, miss insertion).
        self._update_counters(item, cache_hit)

        # ── DQN-J FIX: update popularity BEFORE _learn_from_request()
        # so popularity[item] in state vectors reflects the current request.
        self.popularity *= self.popularity_decay
        self.popularity[item] += (1.0 - self.popularity_decay)
        # BUG-DQN-2 FIX: guard float32 underflow → NaN
        total = self.popularity.sum()
        if total > 1e-30:
            self.popularity /= total
        else:
            self.popularity = np.ones(self.num_files, dtype=np.float32) / self.num_files

        if not self.eval_mode:
            self._learn_from_request(
                file_id=item, cache_hit=cache_hit,
                cic_enabled=result['cic_enabled'],
                noma_success=noma_success, outage=outage,
                ber=ber, episode_done=episode_done)

        return result

    # -------------------------------------------------------------------------
    # DEFERRED CREDIT ASSIGNMENT
    # -------------------------------------------------------------------------
    def _learn_from_request(
        self, file_id: int, cache_hit: bool, cic_enabled: bool,
        noma_success: bool, outage: bool, ber: Optional[float], episode_done: bool
    ):
        """
        DQN update using deferred pending_transitions.

        By the time this is called, _update_counters() and popularity
        have already been updated in request(), so all
        _get_state_vector() calls here see the correct post-request
        cache state (DQN-A / DQN-J fix).
        """
        # Resolve deferred reward for file_id if it was previously evicted
        if file_id in self.pending_transitions:
            pending = self.pending_transitions.pop(file_id)
            reward  = self._compute_reward(
                cache_hit=cache_hit, cic_enabled=cic_enabled,
                noma_success=noma_success, outage=outage, ber=ber)
            self._push_to_buffer({
                'state':      pending['state'],
                'action':     pending['action'],
                'reward':     reward,
                'next_state': self._get_state_vector(file_id),  # fresh state
                'done':       episode_done,
            })
            self.cumulative_reward += reward

        if not cache_hit:
            empty_slots  = [i for i, f in enumerate(self.cache_slots) if f == -1]
            cache_full   = len(empty_slots) == 0
            state_before = self._get_state_vector(file_id)  # fresh state
            action       = self._select_action(state_before, file_id)

            if action >= 0:
                if cache_full:
                    evicted_file = self.cache_slots[action]
                    self._execute_action(action, file_id)
                    state_after = self._get_state_vector(file_id)
                    if evicted_file != -1:
                        if evicted_file in self.pending_transitions:
                            old = self.pending_transitions.pop(evicted_file)
                            self._push_to_buffer({
                                'state':      old['state'],
                                'action':     old['action'],
                                'reward':     0.0,
                                'next_state': state_before,
                                'done':       False,
                            })
                        self.pending_transitions[evicted_file] = {
                            'state':  state_before,
                            'action': action,
                            'next_state': state_after,
                        }
                else:
                    # BUG-DQN-4 FIX: filling an empty slot is neutral —
                    # no eviction trade-off, use reward=0.0.
                    self._execute_action(action, file_id)
                    state_after = self._get_state_vector(file_id)
                    self._push_to_buffer({
                        'state':      state_before,
                        'action':     action,
                        'reward':     0.0,
                        'next_state': state_after,
                        'done':       episode_done,
                    })

        self.training_step += 1
        buf_len = len(self.replay_buffer) if self.replay_buffer is not None else 0
        if (self.use_nn
                and buf_len >= self.warm_up_steps
                and self.training_step % self.train_freq == 0
                and buf_len >= self.batch_size):
            self._train_step()

        if not self.eval_mode and self.epsilon > self.epsilon_end:
            self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)

        if episode_done:
            self.episode_rewards.append(self.cumulative_reward)
            self.cumulative_reward = 0.0
            for ef, pending in list(self.pending_transitions.items()):
                self._push_to_buffer({
                    'state':      pending['state'],
                    'action':     pending['action'],
                    'reward':     0.0,
                    'next_state': self._get_state_vector(file_id),
                    'done':       True,
                })
            self.pending_transitions.clear()

    def _push_to_buffer(self, exp: Dict):
        if self.use_nn and self.replay_buffer is not None:
            if self.use_prioritized:
                self.replay_buffer.add(exp)
            else:
                self.replay_buffer.append(exp)
        elif not self.use_nn:
            self._update_q_table(exp)

    def _execute_action(self, action: int, file_id: int):
        """Place file_id into cache slot `action`, evicting whatever was there."""
        if action < 0 or action >= self.capacity:
            return
        old = self.cache_slots[action]
        if old != -1 and old in self.file_to_slot:
            del self.file_to_slot[old]
        self.cache_slots[action]   = file_id
        self.file_to_slot[file_id] = action
        self.lru_counters[action]  = 0
        self.lfu_counters[action]  = 1

    def _update_counters(self, file_id: int, cache_hit: bool):
        """
        Age all LRU counters by 1; refresh the hit slot's LRU to 0
        and increment its LFU.

        DQN-A FIX: called BEFORE _learn_from_request() in request()
        so state vectors inside the learning step see fresh counters.
        """
        for i in range(self.capacity):
            if self.cache_slots[i] != -1:
                self.lru_counters[i] += 1
        if cache_hit and file_id in self.file_to_slot:
            s = self.file_to_slot[file_id]
            self.lru_counters[s] = 0
            self.lfu_counters[s] += 1

    # -------------------------------------------------------------------------
    # TRAINING (Double DQN + Soft Target Update + PER)
    # -------------------------------------------------------------------------
    def _train_step(self):
        if not self.use_nn or len(self.replay_buffer) < self.batch_size:
            return
        if self.use_prioritized:
            exps, weights, indices = self.replay_buffer.sample(self.batch_size)
            if exps is None:
                return
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            exps    = random.sample(list(self.replay_buffer), self.batch_size)
            weights = torch.ones(self.batch_size).to(self.device)
            indices = None

        def _t(key, dtype):
            return torch.from_numpy(
                np.array([e[key] for e in exps], dtype=dtype)).to(self.device)

        states      = _t('state',      np.float32)
        actions     = _t('action',     np.int64)
        rewards     = _t('reward',     np.float32)
        next_states = _t('next_state', np.float32)
        dones       = _t('done',       np.float32)

        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_a   = self.q_network(next_states).argmax(1)
            next_q   = self.target_network(next_states).gather(
                           1, next_a.unsqueeze(1)).squeeze(1)
            target_q = rewards + (1.0 - dones) * self.gamma * next_q
        td_errors = target_q - current_q
        loss = (weights * F.smooth_l1_loss(
            current_q, target_q, reduction='none')).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.gradient_clip)
        self.optimizer.step()
        self._soft_update_target()

        if self.use_prioritized and indices is not None:
            self.replay_buffer.update_priorities(
                indices, td_errors.detach().cpu().numpy())
        self.losses.append(float(loss.item()))

    def _soft_update_target(self):
        """Polyak averaging: theta_target <- tau*theta + (1-tau)*theta_target"""
        for tp, p in zip(self.target_network.parameters(),
                         self.q_network.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

    def _update_q_table(self, exp: Dict):
        sk  = self._discretize_state(exp['state'])
        nk  = self._discretize_state(exp['next_state'])
        a, r, done = exp['action'], exp['reward'], exp['done']
        cq  = self.q_table[sk][a]
        nq  = 0.0 if done else float(np.max(self.q_table[nk]))
        self.q_table[sk][a] += self.lr * (r + self.gamma * nq - cq)

    # -------------------------------------------------------------------------
    # CACHE INTERFACE
    # -------------------------------------------------------------------------
    def populate(self, items: Optional[Iterable[int]] = None):
        """Pre-load cache with top-popularity files."""
        self.pending_transitions.clear()
        top = (np.argsort(-self.popularity)[:self.capacity]
               if items is None else list(items)[:self.capacity])
        self.cache_slots  = [-1] * self.capacity
        self.file_to_slot.clear()
        for slot, fid in enumerate(top):
            self.cache_slots[slot]      = int(fid)
            self.file_to_slot[int(fid)] = slot
            self.lfu_counters[slot]     = 1
            self.lru_counters[slot]     = 0

    def is_hit(self, item: int, update_stats: bool = True) -> bool:
        hit = int(item) in self.file_to_slot
        if update_stats:
            self._record_hit() if hit else self._record_miss()
        return hit

    def get_contents(self) -> Set[int]:
        return set(f for f in self.cache_slots if f != -1)

    def clear(self):
        """
        Reset cache state.
        DQN-I FIX: also resets cumulative_reward to 0.0 to prevent
        reward accumulator carry-over between evaluation runs.
        Note: episode_rewards and losses are intentionally preserved
        as training history (useful for post-hoc analysis).
        """
        self.cache_slots       = [-1] * self.capacity
        self.file_to_slot.clear()
        self.lru_counters      = np.zeros(self.capacity, dtype=np.int32)
        self.lfu_counters      = np.zeros(self.capacity, dtype=np.int32)
        self.pending_transitions.clear()
        self.cumulative_reward = 0.0   # DQN-I FIX
        self.reset_stats()

    def set_eval_mode(self, eval_mode: bool = True):
        """Toggle eval mode; preserves/restores training epsilon."""
        self.eval_mode = eval_mode
        if eval_mode:
            if not hasattr(self, '_training_epsilon'):
                self._training_epsilon = self.epsilon
            self.epsilon = 0.0
            if self.use_nn: self.q_network.eval()
        else:
            if hasattr(self, '_training_epsilon'):
                self.epsilon = self._training_epsilon
                delattr(self, '_training_epsilon')
            if self.use_nn: self.q_network.train()

    # -------------------------------------------------------------------------
    # MODEL PERSISTENCE
    # -------------------------------------------------------------------------
    def save_model(self, filepath: str):
        if self.use_nn:
            d = {
                'q_network':           self.q_network.state_dict(),
                'target_network':      self.target_network.state_dict(),
                'optimizer':           self.optimizer.state_dict(),
                'training_step':       self.training_step,
                'epsilon':             self.epsilon,
                'popularity':          self.popularity,
                'cache_slots':         self.cache_slots,
                'file_to_slot':        self.file_to_slot,
                'lru_counters':        self.lru_counters,
                'lfu_counters':        self.lfu_counters,
                'pending_transitions': self.pending_transitions,
            }
            if self.use_prioritized:
                d['per_beta']      = self.replay_buffer.get_beta()
                d['per_frame_idx'] = self.replay_buffer.frame_idx
            torch.save(d, filepath)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'q_table':             dict(self.q_table),
                    'training_step':       self.training_step,
                    'epsilon':             self.epsilon,
                    'popularity':          self.popularity,
                    'cache_slots':         self.cache_slots,
                    'file_to_slot':        self.file_to_slot,
                    'pending_transitions': self.pending_transitions,
                }, f)
        print(f'Model saved to {filepath}')

    def load_model(self, filepath: str):
        if self.use_nn:
            ckpt = torch.load(filepath, map_location=self.device,
                              weights_only=False)
            self.q_network.load_state_dict(ckpt['q_network'])
            self.target_network.load_state_dict(ckpt['target_network'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.training_step       = ckpt['training_step']
            self.epsilon             = ckpt['epsilon']
            self.popularity          = ckpt['popularity']
            self.cache_slots         = ckpt['cache_slots']
            self.file_to_slot        = ckpt['file_to_slot']
            self.lru_counters        = ckpt.get('lru_counters', self.lru_counters)
            self.lfu_counters        = ckpt.get('lfu_counters', self.lfu_counters)
            self.pending_transitions = ckpt.get('pending_transitions', {})
            if self.use_prioritized and 'per_beta' in ckpt:
                self.replay_buffer.beta      = ckpt['per_beta']
                self.replay_buffer.frame_idx = ckpt.get('per_frame_idx', 0)
        else:
            with open(filepath, 'rb') as f:
                ckpt = pickle.load(f)
            self.q_table             = defaultdict(
                lambda: np.zeros(self.action_dim), ckpt['q_table'])
            self.training_step       = ckpt['training_step']
            self.epsilon             = ckpt['epsilon']
            self.popularity          = ckpt['popularity']
            self.cache_slots         = ckpt['cache_slots']
            self.file_to_slot        = ckpt['file_to_slot']
            self.pending_transitions = ckpt.get('pending_transitions', {})
        print(f'Model loaded from {filepath}')

    # -------------------------------------------------------------------------
    # STATISTICS
    # -------------------------------------------------------------------------
    def get_stats(self) -> Dict:
        base = super().stats()
        dqn  = {
            'training_step':       self.training_step,
            'epsilon':             self.epsilon,
            'eval_mode':           self.eval_mode,
            'avg_episode_reward':  float(np.mean(self.episode_rewards[-100:])) if self.episode_rewards else 0.0,
            'avg_loss':            float(np.mean(self.losses[-100:]))           if self.losses          else 0.0,
            'replay_buffer_size':  len(self.replay_buffer) if self.replay_buffer else 0,
            'use_neural_network':  self.use_nn,
            'cic_count':           self.cic_count,
            'sic_count':           self.sic_count,
            'warm_up_steps':       self.warm_up_steps,
            'pending_transitions': len(self.pending_transitions),
            'beta':                self.replay_buffer.get_beta() if (
                                       self.use_prioritized and self.replay_buffer
                                   ) else 0.0,
        }
        return {**base, **dqn}


# Alias for compatibility
StableDQNCache = DQNCache
