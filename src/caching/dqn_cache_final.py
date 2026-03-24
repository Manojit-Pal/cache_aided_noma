"""
src/caching/dqn_cache_final.py

NOMA-AWARE DEEP Q-NETWORK CACHE  (v2 — Binary-Action Redesign)
===============================================================

Complete redesign of the DQN cache to fix fundamental learning issues:

1. BINARY ACTION SPACE: action ∈ {0=skip, 1=cache}
   - When caching + cache full → evict least-popular slot (heuristic)
   - Reduces action space from 200 to 2 → tractable learning

2. COMPACT STATE VECTOR (~15 dims instead of 406):
   - Requested file popularity rank (normalized)
   - Requested file recent frequency
   - Whether file already cached
   - Cache occupancy
   - Mean/min popularity of cached files
   - CIC potential
   - Channel & NOMA statistics

3. IMMEDIATE REWARDS (no deferred credit assignment):
   - Based on caching decision quality relative to file popularity
   - Cache hits, CIC enablement, popularity-aware eviction quality

4. NO pending_transitions — reward is computed immediately

References:
- IEEE DeepChunk (2019): Deep Q-Learning for Chunk-based Caching
- Wang et al. (2016): Dueling DQN architecture
- Schaul et al. (2016): Prioritized Experience Replay

Author: Cache-Aided NOMA Team
Date: March 2026 (v2 redesign)
"""

import numpy as np
import random
from collections import deque, defaultdict, Counter
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
# DUELING DQN NETWORK (smaller for binary action space)
# ============================================================================

class DuelingDQN(nn.Module):
    """
    Dueling DQN (Wang et al., ICML 2016).
    Q(s,a) = V(s) + [A(s,a) - mean_a A(s,a)]

    Compact network for binary action space.
    """
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: List[int] = [64, 32]):
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
# NOMA-AWARE DQN CACHE (v2 — Binary Action)
# ============================================================================

class DQNCache(CacheBase):
    """
    Deep Q-Network Cache for Cache-Aided NOMA (v2 redesign).

    MDP:
      State  : compact ~15-dim vector (file popularity, cache state, NOMA stats)
      Action : binary {0=skip, 1=cache this file}
      Reward : immediate, based on caching decision quality

    Key design decisions:
      - Binary action makes learning tractable (2 actions instead of 200)
      - When action=1 and cache is full, evict the least-popular cached file
      - Immediate rewards provide dense learning signal
      - Compact state focuses on what matters: file popularity & cache state
    """

    # Number of recent requests to track for frequency estimation
    RECENT_WINDOW = 500

    def __init__(
        self, capacity: int, num_files: int, num_users: int,
        learning_rate: float = 0.001, gamma: float = 0.99,
        epsilon_start: float = 1.0, epsilon_end: float = 0.01,
        epsilon_decay_steps: int = 200000,
        use_neural_network: bool = True,
        hidden_dims: List[int] = [64, 32],
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
        self.warm_up_steps = max(5 * batch_size, 500) if warm_up_steps is None else warm_up_steps
        self._set_seeds(seed)

        self.epsilon       = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon_decay = (epsilon_start - epsilon_end) / max(1, epsilon_decay_steps)
        self.eval_mode     = False

        # Cache storage
        self.cache_set    = set()       # set of cached file IDs
        self.file_to_slot = {}          # file_id -> slot index (for compat)
        self.cache_slots  = [-1] * capacity
        self.timestep     = 0

        # Popularity tracking
        self.request_counts = np.zeros(num_files, dtype=np.float64)
        self.recent_requests = deque(maxlen=self.RECENT_WINDOW)
        self.popularity_rank = np.arange(num_files)  # initial: 0 is most popular

        # NOMA tracking
        self.channel_history = deque(maxlen=500)
        self.noma_history    = deque(maxlen=500)
        self.cic_count       = 0
        self.sic_count       = 0

        # Binary action space: {0=skip, 1=cache}
        self.action_dim = 2

        self.use_nn        = use_neural_network and TORCH_AVAILABLE
        self.training_step = 0

        # State dimension: compact features
        # [file_pop_rank, file_recent_freq, is_cached, cache_occupancy,
        #  mean_pop_cached, min_pop_cached, max_pop_cached,
        #  would_improve_cache, would_improve_ratio,
        #  cic_potential, channel_mean, channel_std,
        #  noma_success_rate, num_unique_recent]
        self.state_dim = 14

        if self.use_nn:
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

        print('DQNCache v2 initialized (binary action space)')
        print(f'  State dim  : {self.state_dim}')
        print(f'  Action dim : {self.action_dim} (0=skip, 1=cache)')
        print(f'  Network    : {hidden_dims}')
        print(f'  Warm-up    : {self.warm_up_steps} steps')

    def _set_seeds(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        if TORCH_AVAILABLE:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    # -------------------------------------------------------------------------
    # POPULARITY TRACKING
    # -------------------------------------------------------------------------
    def _update_popularity(self, file_id: int):
        """Update request counts and popularity ranking."""
        self.request_counts[file_id] += 1
        self.recent_requests.append(file_id)

    def _get_file_popularity_rank(self, file_id: int) -> float:
        """
        Get normalized popularity rank of a file (0=most popular, 1=least).
        Based on cumulative request counts.
        """
        # Rank files by request count (descending)
        rank = np.sum(self.request_counts > self.request_counts[file_id])
        return float(rank) / max(self.num_files - 1, 1)

    def _get_file_recent_freq(self, file_id: int) -> float:
        """Frequency of file in recent request window."""
        if len(self.recent_requests) == 0:
            return 0.0
        count = sum(1 for f in self.recent_requests if f == file_id)
        return float(count) / len(self.recent_requests)

    def _get_min_popularity_slot(self) -> Tuple[int, int]:
        """
        Find the cache slot with the least popular file (best eviction candidate).
        Returns (slot_index, file_id).
        """
        worst_slot = -1
        worst_file = -1
        worst_count = float('inf')
        for slot_idx in range(self.capacity):
            fid = self.cache_slots[slot_idx]
            if fid == -1:
                continue
            if self.request_counts[fid] < worst_count:
                worst_count = self.request_counts[fid]
                worst_slot  = slot_idx
                worst_file  = fid
        return worst_slot, worst_file

    # -------------------------------------------------------------------------
    # STATE (compact ~15 dims)
    # -------------------------------------------------------------------------
    def _get_state_vector(self, requested_file: int) -> np.ndarray:
        """
        Compact state vector focused on what matters for caching decisions.
        """
        state = []

        # 1. Requested file's popularity rank (normalized 0=most popular)
        pop_rank = self._get_file_popularity_rank(requested_file)
        state.append(pop_rank)

        # 2. Requested file's recent frequency
        recent_freq = self._get_file_recent_freq(requested_file)
        state.append(recent_freq)

        # 3. Whether file is already cached (0 or 1)
        is_cached = 1.0 if requested_file in self.cache_set else 0.0
        state.append(is_cached)

        # 4. Cache occupancy (0 to 1)
        occupancy = len(self.cache_set) / self.capacity
        state.append(occupancy)

        # Cache popularity statistics
        if self.cache_set and len(self.cache_set) > 0:
            cached_counts = np.array([self.request_counts[f] for f in self.cache_set if f >= 0])
            total_counts = max(self.request_counts.sum(), 1.0)
            
            if len(cached_counts) > 0:
                cached_pops = cached_counts / total_counts
                # 5. Mean popularity of cached files
                state.append(float(cached_pops.mean()))
                # 6. Min popularity (= worst cached file)
                state.append(float(cached_pops.min()))
                # 7. Max popularity (= best cached file)
                state.append(float(cached_pops.max()))
            else:
                state.extend([0.0, 0.0, 0.0])
        else:
            state.extend([0.0, 0.0, 0.0])

        # 8. Popularity of worst cached file compared to requested file
        if self.cache_set:
            _, worst_file = self._get_min_popularity_slot()
            if worst_file >= 0:
                would_improve = 1.0 if self.request_counts[requested_file] > self.request_counts[worst_file] else 0.0
            else:
                would_improve = 1.0
        else:
            would_improve = 1.0
        state.append(would_improve)

        # 9. Would caching improve — ratio of request counts
        if self.cache_set and len(self.cache_set) >= self.capacity:
            _, worst_file = self._get_min_popularity_slot()
            if worst_file >= 0 and self.request_counts[worst_file] > 0:
                ratio = float(self.request_counts[requested_file] / self.request_counts[worst_file])
                state.append(min(ratio, 5.0) / 5.0)  # normalized 0-1
            else:
                state.append(1.0)
        else:
            state.append(1.0)  # cache not full, always beneficial to add

        # 10. CIC potential (stub — we don't know paired file yet, use history)
        if self.noma_history:
            nh = list(self.noma_history)[-100:]
            cic_rate = float(sum(1 for x in nh if x.get('cic', False)) / len(nh))
        else:
            cic_rate = 0.0
        state.append(cic_rate)

        # 11-12. Channel statistics
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

        # 14. Number of unique files in recent window (diversity metric)
        if self.recent_requests:
            unique_ratio = len(set(self.recent_requests)) / len(self.recent_requests)
        else:
            unique_ratio = 1.0
        state.append(unique_ratio)

        return np.array(state, dtype=np.float32)

    # -------------------------------------------------------------------------
    # ACTION SELECTION (binary: 0=skip, 1=cache)
    # -------------------------------------------------------------------------
    def _select_action(self, state: np.ndarray, file_id: int) -> int:
        """
        Select binary action: 0=skip (don't cache), 1=cache this file.
        If file already cached, always return 0 (skip, it's already there).
        """
        if file_id in self.cache_set:
            return 0  # Already cached, nothing to do

        if self.eval_mode or random.random() >= self.epsilon:
            if self.use_nn:
                with torch.no_grad():
                    t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    q_values = self.q_network(t).cpu().numpy()[0]
                    return int(q_values.argmax())
            else:
                return int(self.q_table[self._discretize_state(state)].argmax())
        return random.randint(0, 1)

    def _discretize_state(self, state: np.ndarray) -> str:
        """Discretize state for Q-table fallback."""
        bins = []
        for v in state[:min(10, len(state))]:
            bins.append('L' if v < 0.33 else 'M' if v < 0.67 else 'H')
        return ''.join(bins)

    # -------------------------------------------------------------------------
    # REWARD (immediate, caching-quality focused)
    # -------------------------------------------------------------------------
    def _compute_reward(
        self, action: int, file_id: int, cache_hit: bool,
        cic_enabled: bool = False, evicted_file: Optional[int] = None
    ) -> float:
        """
        Immediate reward based on caching decision quality.

        Reward hierarchy:
          Cache hit:  +2.0  (file was already cached — good previous decision)
          Cache popular file replacing unpopular: +1.0 to +2.0
          CIC enabled by this cache: +1.5 bonus
          Cache unpopular file: -0.5
          Skip when should have cached (popular file, cache not full): -0.3
          Skip when correct to skip: +0.1
        """
        if cache_hit:
            # Reward for a hit — this file was correctly cached before
            return 2.0

        pop_rank = self._get_file_popularity_rank(file_id)

        if action == 1:  # Agent chose to cache
            # Base reward: how popular is what we cached?
            if pop_rank < 0.1:       # Top 10% most popular
                reward = 1.5
            elif pop_rank < 0.3:     # Top 30%
                reward = 0.8
            elif pop_rank < 0.5:     # Top 50%
                reward = 0.2
            else:                     # Bottom 50%
                reward = -0.5

            # Bonus for eviction quality (did we evict something worse?)
            if evicted_file is not None:
                evicted_rank = self._get_file_popularity_rank(evicted_file)
                if evicted_rank > pop_rank:
                    # Evicted less popular file → good swap
                    reward += 0.5
                else:
                    # Evicted more popular file → bad swap
                    reward -= 0.5

            # CIC bonus
            if cic_enabled:
                reward += 1.5

            return reward

        else:  # action == 0: Agent chose to skip
            if pop_rank < 0.2 and len(self.cache_set) < self.capacity:
                # Skipped a popular file when cache had room — bad
                return -0.5
            elif pop_rank < 0.2:
                # Skipped popular file but cache is full — check if swap would help
                _, worst_file = self._get_min_popularity_slot()
                if worst_file >= 0:
                    worst_rank = self._get_file_popularity_rank(worst_file)
                    if worst_rank > pop_rank + 0.2:
                        # Should have swapped
                        return -0.3
                return 0.0  # Neutral — cache is full, swap wouldn't help much
            else:
                # Correctly skipped unpopular file
                return 0.1

    # -------------------------------------------------------------------------
    # EXECUTE CACHING ACTION
    # -------------------------------------------------------------------------
    def _execute_cache(self, file_id: int) -> Optional[int]:
        """
        Insert file_id into cache. If full, evict least popular file.
        Returns evicted file_id or None.
        """
        if file_id in self.cache_set:
            return None  # Already cached

        evicted_file = None

        if len(self.cache_set) >= self.capacity:
            # Evict least popular cached file
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
            # Find empty slot
            for slot_idx in range(self.capacity):
                if self.cache_slots[slot_idx] == -1:
                    self.cache_slots[slot_idx] = file_id
                    self.file_to_slot[file_id] = slot_idx
                    self.cache_set.add(file_id)
                    break

        return evicted_file

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

        v2 redesign: binary action, immediate reward, compact state.
        """
        self.timestep += 1

        # Track popularity BEFORE decision (so state reflects current request)
        self._update_popularity(item)

        cache_hit     = self.is_hit(item, update_stats=True)
        paired_cached = (
            self.is_hit(paired_file, update_stats=False)
            if paired_file is not None else False
        )

        # Build result dict
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

        # Update tracking
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

        # ── DQN Learning ──
        if not self.eval_mode:
            state = self._get_state_vector(item)
            action = self._select_action(state, item)

            evicted_file = None
            if cache_hit:
                action = 0  # File already cached, no action needed
            elif action == 1:
                evicted_file = self._execute_cache(item)
                # Re-check CIC after caching
                if paired_file is not None and item in self.cache_set:
                    # Caching might have enabled CIC for paired user
                    result['cic_enabled'] = True

            # Compute immediate reward
            reward = self._compute_reward(
                action=action, file_id=item, cache_hit=cache_hit,
                cic_enabled=result['cic_enabled'], evicted_file=evicted_file
            )
            self.cumulative_reward += reward

            # Get next state and push to buffer
            next_state = self._get_state_vector(item)
            self._push_to_buffer({
                'state':      state,
                'action':     action,
                'reward':     reward,
                'next_state': next_state,
                'done':       episode_done,
            })

            # Training step
            self.training_step += 1
            buf_len = len(self.replay_buffer) if self.replay_buffer is not None else 0
            if (self.use_nn
                    and buf_len >= self.warm_up_steps
                    and self.training_step % self.train_freq == 0
                    and buf_len >= self.batch_size):
                self._train_step()

            # Epsilon decay
            if self.epsilon > self.epsilon_end:
                self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)

            if episode_done:
                self.episode_rewards.append(self.cumulative_reward)
                self.cumulative_reward = 0.0

        else:
            # Eval mode: use learned policy but don't train
            if not cache_hit:
                state = self._get_state_vector(item)
                action = self._select_action(state, item)
                if action == 1:
                    self._execute_cache(item)

        return result

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

    def _push_to_buffer(self, exp: Dict):
        if self.use_nn and self.replay_buffer is not None:
            if self.use_prioritized:
                self.replay_buffer.add(exp)
            else:
                self.replay_buffer.append(exp)
        elif not self.use_nn:
            self._update_q_table(exp)

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
        if items is None:
            # Use request counts to determine popular files
            top = np.argsort(-self.request_counts)[:self.capacity]
        else:
            top = list(items)[:self.capacity]

        self.cache_slots  = [-1] * self.capacity
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

    def get_contents(self) -> Set[int]:
        return set(self.cache_set)

    def clear(self):
        """
        Reset cache state (but preserve learned model weights).
        """
        self.cache_slots  = [-1] * self.capacity
        self.cache_set.clear()
        self.file_to_slot.clear()
        self.cumulative_reward = 0.0
        self.reset_stats()

    def reset_popularity(self):
        """Reset popularity tracking (for new episode)."""
        self.request_counts = np.zeros(self.num_files, dtype=np.float64)
        self.recent_requests.clear()

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
                'q_network':       self.q_network.state_dict(),
                'target_network':  self.target_network.state_dict(),
                'optimizer':       self.optimizer.state_dict(),
                'training_step':   self.training_step,
                'epsilon':         self.epsilon,
                'request_counts':  self.request_counts,
                'cache_slots':     self.cache_slots,
                'cache_set':       list(self.cache_set),
                'file_to_slot':    self.file_to_slot,
            }
            if self.use_prioritized:
                d['per_beta']      = self.replay_buffer.get_beta()
                d['per_frame_idx'] = self.replay_buffer.frame_idx
            torch.save(d, filepath)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'q_table':         dict(self.q_table),
                    'training_step':   self.training_step,
                    'epsilon':         self.epsilon,
                    'request_counts':  self.request_counts,
                    'cache_slots':     self.cache_slots,
                    'cache_set':       list(self.cache_set),
                    'file_to_slot':    self.file_to_slot,
                }, f)
        print(f'Model saved to {filepath}')

    def load_model(self, filepath: str):
        if self.use_nn:
            ckpt = torch.load(filepath, map_location=self.device,
                              weights_only=False)
            self.q_network.load_state_dict(ckpt['q_network'])
            self.target_network.load_state_dict(ckpt['target_network'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.training_step  = ckpt['training_step']
            self.epsilon        = ckpt['epsilon']
            self.request_counts = ckpt.get('request_counts', self.request_counts)
            self.cache_slots    = ckpt.get('cache_slots', self.cache_slots)
            self.cache_set      = set(ckpt.get('cache_set', []))
            self.file_to_slot   = ckpt.get('file_to_slot', {})
            if self.use_prioritized and 'per_beta' in ckpt:
                self.replay_buffer.beta      = ckpt['per_beta']
                self.replay_buffer.frame_idx = ckpt.get('per_frame_idx', 0)
        else:
            with open(filepath, 'rb') as f:
                ckpt = pickle.load(f)
            self.q_table        = defaultdict(
                lambda: np.zeros(self.action_dim), ckpt['q_table'])
            self.training_step  = ckpt['training_step']
            self.epsilon        = ckpt['epsilon']
            self.request_counts = ckpt.get('request_counts', self.request_counts)
            self.cache_slots    = ckpt.get('cache_slots', self.cache_slots)
            self.cache_set      = set(ckpt.get('cache_set', []))
            self.file_to_slot   = ckpt.get('file_to_slot', {})
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
            'beta':                self.replay_buffer.get_beta() if (
                                       self.use_prioritized and self.replay_buffer
                                   ) else 0.0,
            'action_dim':          self.action_dim,
            'state_dim':           self.state_dim,
        }
        return {**base, **dqn}


# Alias for compatibility
StableDQNCache = DQNCache
