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

Bug Fixes Applied:
- BUG #1 (CRITICAL): Fixed popularity EMA double-decay error
- BUG #2 (CRITICAL): Added beta annealing to prioritized replay
- BUG #3 (MODERATE): Smart sampling strategy (with/without replacement)
- BUG #4 (MINOR): Proper soft target update every training step
- BUG #5 (MINOR): Empty slot LRU representation fixed
- ENHANCEMENT #6: Added warm-up period before training

2026 Fixes:
- BUG-1 (CRITICAL): Replaced last_state/last_action pattern with
  pending_transitions dict for correct deferred reward assignment.
- BUG-3 (CRITICAL): Fixed EMA popularity to decay ALL files every step.
- BUG-4 (MODERATE): O(N²) _get_state_vector fixed to O(N) by
  precomputing occupied_mask, max_lru, max_lfu before slot loops.
- BUG-9 (MODERATE): populate() now clears pending_transitions to
  prevent stale state vectors poisoning the replay buffer.
- BUG-12 (MODERATE): save_model/load_model now persists
  pending_transitions so in-flight rewards survive checkpointing.
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
    """
    
    def __init__(self, state_dim: int, action_dim: int, 
                 hidden_dims: List[int] = [128, 64]):
        super(DuelingDQN, self).__init__()
        
        self.feature_layers = nn.ModuleList()
        prev_dim = state_dim
        for hdim in hidden_dims:
            self.feature_layers.append(nn.Linear(prev_dim, hdim))
            prev_dim = hdim
        
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1] // 2, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1] // 2, action_dim)
        )
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = state
        for layer in self.feature_layers:
            x = F.relu(layer(x))
        value     = self.value_stream(x)       # (batch, 1)
        advantage = self.advantage_stream(x)   # (batch, action_dim)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


# ====================================================================================
# PRIORITIZED EXPERIENCE REPLAY (Schaul et al., ICLR 2016)
# ============================================================================

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay with beta annealing.
    Schaul et al., "Prioritized Experience Replay", ICLR 2016.
    """
    
    def __init__(self, capacity: int, alpha: float = 0.6,
                 beta_start: float = 0.4, beta_end: float = 1.0,
                 beta_frames: int = 100000):
        self.capacity    = capacity
        self.alpha       = alpha
        self.beta        = beta_start
        self.beta_start  = beta_start
        self.beta_end    = beta_end
        self.beta_frames = beta_frames
        self.frame_idx   = 0
        self.buffer      = deque(maxlen=capacity)
        self.priorities  = deque(maxlen=capacity)
        self.max_priority = 1.0
    
    def add(self, experience: Dict):
        self.buffer.append(experience)
        self.priorities.append(self.max_priority)
    
    def sample(self, batch_size: int) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
        if len(self.buffer) < batch_size:
            return None, None, None
        
        # Anneal beta: β₀ → 1.0 (Schaul et al., 2016)
        self.frame_idx += 1
        self.beta = min(
            self.beta_end,
            self.beta_start + (self.beta_end - self.beta_start) * (self.frame_idx / self.beta_frames)
        )
        
        priorities = np.array(self.priorities, dtype=np.float64)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Use replacement when buffer is small to reduce inter-batch correlation
        use_replacement = len(self.buffer) < 3 * batch_size
        try:
            indices = np.random.choice(len(self.buffer), batch_size, p=probs,
                                       replace=use_replacement)
        except ValueError:
            indices = np.random.choice(len(self.buffer), batch_size,
                                       replace=use_replacement)
        
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        return [self.buffer[i] for i in indices], weights, indices
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        for idx, error in zip(indices, td_errors):
            priority = float(abs(error) + 1e-6)
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = priority
                self.max_priority = max(self.max_priority, priority)
    
    def get_beta(self) -> float:
        return self.beta
    
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
        LRU counters + LFU counters + requested file popularity +
        cache occupancy + channel quality (mean, std) +
        NOMA performance (CIC rate, success rate)
        Total: 2*capacity + 6 dimensions
    
    Action a_t:
        Which cache slot to evict (0 … capacity-1)
    
    Reward r_t  [DEFERRED — delivered when evicted file is next requested]:
        +10: cache hit on the evicted file
        +2:  miss + CIC enabled
        -1:  miss + NOMA success
        -5:  miss + NOMA failure
        -10: outage
    
    **Credit Assignment:**
    OLD (broken): reward at step t stored with action at step t-1.
    NEW (correct): reward stored only when evicted file is next requested.
    See pending_transitions for implementation.
    """
    
    def __init__(
        self,
        capacity: int,
        num_files: int,
        num_users: int,
        learning_rate: float = 0.0001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay_steps: int = 200000,
        use_neural_network: bool = True,
        hidden_dims: List[int] = [128, 128],
        batch_size: int = 64,
        replay_buffer_size: int = 50000,
        train_freq: int = 4,
        warm_up_steps: Optional[int] = None,
        use_prioritized_replay: bool = True,
        priority_alpha: float = 0.6,
        priority_beta_start: float = 0.4,
        priority_beta_end: float = 1.0,
        priority_beta_frames: int = 100000,
        gradient_clip: float = 10.0,
        tau: float = 0.005,
        enable_noma_awareness: bool = True,
        seed: int = 2025
    ):
        super().__init__(capacity, enable_noma_awareness)
        
        self.num_files      = num_files
        self.num_users      = num_users
        self.lr             = learning_rate
        self.gamma          = gamma
        self.batch_size     = batch_size
        self.train_freq     = train_freq
        self.gradient_clip  = gradient_clip
        self.tau            = tau
        
        self.warm_up_steps = max(10 * batch_size, 1000) if warm_up_steps is None else warm_up_steps
        
        self._set_seeds(seed)
        
        self.epsilon        = epsilon_start
        self.epsilon_start  = epsilon_start
        self.epsilon_end    = epsilon_end
        self.epsilon_decay  = (epsilon_start - epsilon_end) / max(1, epsilon_decay_steps)
        self.eval_mode      = False
        self._eval_epsilon  = 0.0
        
        self.cache_slots  = [-1] * capacity
        self.file_to_slot = {}
        
        self.lru_counters = np.zeros(capacity, dtype=np.int32)
        self.lfu_counters = np.zeros(capacity, dtype=np.int32)
        self.timestep     = 0
        
        # EMA popularity: correct form decays ALL files each step.
        self.popularity       = np.ones(num_files, dtype=np.float32) / num_files
        self.popularity_decay = 0.9
        
        self.channel_history = deque(maxlen=500)
        self.noma_history    = deque(maxlen=500)
        self.cic_count       = 0
        self.sic_count       = 0
        
        self.use_nn       = use_neural_network and TORCH_AVAILABLE
        self.training_step = 0
        self.action_dim   = capacity
        
        if self.use_nn:
            self.state_dim = 2 * capacity + 6
            self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            self.q_network     = DuelingDQN(self.state_dim, self.action_dim, hidden_dims).to(self.device)
            self.target_network = DuelingDQN(self.state_dim, self.action_dim, hidden_dims).to(self.device)
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.target_network.eval()
            
            self.optimizer = optim.Adam(self.q_network.parameters(),
                                        lr=self.lr, weight_decay=1e-5)
            
            if use_prioritized_replay:
                self.replay_buffer = PrioritizedReplayBuffer(
                    capacity=replay_buffer_size, alpha=priority_alpha,
                    beta_start=priority_beta_start, beta_end=priority_beta_end,
                    beta_frames=priority_beta_frames)
                self.use_prioritized = True
            else:
                self.replay_buffer   = deque(maxlen=replay_buffer_size)
                self.use_prioritized = False
        else:
            self.q_table         = defaultdict(lambda: np.zeros(self.action_dim))
            self.replay_buffer   = None
            self.use_prioritized = False
        
        self.episode_rewards  = []
        self.losses           = []
        self.cumulative_reward = 0.0
        
        # Deferred credit assignment (BUG-1 fix).
        # Key:   evicted_file_id
        # Value: {'state': state_before, 'action': slot, 'next_state': state_after}
        # Reward is appended when the evicted file is next requested.
        self.pending_transitions: Dict[int, Dict] = {}
        
        print(f"✅ DQNCache initialized")
        print(f"   Mode        : {'Neural Network (DQN)' if self.use_nn else 'Q-table'}")
        print(f"   State dim   : {self.state_dim if self.use_nn else 'N/A'}")
        print(f"   Action dim  : {self.action_dim}")
        print(f"   Device      : {self.device if self.use_nn else 'CPU'}")
        print(f"   NOMA-aware  : {self.enable_noma_awareness}")
        print(f"   Warm-up     : {self.warm_up_steps} steps")
        print(f"   Credit      : deferred via pending_transitions ✅")
        print(f"   EMA pop.    : full-vector decay ✅")
        if self.use_prioritized:
            print(f"   PER beta    : {priority_beta_start:.2f} → {priority_beta_end:.2f} over {priority_beta_frames} frames")
    
    def _set_seeds(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        if TORCH_AVAILABLE:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
    
    # ========================================================================
    # STATE REPRESENTATION  (BUG-4 FIX: O(N²) → O(N))
    # ========================================================================
    
    def _get_state_vector(self, requested_file: int) -> np.ndarray:
        """
        Build the state vector for the DQN.
        
        Dimensions: LRU(capacity) + LFU(capacity) + 6 global features
        Total: 2*capacity + 6
        
        BUG-4 fix: occupied_mask, max_lru, max_lfu are now computed
        ONCE before the loops, not re-derived inside every iteration.
        This reduces complexity from O(capacity²) to O(capacity).
        """
        state = []
        
        # ---- Precompute once (BUG-4 fix) -----------------------------------
        occupied_mask = np.array([s != -1 for s in self.cache_slots], dtype=bool)
        has_occupied  = occupied_mask.any()
        
        if has_occupied:
            occ_lru = self.lru_counters[occupied_mask]
            occ_lfu = self.lfu_counters[occupied_mask]
            max_lru = max(int(occ_lru.max()), 1)   # avoid div-by-zero
            max_lfu = max(int(occ_lfu.max()), 1)
        else:
            max_lru = 1
            max_lfu = 1
        # --------------------------------------------------------------------
        
        # 1. LRU counters
        #    Empty slots  → -1.0   (special sentinel: slot is free)
        #    Occupied slot → normalised [0, 1]  (0 = most recently used)
        for i in range(self.capacity):
            if not occupied_mask[i]:
                state.append(-1.0)
            else:
                state.append(float(self.lru_counters[i] / max_lru))
        
        # 2. LFU counters
        #    Empty slots  → 0.0
        #    Occupied slot → normalised [0, 1]
        for i in range(self.capacity):
            if not occupied_mask[i]:
                state.append(0.0)
            else:
                state.append(float(self.lfu_counters[i] / max_lfu))
        
        # 3. Requested file popularity
        state.append(float(self.popularity[requested_file]))
        
        # 4. Cache occupancy ratio
        state.append(float(occupied_mask.sum() / self.capacity))
        
        # 5-6. Channel quality (mean, std over last 100 observations)
        if len(self.channel_history) > 0:
            ch = list(self.channel_history)[-100:]
            state.append(float(np.mean(ch)))
            state.append(float(np.std(ch)))
        else:
            state.extend([0.5, 0.1])
        
        # 7-8. NOMA performance (CIC rate, success rate over last 100)
        if self.enable_noma_awareness and len(self.noma_history) > 0:
            nh = list(self.noma_history)[-100:]
            cic_rate     = sum(1 for x in nh if x.get('cic',     False)) / len(nh)
            success_rate = sum(1 for x in nh if x.get('success', False)) / len(nh)
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
        Returns slot index (0 … capacity-1) or -1 if already cached.
        """
        if file_id in self.file_to_slot:
            return -1
        
        empty_slots = [i for i, f in enumerate(self.cache_slots) if f == -1]
        if empty_slots:
            return empty_slots[0]  # Fill first available slot (no eviction)
        
        # Cache full: epsilon-greedy eviction choice
        if self.eval_mode or random.random() >= self.epsilon:
            if self.use_nn:
                with torch.no_grad():
                    t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    return int(self.q_network(t).cpu().numpy()[0].argmax())
            else:
                return int(self.q_table[self._discretize_state(state)].argmax())
        else:
            return random.randint(0, self.capacity - 1)
    
    def _discretize_state(self, state: np.ndarray) -> str:
        bins = []
        for val in state[:min(10, len(state))]:
            if   val < 0.0:  bins.append('E')
            elif val < 0.33: bins.append('L')
            elif val < 0.67: bins.append('M')
            else:            bins.append('H')
        return ''.join(bins)
    
    # ========================================================================
    # REWARD FUNCTION (NOMA-Aware)
    # ========================================================================
    
    def _compute_reward(
        self, cache_hit: bool, cic_enabled: bool = False,
        noma_success: bool = True, outage: bool = False,
        ber: Optional[float] = None
    ) -> float:
        """
        NOMA-aware reward delivered when the evicted file is next requested.
        
        Hierarchy (strict ordering prevents perverse incentives):
            +10  cache_hit
            + 2  miss + CIC enabled
            - 1  miss + NOMA success
            - 5  miss + NOMA failure
            -10  outage
        Optional BER modifier: ±1–2.
        """
        if cache_hit:      return 10.0
        if outage:         return -10.0
        if not noma_success: return -5.0
        
        reward = 2.0 if cic_enabled else -1.0
        
        if ber is not None:
            if   ber < 1e-4: reward += 1.0
            elif ber > 1e-2: reward -= 2.0
        
        return reward
    
    # ========================================================================
    # MAIN REQUEST INTERFACE
    # ========================================================================
    
    def request(
        self, item: int,
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
        Handle one file request with NOMA-aware DQN learning.
        Returns dict with hit/CIC/benefit flags.
        """
        self.timestep += 1
        
        # 1. Single is_hit call — no double-counting
        cache_hit = self.is_hit(item, update_stats=True)
        
        # 2. NOMA result dict
        paired_cached = self.is_hit(paired_file, update_stats=False) if paired_file is not None else False
        result = {
            'hit': cache_hit,
            'cic_enabled': False,
            'paired_user_cached': paired_cached,
            'weak_user_benefit': False,
            'strong_user_benefit': False,
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
        
        # 3. NOMA history
        if channel_gain is not None:
            self.channel_history.append(float(channel_gain))
        if self.enable_noma_awareness:
            self.noma_history.append({'cic': result['cic_enabled'],
                                      'success': noma_success,
                                      'sinr_weak': sinr_weak,
                                      'sinr_strong': sinr_strong})
            if result['cic_enabled']:          self.cic_count += 1
            if result['strong_user_benefit']:  self.sic_count += 1
        
        # 4. DQN learning
        if not self.eval_mode:
            self._learn_from_request(
                file_id=item, cache_hit=cache_hit,
                cic_enabled=result['cic_enabled'],
                noma_success=noma_success, outage=outage,
                ber=ber, episode_done=episode_done)
        
        # 5. Update LRU/LFU counters AFTER learning
        self._update_counters(item, cache_hit)
        
        # 6. Correct EMA popularity (BUG-3 fix: decay ALL files)
        self.popularity *= self.popularity_decay
        self.popularity[item] += (1.0 - self.popularity_decay)
        self.popularity /= self.popularity.sum()
        
        return result
    
    # ========================================================================
    # DEFERRED CREDIT ASSIGNMENT
    # ========================================================================
    
    def _learn_from_request(
        self, file_id: int, cache_hit: bool, cic_enabled: bool,
        noma_success: bool, outage: bool, ber: Optional[float],
        episode_done: bool
    ):
        """
        DQN update using deferred pending_transitions.
        
        Steps:
          A. Complete pending transition for file_id (if previously evicted)
          B. Make eviction decision on miss; park new pending transition
          C. Train network (after warm-up)
        """
        
        # ---- A: Complete pending transition --------------------------------
        if file_id in self.pending_transitions:
            pending = self.pending_transitions.pop(file_id)
            reward  = self._compute_reward(
                cache_hit=cache_hit, cic_enabled=cic_enabled,
                noma_success=noma_success, outage=outage, ber=ber)
            self._push_to_buffer({
                'state':      pending['state'],
                'action':     pending['action'],
                'reward':     reward,
                'next_state': self._get_state_vector(file_id),
                'done':       episode_done
            })
            self.cumulative_reward += reward
        
        # ---- B: Eviction on cache miss -------------------------------------
        if not cache_hit:
            empty_slots = [i for i, f in enumerate(self.cache_slots) if f == -1]
            cache_full  = len(empty_slots) == 0
            state_before = self._get_state_vector(file_id)
            action       = self._select_action(state_before, file_id)
            
            if action >= 0:
                if cache_full:
                    evicted_file = self.cache_slots[action]
                    self._execute_action(action, file_id)
                    state_after = self._get_state_vector(file_id)
                    
                    if evicted_file != -1:
                        # Flush any stale pending for the same evicted file
                        if evicted_file in self.pending_transitions:
                            old = self.pending_transitions.pop(evicted_file)
                            self._push_to_buffer({
                                'state':      old['state'],
                                'action':     old['action'],
                                'reward':     -1.0,
                                'next_state': state_before,
                                'done':       False
                            })
                        self.pending_transitions[evicted_file] = {
                            'state':      state_before,
                            'action':     action,
                            'next_state': state_after
                        }
                else:
                    # Empty slot: immediate experience (no deferred reward)
                    self._execute_action(action, file_id)
                    state_after = self._get_state_vector(file_id)
                    imm_reward  = self._compute_reward(
                        cache_hit=False, cic_enabled=cic_enabled,
                        noma_success=noma_success, outage=outage, ber=ber)
                    self._push_to_buffer({
                        'state':      state_before,
                        'action':     action,
                        'reward':     imm_reward,
                        'next_state': state_after,
                        'done':       episode_done
                    })
                    self.cumulative_reward += imm_reward
        
        # ---- C: Train -------------------------------------------------------
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
            # Flush unresolved pending transitions with neutral reward
            for evicted_file, pending in list(self.pending_transitions.items()):
                self._push_to_buffer({
                    'state':      pending['state'],
                    'action':     pending['action'],
                    'reward':     0.0,
                    'next_state': self._get_state_vector(file_id),
                    'done':       True
                })
            self.pending_transitions.clear()
    
    def _push_to_buffer(self, experience: Dict):
        if self.use_nn and self.replay_buffer is not None:
            if self.use_prioritized:
                self.replay_buffer.add(experience)
            else:
                self.replay_buffer.append(experience)
        elif not self.use_nn:
            self._update_q_table(experience)
    
    def _execute_action(self, action: int, file_id: int):
        """Place file_id into slot `action`, evicting whatever was there."""
        if action < 0 or action >= self.capacity:
            return
        old_file = self.cache_slots[action]
        if old_file != -1 and old_file in self.file_to_slot:
            del self.file_to_slot[old_file]
        self.cache_slots[action]  = file_id
        self.file_to_slot[file_id] = action
        self.lru_counters[action]  = 0
        self.lfu_counters[action]  = 1
    
    def _update_counters(self, file_id: int, cache_hit: bool):
        """Age all LRU counters; refresh hit slot."""
        for i in range(self.capacity):
            if self.cache_slots[i] != -1:
                self.lru_counters[i] += 1
        if cache_hit and file_id in self.file_to_slot:
            slot = self.file_to_slot[file_id]
            self.lru_counters[slot] = 0
            self.lfu_counters[slot] += 1
    
    # ========================================================================
    # TRAINING (Double DQN + Prioritized Replay)
    # ========================================================================
    
    def _train_step(self):
        """One Double-DQN gradient step with soft target update."""
        if not self.use_nn or len(self.replay_buffer) < self.batch_size:
            return
        
        if self.use_prioritized:
            experiences, weights, indices = self.replay_buffer.sample(self.batch_size)
            if experiences is None:
                return
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            experiences = random.sample(list(self.replay_buffer), self.batch_size)
            weights     = torch.ones(self.batch_size).to(self.device)
            indices     = None
        
        states      = torch.from_numpy(np.array([e['state']      for e in experiences], dtype=np.float32)).to(self.device)
        actions     = torch.from_numpy(np.array([e['action']     for e in experiences], dtype=np.int64)).to(self.device)
        rewards     = torch.from_numpy(np.array([e['reward']     for e in experiences], dtype=np.float32)).to(self.device)
        next_states = torch.from_numpy(np.array([e['next_state'] for e in experiences], dtype=np.float32)).to(self.device)
        dones       = torch.from_numpy(np.array([e['done']       for e in experiences], dtype=np.float32)).to(self.device)
        
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q       = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q     = rewards + (1.0 - dones) * self.gamma * next_q
        
        td_errors = target_q - current_q
        loss = (weights * F.smooth_l1_loss(current_q, target_q, reduction='none')).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.gradient_clip)
        self.optimizer.step()
        self._soft_update_target()
        
        if self.use_prioritized and indices is not None:
            self.replay_buffer.update_priorities(indices, td_errors.detach().cpu().numpy())
        
        self.losses.append(float(loss.item()))
    
    def _soft_update_target(self):
        """θ_target ← τ*θ_policy + (1-τ)*θ_target"""
        for tp, p in zip(self.target_network.parameters(), self.q_network.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)
    
    def _update_q_table(self, experience: Dict):
        """Q-table update for fallback mode."""
        sk = self._discretize_state(experience['state'])
        nk = self._discretize_state(experience['next_state'])
        a, r, done = experience['action'], experience['reward'], experience['done']
        cq = self.q_table[sk][a]
        nq = 0.0 if done else float(np.max(self.q_table[nk]))
        self.q_table[sk][a] += self.lr * (r + self.gamma * nq - cq)
    
    # ========================================================================
    # CACHE INTERFACE
    # ========================================================================
    
    def populate(self, items: Optional[Iterable[int]] = None):
        """
        Pre-load cache with the most popular files.
        
        BUG-9 fix: pending_transitions is cleared first so that any
        in-flight eviction transitions from a previous episode don't
        reference file IDs that no longer match the new cache layout.
        """
        # Flush pending before changing cache layout (BUG-9 fix)
        self.pending_transitions.clear()
        
        top_files = (np.argsort(-self.popularity)[:self.capacity]
                     if items is None else list(items)[:self.capacity])
        
        self.cache_slots = [-1] * self.capacity
        self.file_to_slot.clear()
        
        for slot, file_id in enumerate(top_files):
            self.cache_slots[slot]    = int(file_id)
            self.file_to_slot[int(file_id)] = slot
            self.lfu_counters[slot]   = 1
            self.lru_counters[slot]   = 0
    
    def is_hit(self, item: int, update_stats: bool = True) -> bool:
        hit = int(item) in self.file_to_slot
        if update_stats:
            self._record_hit() if hit else self._record_miss()
        return hit
    
    def get_contents(self) -> Set[int]:
        return set(f for f in self.cache_slots if f != -1)
    
    def clear(self):
        self.cache_slots  = [-1] * self.capacity
        self.file_to_slot.clear()
        self.lru_counters = np.zeros(self.capacity, dtype=np.int32)
        self.lfu_counters = np.zeros(self.capacity, dtype=np.int32)
        self.pending_transitions.clear()
        self.reset_stats()
    
    def set_eval_mode(self, eval_mode: bool = True):
        """Toggle eval mode; saves/restores epsilon to prevent decay corruption."""
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
    
    # ========================================================================
    # MODEL PERSISTENCE  (BUG-12 FIX: persist pending_transitions)
    # ========================================================================
    
    def save_model(self, filepath: str):
        """
        Save model checkpoint.
        
        BUG-12 fix: pending_transitions is now included in the checkpoint.
        Previously, any in-flight eviction rewards were silently dropped
        when saving mid-episode, making those experiences unrecoverable.
        """
        if self.use_nn:
            save_dict = {
                'q_network':            self.q_network.state_dict(),
                'target_network':       self.target_network.state_dict(),
                'optimizer':            self.optimizer.state_dict(),
                'training_step':        self.training_step,
                'epsilon':              self.epsilon,
                'popularity':           self.popularity,
                'cache_slots':          self.cache_slots,
                'file_to_slot':         self.file_to_slot,
                'lru_counters':         self.lru_counters,
                'lfu_counters':         self.lfu_counters,
                # BUG-12 fix: persist in-flight eviction transitions
                'pending_transitions':  self.pending_transitions,
            }
            if self.use_prioritized:
                save_dict['per_beta']      = self.replay_buffer.get_beta()
                save_dict['per_frame_idx'] = self.replay_buffer.frame_idx
            torch.save(save_dict, filepath)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'q_table':             dict(self.q_table),
                    'training_step':       self.training_step,
                    'epsilon':             self.epsilon,
                    'popularity':          self.popularity,
                    'cache_slots':         self.cache_slots,
                    'file_to_slot':        self.file_to_slot,
                    # BUG-12 fix
                    'pending_transitions': self.pending_transitions,
                }, f)
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load model checkpoint.
        
        BUG-12 fix: restores pending_transitions from checkpoint so that
        in-flight eviction rewards survive a save/load cycle.
        """
        if self.use_nn:
            ckpt = torch.load(filepath, map_location=self.device, weights_only=False)
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
            # BUG-12 fix: restore pending transitions (empty dict if absent
            # for backward compatibility with older checkpoints)
            self.pending_transitions = ckpt.get('pending_transitions', {})
            if self.use_prioritized and 'per_beta' in ckpt:
                self.replay_buffer.beta      = ckpt['per_beta']
                self.replay_buffer.frame_idx = ckpt.get('per_frame_idx', 0)
        else:
            with open(filepath, 'rb') as f:
                ckpt = pickle.load(f)
            self.q_table             = defaultdict(lambda: np.zeros(self.action_dim), ckpt['q_table'])
            self.training_step       = ckpt['training_step']
            self.epsilon             = ckpt['epsilon']
            self.popularity          = ckpt['popularity']
            self.cache_slots         = ckpt['cache_slots']
            self.file_to_slot        = ckpt['file_to_slot']
            # BUG-12 fix
            self.pending_transitions = ckpt.get('pending_transitions', {})
        print(f"✅ Model loaded from {filepath}")
    
    # ========================================================================
    # STATISTICS
    # ========================================================================
    
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
            'beta':                self.replay_buffer.get_beta() if (self.use_prioritized and self.replay_buffer) else 0.0,
        }
        return {**base, **dqn}


# Alias for compatibility
StableDQNCache = DQNCache
