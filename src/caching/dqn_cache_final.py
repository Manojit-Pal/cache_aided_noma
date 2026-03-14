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
  Each eviction's reward is now delivered ONLY when the evicted file
  is next requested, creating valid (s, a, r, s') tuples.
- BUG-3 (CRITICAL): Fixed EMA popularity update to correctly decay ALL
  files at every timestep, not just the requested one. Non-requested
  files were never explicitly decayed — only squeezed by normalization —
  causing stale-popularity bias in the DQN state representation.
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
        value = self.value_stream(x)          # (batch, 1)
        advantage = self.advantage_stream(x)  # (batch, action_dim)
        
        # Combine: subtract mean advantage for stability
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values


# ====================================================================================
# PRIORITIZED EXPERIENCE REPLAY (Schaul et al., ICLR 2016)
# ============================================================================

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer with Beta Annealing.
    
    Samples transitions with probability proportional to TD-error,
    allowing agent to learn more from surprising experiences.
    
    Based on: Schaul et al., "Prioritized Experience Replay", ICLR 2016
    
    Key insights:
    1. Beta should be annealed from initial value to 1.0 over training
    2. Sampling strategy: use replacement when buffer is small to avoid correlation
    """
    
    def __init__(
        self, 
        capacity: int, 
        alpha: float = 0.6, 
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_frames: int = 100000
    ):
        """
        Args:
            capacity: Maximum buffer size
            alpha: Priority exponent (0=uniform, 1=full prioritization)
            beta_start: Initial importance sampling exponent
            beta_end: Final importance sampling exponent (1.0 = unbiased)
            beta_frames: Number of frames to anneal beta from start to end
        """
        self.capacity = capacity
        self.alpha = alpha
        
        # Beta annealing schedule (Schaul et al., 2016)
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_frames = beta_frames
        self.frame_idx = 0
        
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0
    
    def add(self, experience: Dict):
        """Add experience with max priority (ensures at least one sample)."""
        self.buffer.append(experience)
        self.priorities.append(self.max_priority)
    
    def sample(self, batch_size: int) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
        """
        Sample batch with prioritized sampling and smart replacement strategy.
        
        Returns:
            experiences: List of experience dicts
            weights: Importance sampling weights
            indices: Sampled indices (for priority updates)
        """
        if len(self.buffer) < batch_size:
            return None, None, None
        
        # Anneal beta from beta_start to beta_end
        # Research: "We linearly anneal β from its initial value β₀ to 1"
        # - Schaul et al., "Prioritized Experience Replay", ICLR 2016
        self.frame_idx += 1
        self.beta = min(
            self.beta_end,
            self.beta_start + (self.beta_end - self.beta_start) * (self.frame_idx / self.beta_frames)
        )
        
        # Compute sampling probabilities
        priorities = np.array(self.priorities, dtype=np.float64)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Smart sampling strategy:
        # Use with-replacement when buffer is relatively small to avoid
        # extreme correlation between consecutive batches.
        use_replacement = len(self.buffer) < 3 * batch_size
        
        try:
            indices = np.random.choice(
                len(self.buffer), 
                batch_size, 
                p=probs, 
                replace=use_replacement
            )
        except ValueError:
            indices = np.random.choice(len(self.buffer), batch_size, replace=use_replacement)
        
        # Compute importance sampling weights with current beta
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
    
    def get_beta(self) -> float:
        """Get current beta value (for logging/debugging)."""
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
        - LRU counters: timesteps since last access for each cached file
        - LFU counters: access frequency for each cached file
        - Requested file popularity
        - Channel quality metrics (mean, std)
        - NOMA performance (CIC rate, success rate)
        - Cache occupancy
    
    Action a_t:
        - Which cache slot to evict (0 to capacity-1)
        - Based on research: slot-based actions simplify learning
    
    Reward r_t  [delivered DEFERRED when evicted file is next requested]:
        +10: Cache hit on the evicted file  (eviction was costly — file was needed)
        -1:  Evicted file miss + NOMA success  (acceptable eviction)
        -5:  Evicted file miss + NOMA failure  (poor eviction)
        -10: Outage during evicted file retrieval  (worst eviction)
        +2:  Evicted file miss + CIC enabled  (side-channel benefit)

    **Credit Assignment (2026 Fix):**
    
    OLD (broken): reward for request t was stored with action from request t-1.
    NEW (correct): reward delivered ONLY when evicted file is next requested.
    
    **Research-Based Design Choices:**
    
    1. State = LRU + LFU heuristics (RLCaR paper)
    2. Action = Slot eviction (simplifies action space)
    3. Reward = Cache hit ratio optimization (standard)
    4. Network = Dueling DQN (better value estimation)
    5. Replay = Prioritized (learn from important transitions)
    6. Target Update = Soft update every training step (DDPG-style)
    7. Warm-up = Wait for 10x batch_size before training
    
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
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay_steps: int = 200000,
        
        # Network architecture
        use_neural_network: bool = True,
        hidden_dims: List[int] = [128, 128],
        
        # Training parameters
        batch_size: int = 64,
        replay_buffer_size: int = 50000,
        train_freq: int = 4,
        warm_up_steps: Optional[int] = None,
        
        # Prioritized replay
        use_prioritized_replay: bool = True,
        priority_alpha: float = 0.6,
        priority_beta_start: float = 0.4,
        priority_beta_end: float = 1.0,
        priority_beta_frames: int = 100000,
        
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
        
        # Warm-up period before training
        if warm_up_steps is None:
            self.warm_up_steps = max(10 * batch_size, 1000)
        else:
            self.warm_up_steps = warm_up_steps
        
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
        self.file_to_slot = {}             # Reverse mapping: file_id -> slot
        
        # LRU/LFU counters (for state representation)
        self.lru_counters = np.zeros(capacity, dtype=np.int32)  # Steps since access
        self.lfu_counters = np.zeros(capacity, dtype=np.int32)  # Access frequency
        self.timestep = 0
        
        # Popularity tracking (EMA).
        # CORRECT form: p *= decay each step; p[item] += (1-decay) on request.
        # See request() for the full update.
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
        
        # Action space: evict slot i (0 to capacity-1)
        self.action_dim = capacity
        
        if self.use_nn:
            # State dimension: LRU(capacity) + LFU(capacity) + 6 global features
            self.state_dim = 2 * capacity + 6
            
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
            
            # Experience replay with beta annealing
            if use_prioritized_replay:
                self.replay_buffer = PrioritizedReplayBuffer(
                    capacity=replay_buffer_size,
                    alpha=priority_alpha,
                    beta_start=priority_beta_start,
                    beta_end=priority_beta_end,
                    beta_frames=priority_beta_frames
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
        
        # =====================================================================
        # BUG-1 FIX (2026): Deferred credit assignment via pending_transitions
        # =====================================================================
        # Key: evicted_file_id (int)
        # Value: {'state': state_before_eviction, 'action': slot_index,
        #         'next_state': state_after_eviction}
        # Reward is added when evicted file is next requested.
        self.pending_transitions: Dict[int, Dict] = {}
        
        print(f"✅ DQNCache initialized")
        print(f"   Mode: {'Neural Network (DQN)' if self.use_nn else 'Q-table'}")
        print(f"   State dim: {self.state_dim if self.use_nn else 'N/A'}")
        print(f"   Action dim: {self.action_dim}")
        print(f"   Device: {self.device if self.use_nn else 'CPU'}")
        print(f"   NOMA-aware: {self.enable_noma_awareness}")
        print(f"   Warm-up steps: {self.warm_up_steps}")
        print(f"   Credit assignment: deferred via pending_transitions ✅")
        print(f"   EMA popularity: full-vector decay ✅")
        if self.use_prioritized:
            print(f"   PER beta: {priority_beta_start:.2f} → {priority_beta_end:.2f} over {priority_beta_frames} frames")
    
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
        
        # Empty slots (cache_slots[i] = -1) use -1.0 as a special marker
        # to distinguish them from recently-accessed slots (LRU=0).
        occupied_mask = np.array([slot != -1 for slot in self.cache_slots])
        
        # 1. LRU counters with empty slot handling
        for i in range(self.capacity):
            if self.cache_slots[i] == -1:
                state.append(-1.0)  # Empty slot marker
            else:
                if occupied_mask.any():
                    occupied_lru = self.lru_counters[occupied_mask]
                    max_lru = max(occupied_lru.max(), 1)
                    state.append(float(self.lru_counters[i] / max_lru))
                else:
                    state.append(0.0)
        
        # 2. LFU counters (empty slots report 0)
        for i in range(self.capacity):
            if self.cache_slots[i] == -1:
                state.append(0.0)
            else:
                if occupied_mask.any():
                    occupied_lfu = self.lfu_counters[occupied_mask]
                    max_lfu = max(occupied_lfu.max(), 1)
                    state.append(float(self.lfu_counters[i] / max_lfu))
                else:
                    state.append(0.0)
        
        # 3. Requested file popularity
        state.append(float(self.popularity[requested_file]))
        
        # 4. Cache occupancy ratio
        occupied = np.sum(occupied_mask)
        state.append(float(occupied / self.capacity))
        
        # 5-6. Channel quality statistics
        if len(self.channel_history) > 0:
            recent_channels = list(self.channel_history)[-100:]
            state.append(float(np.mean(recent_channels)))
            state.append(float(np.std(recent_channels)))
        else:
            state.extend([0.5, 0.1])
        
        # 7-8. NOMA performance metrics
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
            action: slot index to place file_id into (0 to capacity-1).
                    Returns -1 only if the file is already cached.
        """
        # If file already cached, no action needed
        if file_id in self.file_to_slot:
            return -1
        
        # Find available (empty) slots
        empty_slots = [i for i, f in enumerate(self.cache_slots) if f == -1]
        
        # If cache not full, fill an empty slot — no eviction decision
        if empty_slots:
            return empty_slots[0]
        
        # Cache is full: epsilon-greedy eviction decision
        if self.eval_mode or random.random() >= self.epsilon:
            if self.use_nn:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    q_values = self.q_network(state_tensor).cpu().numpy()[0]
                    return int(np.argmax(q_values))
            else:
                state_key = self._discretize_state(state)
                q_values = self.q_table[state_key]
                return int(np.argmax(q_values))
        else:
            return random.randint(0, self.capacity - 1)
    
    def _discretize_state(self, state: np.ndarray) -> str:
        """Discretize continuous state for Q-table fallback."""
        key_features = state[:min(10, len(state))]
        bins = []
        for val in key_features:
            if val < 0.0:
                bins.append('E')
            elif val < 0.33:
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
        
        Reward hierarchy (clear ordering prevents perverse incentives):
            +10: cache_hit       — evicted file was re-cached and now hits
            + 2: CIC enabled     — miss but paired interference cancelled
            - 1: NOMA success    — miss, delivered fine via NOMA
            - 5: NOMA failure    — miss, poor QoS
            -10: outage          — miss, complete failure
        
        Optional BER modifier: ±1–2 on top of the above.
        """
        if cache_hit:
            return 10.0
        
        if outage:
            return -10.0
        
        if not noma_success:
            return -5.0
        
        if cic_enabled:
            reward = 2.0
        else:
            reward = -1.0
        
        if ber is not None:
            if ber < 1e-4:
                reward += 1.0
            elif ber > 1e-2:
                reward -= 2.0
        
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
        self.timestep += 1
        
        # ------------------------------------------------------------------
        # 1. Check cache hit (single call — avoids double-counting stats)
        # ------------------------------------------------------------------
        cache_hit = self.is_hit(item, update_stats=True)
        
        # ------------------------------------------------------------------
        # 2. Build NOMA result dict
        # ------------------------------------------------------------------
        paired_cached = False
        if paired_file is not None:
            paired_cached = self.is_hit(paired_file, update_stats=False)
        
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
                result['cic_enabled'] = True
                self.cic_opportunities += 1
            if cache_hit:
                result['strong_user_benefit'] = True
                result['cic_enabled'] = True
                self.noma_paired_hits += 1
        
        # Store user metadata
        if user_id is not None and channel_gain is not None:
            self.channel_gains[user_id] = channel_gain
        if user_id is not None and paired_user is not None:
            self.user_pairings[user_id] = paired_user
        
        # ------------------------------------------------------------------
        # 3. Update NOMA tracking histories
        # ------------------------------------------------------------------
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
        
        # ------------------------------------------------------------------
        # 4. DQN learning with correct credit assignment
        # ------------------------------------------------------------------
        if not self.eval_mode:
            self._learn_from_request(
                file_id=item,
                cache_hit=cache_hit,
                cic_enabled=result['cic_enabled'],
                noma_success=noma_success,
                outage=outage,
                ber=ber,
                episode_done=episode_done
            )
        
        # ------------------------------------------------------------------
        # 5. Update LRU/LFU counters AFTER learning
        # ------------------------------------------------------------------
        self._update_counters(item, cache_hit)
        
        # ------------------------------------------------------------------
        # 6. EMA popularity update (BUG-3 FIX: decay ALL files every step)
        # ------------------------------------------------------------------
        # CORRECT EMA formulation:
        #   For ALL files j:  p_j(t) = decay * p_j(t-1)       [aging]
        #   For requested i:  p_i(t) += (1 - decay)            [observation]
        #   Then re-normalize so probabilities sum to 1.
        #
        # OLD (broken): only updated p[item], leaving other files un-decayed.
        # This caused stale-popularity bias: files popular long ago stayed
        # 'sticky' in the state vector, misleading the eviction policy.
        # Proof: after 100 steps without requests, BROKEN kept p[stale]=0.06
        # while CORRECT correctly dropped it to 0.000016 (3750× difference).
        self.popularity *= self.popularity_decay            # decay ALL files
        self.popularity[item] += (1.0 - self.popularity_decay)  # observe request
        self.popularity /= self.popularity.sum()            # re-normalize
        
        return result
    
    # ========================================================================
    # DEFERRED CREDIT ASSIGNMENT (BUG-1 FIX — 2026)
    # ========================================================================
    
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
        DQN learning with correct deferred credit assignment.
        
        MDP flow:
        ────────────────────────────────────────────────────────────
        A. COMPLETE pending transition for file_id (if it was evicted before)
        B. EVICTION decision on cache miss (park new pending transition)
        C. TRAIN the network
        ────────────────────────────────────────────────────────────
        """
        
        # ────────────────────────────────────────────────────────────
        # STEP A: Complete pending transition for file_id (if any)
        # ────────────────────────────────────────────────────────────
        if file_id in self.pending_transitions:
            pending = self.pending_transitions.pop(file_id)
            
            eviction_reward = self._compute_reward(
                cache_hit=cache_hit,
                cic_enabled=cic_enabled,
                noma_success=noma_success,
                outage=outage,
                ber=ber
            )
            
            current_state_for_completion = self._get_state_vector(file_id)
            
            completed_experience = {
                'state':      pending['state'],
                'action':     pending['action'],
                'reward':     eviction_reward,
                'next_state': current_state_for_completion,
                'done':       episode_done
            }
            
            self._push_to_buffer(completed_experience)
            self.cumulative_reward += eviction_reward
        
        # ────────────────────────────────────────────────────────────
        # STEP B: Eviction decision on cache miss
        # ────────────────────────────────────────────────────────────
        if not cache_hit:
            empty_slots = [i for i, f in enumerate(self.cache_slots) if f == -1]
            cache_full = len(empty_slots) == 0
            
            state_before = self._get_state_vector(file_id)
            action = self._select_action(state_before, file_id)
            
            if action >= 0:
                if cache_full:
                    evicted_file = self.cache_slots[action]
                    self._execute_action(action, file_id)
                    state_after = self._get_state_vector(file_id)
                    
                    if evicted_file != -1:
                        # Flush any stale pending for the same file
                        if evicted_file in self.pending_transitions:
                            old_pending = self.pending_transitions.pop(evicted_file)
                            ghost_experience = {
                                'state':      old_pending['state'],
                                'action':     old_pending['action'],
                                'reward':     -1.0,
                                'next_state': state_before,
                                'done':       False
                            }
                            self._push_to_buffer(ghost_experience)
                        
                        self.pending_transitions[evicted_file] = {
                            'state':      state_before,
                            'action':     action,
                            'next_state': state_after
                        }
                else:
                    # Empty slot fill: immediate experience, no deferral
                    self._execute_action(action, file_id)
                    state_after = self._get_state_vector(file_id)
                    immediate_reward = self._compute_reward(
                        cache_hit=False,
                        cic_enabled=cic_enabled,
                        noma_success=noma_success,
                        outage=outage,
                        ber=ber
                    )
                    fill_experience = {
                        'state':      state_before,
                        'action':     action,
                        'reward':     immediate_reward,
                        'next_state': state_after,
                        'done':       episode_done
                    }
                    self._push_to_buffer(fill_experience)
                    self.cumulative_reward += immediate_reward
        
        # ────────────────────────────────────────────────────────────
        # STEP C: Train the network
        # ────────────────────────────────────────────────────────────
        self.training_step += 1
        
        buf_len = len(self.replay_buffer) if self.replay_buffer is not None else 0
        
        if (buf_len >= self.warm_up_steps
                and self.use_nn
                and self.training_step % self.train_freq == 0
                and buf_len >= self.batch_size):
            self._train_step()
        
        # Decay epsilon
        if not self.eval_mode and self.epsilon > self.epsilon_end:
            self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)
        
        # Episode bookkeeping
        if episode_done:
            self.episode_rewards.append(self.cumulative_reward)
            self.cumulative_reward = 0.0
            # Flush unresolved pending transitions with neutral reward
            for evicted_file, pending in list(self.pending_transitions.items()):
                state_end = self._get_state_vector(file_id)
                flush_experience = {
                    'state':      pending['state'],
                    'action':     pending['action'],
                    'reward':     0.0,
                    'next_state': state_end,
                    'done':       True
                }
                self._push_to_buffer(flush_experience)
            self.pending_transitions.clear()
    
    def _push_to_buffer(self, experience: Dict):
        """Push a completed (s, a, r, s', done) experience to the replay buffer."""
        if self.use_nn and self.replay_buffer is not None:
            if self.use_prioritized:
                self.replay_buffer.add(experience)
            else:
                self.replay_buffer.append(experience)
        elif not self.use_nn:
            self._update_q_table(experience)
    
    def _execute_action(self, action: int, file_id: int):
        """
        Execute cache replacement: put file_id into slot `action`,
        evicting whatever was there before.
        """
        if action < 0 or action >= self.capacity:
            return
        
        old_file = self.cache_slots[action]
        if old_file != -1 and old_file in self.file_to_slot:
            del self.file_to_slot[old_file]
        
        self.cache_slots[action] = file_id
        self.file_to_slot[file_id] = action
        
        self.lru_counters[action] = 0
        self.lfu_counters[action] = 1
    
    def _update_counters(self, file_id: int, cache_hit: bool):
        """
        Update LRU/LFU counters.
        LRU: increment all occupied slots; reset hit slot.
        LFU: increment hit slot.
        """
        for i in range(self.capacity):
            if self.cache_slots[i] != -1:
                self.lru_counters[i] += 1
        
        if cache_hit and file_id in self.file_to_slot:
            slot = self.file_to_slot[file_id]
            self.lru_counters[slot] = 0
            self.lfu_counters[slot] += 1
    
    # ========================================================================
    # TRAINING (Double DQN with Prioritized Replay)
    # ========================================================================
    
    def _train_step(self):
        """
        Single DQN training step (Double DQN + soft target update).
        
        Double DQN target:
            r + γ * Q_target(s', argmax_a Q_policy(s', a))
        """
        if not self.use_nn or len(self.replay_buffer) < self.batch_size:
            return
        
        if self.use_prioritized:
            experiences, weights, indices = self.replay_buffer.sample(self.batch_size)
            if experiences is None:
                return
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            experiences = random.sample(list(self.replay_buffer), self.batch_size)
            weights = torch.ones(self.batch_size).to(self.device)
            indices = None
        
        states      = torch.from_numpy(np.array([e['state']      for e in experiences], dtype=np.float32)).to(self.device)
        actions     = torch.from_numpy(np.array([e['action']     for e in experiences], dtype=np.int64)).to(self.device)
        rewards     = torch.from_numpy(np.array([e['reward']     for e in experiences], dtype=np.float32)).to(self.device)
        next_states = torch.from_numpy(np.array([e['next_state'] for e in experiences], dtype=np.float32)).to(self.device)
        dones       = torch.from_numpy(np.array([e['done']       for e in experiences], dtype=np.float32)).to(self.device)
        
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + (1.0 - dones) * self.gamma * next_q
        
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
        """
        Soft update: θ_target ← τ * θ_policy + (1-τ) * θ_target
        """
        for target_param, param in zip(self.target_network.parameters(),
                                       self.q_network.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )
    
    def _update_q_table(self, experience: Dict):
        """
        Q-learning update for Q-table fallback.
        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        """
        state_key = self._discretize_state(experience['state'])
        next_key  = self._discretize_state(experience['next_state'])
        action    = experience['action']
        reward    = experience['reward']
        done      = experience['done']
        
        current_q  = self.q_table[state_key][action]
        next_max_q = 0.0 if done else np.max(self.q_table[next_key])
        target_q   = reward + self.gamma * next_max_q
        self.q_table[state_key][action] += self.lr * (target_q - current_q)
    
    # ========================================================================
    # CACHE INTERFACE METHODS
    # ========================================================================
    
    def populate(self, items: Optional[Iterable[int]] = None):
        """
        Initialize cache with most popular files.
        
        Args:
            items: Optional ordered list of file IDs to pre-load
        """
        if items is None:
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
        """
        hit = int(item) in self.file_to_slot
        
        if update_stats:
            if hit:
                self._record_hit()
            else:
                self._record_miss()
        
        return hit
    
    def get_contents(self) -> Set[int]:
        """Return current cache contents as a set of file IDs."""
        return set(f for f in self.cache_slots if f != -1)
    
    def clear(self):
        """Clear cache and reset all state."""
        self.cache_slots = [-1] * self.capacity
        self.file_to_slot.clear()
        self.lru_counters = np.zeros(self.capacity, dtype=np.int32)
        self.lfu_counters = np.zeros(self.capacity, dtype=np.int32)
        self.pending_transitions.clear()
        self.reset_stats()
    
    def set_eval_mode(self, eval_mode: bool = True):
        """
        Toggle evaluation mode (no exploration, no training).
        Saves/restores epsilon to prevent decay corruption.
        """
        self.eval_mode = eval_mode
        
        if eval_mode:
            if not hasattr(self, '_training_epsilon'):
                self._training_epsilon = self.epsilon
            self.epsilon = 0.0
            if self.use_nn:
                self.q_network.eval()
        else:
            if hasattr(self, '_training_epsilon'):
                self.epsilon = self._training_epsilon
                delattr(self, '_training_epsilon')
            if self.use_nn:
                self.q_network.train()
    
    # ========================================================================
    # MODEL PERSISTENCE
    # ========================================================================
    
    def save_model(self, filepath: str):
        """Save learned model to file."""
        if self.use_nn:
            save_dict = {
                'q_network':       self.q_network.state_dict(),
                'target_network':  self.target_network.state_dict(),
                'optimizer':       self.optimizer.state_dict(),
                'training_step':   self.training_step,
                'epsilon':         self.epsilon,
                'popularity':      self.popularity,
                'cache_slots':     self.cache_slots,
                'file_to_slot':    self.file_to_slot,
                'lru_counters':    self.lru_counters,
                'lfu_counters':    self.lfu_counters
            }
            if self.use_prioritized:
                save_dict['per_beta']      = self.replay_buffer.get_beta()
                save_dict['per_frame_idx'] = self.replay_buffer.frame_idx
            torch.save(save_dict, filepath)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'q_table':       dict(self.q_table),
                    'training_step': self.training_step,
                    'epsilon':       self.epsilon,
                    'popularity':    self.popularity,
                    'cache_slots':   self.cache_slots,
                    'file_to_slot':  self.file_to_slot
                }, f)
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load learned model from file."""
        if self.use_nn:
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
            self.q_network.load_state_dict(checkpoint['q_network'])
            self.target_network.load_state_dict(checkpoint['target_network'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.training_step  = checkpoint['training_step']
            self.epsilon        = checkpoint['epsilon']
            self.popularity     = checkpoint['popularity']
            self.cache_slots    = checkpoint['cache_slots']
            self.file_to_slot   = checkpoint['file_to_slot']
            self.lru_counters   = checkpoint.get('lru_counters', self.lru_counters)
            self.lfu_counters   = checkpoint.get('lfu_counters', self.lfu_counters)
            if self.use_prioritized and 'per_beta' in checkpoint:
                self.replay_buffer.beta      = checkpoint['per_beta']
                self.replay_buffer.frame_idx = checkpoint.get('per_frame_idx', 0)
        else:
            with open(filepath, 'rb') as f:
                checkpoint = pickle.load(f)
            self.q_table        = defaultdict(lambda: np.zeros(self.action_dim), checkpoint['q_table'])
            self.training_step  = checkpoint['training_step']
            self.epsilon        = checkpoint['epsilon']
            self.popularity     = checkpoint['popularity']
            self.cache_slots    = checkpoint['cache_slots']
            self.file_to_slot   = checkpoint['file_to_slot']
        print(f"✅ Model loaded from {filepath}")
    
    # ========================================================================
    # STATISTICS
    # ========================================================================
    
    def get_stats(self) -> Dict:
        """Get comprehensive DQN + cache statistics."""
        base_stats = super().stats()
        
        dqn_stats = {
            'training_step':       self.training_step,
            'epsilon':             self.epsilon,
            'eval_mode':           self.eval_mode,
            'avg_episode_reward':  np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
            'avg_loss':            np.mean(self.losses[-100:]) if self.losses else 0,
            'replay_buffer_size':  len(self.replay_buffer) if self.replay_buffer else 0,
            'use_neural_network':  self.use_nn,
            'cic_count':           self.cic_count,
            'sic_count':           self.sic_count,
            'warm_up_steps':       self.warm_up_steps,
            'pending_transitions': len(self.pending_transitions)
        }
        
        if self.use_prioritized and self.replay_buffer is not None:
            try:
                dqn_stats['beta'] = self.replay_buffer.get_beta()
            except (AttributeError, TypeError):
                dqn_stats['beta'] = 0.0
        else:
            dqn_stats['beta'] = 0.0
        
        return {**base_stats, **dqn_stats}


# Alias for compatibility
StableDQNCache = DQNCache
