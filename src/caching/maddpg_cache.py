"""
src/caching/maddpg_cache.py

MADDPG (Multi-Agent Deep Deterministic Policy Gradient) Cache Agent
===================================================================

Multi-agent actor-critic caching policy based on the paper:
"Multi-Agent DRL for Resource Allocation and Cache Design in
Terrestrial-Satellite Networks" (Li et al., IEEE TWC 2023).

Adapted for single-cell NOMA: Instead of multiple BSs/satellites as
agents, we partition users into K groups. Each agent handles caching
decisions for its group with a decentralized actor but centralized
critic that sees ALL agents' observations and actions.

Key features:
  - K decentralized actors (one per user group)
  - 1 centralized critic per agent (sees global state + all actions)
  - Same 14-dim per-agent state, same reward as DQN for fair comparison
  - Global coordination through centralized training

Reference: Lowe et al., "Multi-Agent Actor-Critic for Mixed Cooperative-
Competitive Environments", NeurIPS 2017.

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
# ACTOR NETWORK (per-agent, decentralized)
# ============================================================================

class MADDPGActor(nn.Module):
    """Per-agent actor: local observation -> action [0,1]."""

    def __init__(self, obs_dim: int, hidden_dims: List[int] = [64, 32]):
        super().__init__()
        layers = []
        prev_dim = obs_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.ReLU())
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        final = list(self.modules())[-2]
        if isinstance(final, nn.Linear):
            nn.init.uniform_(final.weight, -3e-3, 3e-3)
            nn.init.uniform_(final.bias, -3e-3, 3e-3)

    def forward(self, obs: 'torch.Tensor') -> 'torch.Tensor':
        return self.network(obs)


# ============================================================================
# CRITIC NETWORK (centralized — sees all agents' obs + actions)
# ============================================================================

class MADDPGCritic(nn.Module):
    """Centralized critic: (all_obs, all_actions) -> Q-value."""

    def __init__(self, total_obs_dim: int, total_action_dim: int,
                 hidden_dims: List[int] = [128, 64]):
        super().__init__()
        input_dim = total_obs_dim + total_action_dim
        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.ReLU())
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, all_obs: 'torch.Tensor', all_actions: 'torch.Tensor') -> 'torch.Tensor':
        x = torch.cat([all_obs, all_actions], dim=1)
        return self.network(x)


# ============================================================================
# REPLAY BUFFER (stores multi-agent transitions)
# ============================================================================

class MAReplayBuffer:
    """Replay buffer for multi-agent transitions."""

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
# MADDPG CACHE AGENT
# ============================================================================

class MADDPGCache(CacheBase):
    """
    Multi-Agent DDPG caching policy.

    Partitions users into K groups. Each group has a dedicated actor
    (decentralized execution). The critic is centralized, seeing all
    agents' observations and actions during training.

    Uses the SAME state space (14 dims per agent), reward function,
    and cache interface as DQN/DDPG for fair comparison. The global
    state for the centralized critic is the concatenation of all
    agents' local observations.
    """

    RECENT_WINDOW = 500

    def __init__(
        self, capacity: int, num_files: int = 2000, num_users: int = 200,
        num_agents: int = 4,
        learning_rate: float = 0.001, gamma: float = 0.95,
        hidden_dims: List[int] = [64, 32],
        critic_hidden_dims: List[int] = [128, 64],
        batch_size: int = 64, replay_buffer_size: int = 50000,
        train_freq: int = 4, warm_up_steps: int = 2000,
        gradient_clip: float = 10.0, tau: float = 0.005,
        ou_sigma: float = 0.2, ou_theta: float = 0.15,
        ou_sigma_min: float = 0.02, ou_sigma_decay: float = 0.9999,
        enable_noma_awareness: bool = True, seed: int = 2025,
        **kwargs
    ):
        super().__init__(capacity, enable_noma_awareness)
        self.num_files = num_files
        self.num_users = num_users
        self.num_agents = num_agents
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.train_freq = train_freq
        self.gradient_clip = gradient_clip
        self.tau = tau
        self.warm_up_steps = warm_up_steps
        self.eval_mode = False
        self._set_seeds(seed)

        # Cache storage (shared across all agents)
        self.cache_set = set()
        self.file_to_slot = {}
        self.cache_slots = [-1] * capacity
        self.timestep = 0

        # Popularity tracking
        self.request_counts = np.zeros(num_files, dtype=np.float64)
        self.recent_requests = deque(maxlen=self.RECENT_WINDOW)

        # NOMA tracking
        self.channel_history = deque(maxlen=500)
        self.noma_history = deque(maxlen=500)
        self.cic_count = 0
        self.sic_count = 0

        # State/action dims
        self.obs_dim = 14  # Per-agent observation
        self.action_dim_per_agent = 1

        # User-to-agent mapping
        self.user_to_agent = {}
        users_per_agent = max(1, num_users // num_agents)
        for u in range(num_users):
            self.user_to_agent[u] = min(u // users_per_agent, num_agents - 1)

        # Per-agent last observations (for centralized critic)
        self.agent_last_obs = [np.zeros(self.obs_dim, dtype=np.float32)
                               for _ in range(num_agents)]
        self.agent_last_action = [0.0] * num_agents

        if TORCH_AVAILABLE:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # Create per-agent actor networks (decentralized)
            self.actors = nn.ModuleList([
                MADDPGActor(self.obs_dim, hidden_dims)
                for _ in range(num_agents)
            ]).to(self.device)

            self.actors_target = nn.ModuleList([
                MADDPGActor(self.obs_dim, hidden_dims)
                for _ in range(num_agents)
            ]).to(self.device)

            # Copy weights to targets
            for i in range(num_agents):
                self.actors_target[i].load_state_dict(self.actors[i].state_dict())
                self.actors_target[i].eval()

            # Centralized critics (one per agent, sees everything)
            total_obs = self.obs_dim * num_agents
            total_act = num_agents  # 1 action per agent
            self.critics = nn.ModuleList([
                MADDPGCritic(total_obs, total_act, critic_hidden_dims)
                for _ in range(num_agents)
            ]).to(self.device)

            self.critics_target = nn.ModuleList([
                MADDPGCritic(total_obs, total_act, critic_hidden_dims)
                for _ in range(num_agents)
            ]).to(self.device)

            for i in range(num_agents):
                self.critics_target[i].load_state_dict(self.critics[i].state_dict())
                self.critics_target[i].eval()

            # Optimizers
            self.actor_optimizers = [
                optim.Adam(self.actors[i].parameters(), lr=learning_rate, weight_decay=1e-5)
                for i in range(num_agents)
            ]
            self.critic_optimizers = [
                optim.Adam(self.critics[i].parameters(), lr=learning_rate, weight_decay=1e-5)
                for i in range(num_agents)
            ]

            # Shared replay buffer
            self.replay_buffer = MAReplayBuffer(replay_buffer_size)

            # Per-agent OU noise
            self.noises = [
                self._create_noise(ou_theta, ou_sigma, ou_sigma_min, ou_sigma_decay)
                for _ in range(num_agents)
            ]
        else:
            self.replay_buffer = None

        self.training_step = 0
        self.episode_rewards = []
        self.losses = []
        self.cumulative_reward = 0.0

        print(f'MADDPGCache initialized ({num_agents} agents, centralized critic)')
        print(f'  Obs dim    : {self.obs_dim} per agent')
        print(f'  Actors     : {num_agents} x {hidden_dims}')
        print(f'  Critics    : {num_agents} x {critic_hidden_dims} (centralized)')
        print(f'  Warm-up    : {self.warm_up_steps} steps')

    def _set_seeds(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        if TORCH_AVAILABLE:
            torch.manual_seed(seed)

    def _create_noise(self, theta, sigma, sigma_min, sigma_decay):
        class _OUNoise:
            def __init__(self):
                self.mu = 0.0
                self.theta = theta
                self.sigma = sigma
                self.sigma_min = sigma_min
                self.sigma_decay = sigma_decay
                self.state = 0.0
            def reset(self):
                self.state = 0.0
            def sample(self):
                dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn()
                self.state += dx
                self.sigma = max(self.sigma_min, self.sigma * self.sigma_decay)
                return self.state
        return _OUNoise()

    # =========================================================================
    # POPULARITY / STATE (identical to DQN/DDPG)
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

    def _get_state_vector(self, requested_file: int) -> np.ndarray:
        """14-dim state vector (identical to DQN/DDPG)."""
        state = []
        state.append(self._get_file_popularity_rank(requested_file))
        state.append(self._get_file_recent_freq(requested_file))
        state.append(1.0 if requested_file in self.cache_set else 0.0)
        state.append(len(self.cache_set) / self.capacity)

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

        if self.cache_set:
            _, worst_file = self._get_min_popularity_slot()
            would_improve = 1.0 if (worst_file >= 0 and self.request_counts[requested_file] > self.request_counts[worst_file]) else (1.0 if worst_file < 0 else 0.0)
        else:
            would_improve = 1.0
        state.append(would_improve)

        if self.cache_set and len(self.cache_set) >= self.capacity:
            _, worst_file = self._get_min_popularity_slot()
            if worst_file >= 0 and self.request_counts[worst_file] > 0:
                ratio = float(self.request_counts[requested_file] / self.request_counts[worst_file])
                state.append(min(ratio, 5.0) / 5.0)
            else:
                state.append(1.0)
        else:
            state.append(1.0)

        if self.noma_history:
            nh = list(self.noma_history)[-100:]
            state.append(float(sum(1 for x in nh if x.get('cic', False)) / len(nh)))
        else:
            state.append(0.0)

        if self.channel_history:
            ch = list(self.channel_history)[-100:]
            state.extend([float(np.mean(ch)), float(np.std(ch))])
        else:
            state.extend([0.5, 0.1])

        if self.noma_history:
            nh = list(self.noma_history)[-100:]
            state.append(float(sum(1 for x in nh if x.get('success', False)) / len(nh)))
        else:
            state.append(0.5)

        if self.recent_requests:
            state.append(len(set(self.recent_requests)) / len(self.recent_requests))
        else:
            state.append(1.0)

        return np.array(state, dtype=np.float32)

    # =========================================================================
    # ACTION SELECTION
    # =========================================================================

    def _select_action(self, obs: np.ndarray, file_id: int,
                       agent_id: int) -> Tuple[float, int]:
        if file_id in self.cache_set:
            return 0.0, 0

        if TORCH_AVAILABLE:
            with torch.no_grad():
                t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                action_value = self.actors[agent_id](t).cpu().numpy()[0, 0]

            if not self.eval_mode:
                noise = self.noises[agent_id].sample()
                action_value = np.clip(action_value + noise, 0.0, 1.0)

            binary = 1 if action_value >= 0.5 else 0
            return float(action_value), binary
        else:
            return 0.5, random.randint(0, 1)

    # =========================================================================
    # REWARD (identical to DQN/DDPG)
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
                reward += 0.5 if evicted_rank > pop_rank else -0.5
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
    # CACHE EXECUTION (shared cache, identical to DQN/DDPG)
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
    # MAIN REQUEST
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

        # Determine which agent handles this user
        agent_id = self.user_to_agent.get(user_id, 0) if user_id is not None else 0

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

        # ── MADDPG Learning ──
        if not self.eval_mode:
            obs = self._get_state_vector(item)
            cont_action, binary_action = self._select_action(obs, item, agent_id)

            # Update per-agent tracking
            self.agent_last_obs[agent_id] = obs.copy()
            self.agent_last_action[agent_id] = cont_action

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

            next_obs = self._get_state_vector(item)

            if TORCH_AVAILABLE and self.replay_buffer is not None:
                # Store multi-agent transition
                self.replay_buffer.push({
                    'agent_id': agent_id,
                    'obs': obs,
                    'action': np.array([cont_action], dtype=np.float32),
                    'reward': reward,
                    'next_obs': next_obs,
                    'done': episode_done,
                    'all_obs': [o.copy() for o in self.agent_last_obs],
                    'all_actions': [a for a in self.agent_last_action],
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
                for noise in self.noises:
                    noise.reset()

        else:
            if not cache_hit:
                obs = self._get_state_vector(item)
                _, binary_action = self._select_action(obs, item, agent_id)
                if binary_action == 1:
                    self._execute_cache(item)

        return result

    # =========================================================================
    # TRAINING STEP (centralized training, decentralized execution)
    # =========================================================================

    def _train_step(self):
        if not TORCH_AVAILABLE or self.replay_buffer is None:
            return
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = self.replay_buffer.sample(self.batch_size)

        # Process batch for each agent
        for agent_idx in range(self.num_agents):
            # Filter transitions for this agent (or use all with agent-specific indexing)
            agent_batch = [t for t in batch if t['agent_id'] == agent_idx]
            if len(agent_batch) < max(self.batch_size // self.num_agents, 4):
                # Use all transitions if not enough for this agent
                agent_batch = batch[:max(self.batch_size // self.num_agents, 8)]

            if len(agent_batch) < 4:
                continue

            obs = torch.FloatTensor(
                np.array([t['obs'] for t in agent_batch])).to(self.device)
            actions = torch.FloatTensor(
                np.array([t['action'] for t in agent_batch])).to(self.device)
            rewards = torch.FloatTensor(
                np.array([t['reward'] for t in agent_batch])).unsqueeze(1).to(self.device)
            next_obs = torch.FloatTensor(
                np.array([t['next_obs'] for t in agent_batch])).to(self.device)
            dones = torch.FloatTensor(
                np.array([float(t['done']) for t in agent_batch])).unsqueeze(1).to(self.device)

            # Build global obs and actions for centralized critic
            all_obs = torch.FloatTensor(
                np.array([np.concatenate(t['all_obs']) for t in agent_batch])).to(self.device)
            all_actions = torch.FloatTensor(
                np.array([t['all_actions'] for t in agent_batch])).to(self.device)

            # Next global state: use current next_obs for this agent, last for others
            next_all_obs_list = []
            next_all_actions_list = []
            for t in agent_batch:
                nao = [o.copy() for o in t['all_obs']]
                nao[agent_idx] = t['next_obs']
                next_all_obs_list.append(np.concatenate(nao))

                with torch.no_grad():
                    next_acts = []
                    for j in range(self.num_agents):
                        o_j = torch.FloatTensor(nao[j]).unsqueeze(0).to(self.device)
                        next_acts.append(self.actors_target[j](o_j).cpu().numpy()[0, 0])
                    next_all_actions_list.append(next_acts)

            next_all_obs = torch.FloatTensor(np.array(next_all_obs_list)).to(self.device)
            next_all_actions = torch.FloatTensor(np.array(next_all_actions_list)).to(self.device)

            # ── Critic update ──
            with torch.no_grad():
                q_target = rewards + self.gamma * (1 - dones) * \
                           self.critics_target[agent_idx](next_all_obs, next_all_actions)

            q_current = self.critics[agent_idx](all_obs, all_actions)
            critic_loss = F.mse_loss(q_current, q_target)

            self.critic_optimizers[agent_idx].zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critics[agent_idx].parameters(), self.gradient_clip)
            self.critic_optimizers[agent_idx].step()

            # ── Actor update ──
            predicted_actions = self.actors[agent_idx](obs)
            # Replace this agent's action in all_actions with predicted
            all_actions_pred = all_actions.clone()
            all_actions_pred[:, agent_idx:agent_idx+1] = predicted_actions
            actor_loss = -self.critics[agent_idx](all_obs, all_actions_pred).mean()

            self.actor_optimizers[agent_idx].zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actors[agent_idx].parameters(), self.gradient_clip)
            self.actor_optimizers[agent_idx].step()

            self.losses.append(float(critic_loss.item()))

        # ── Soft target updates ──
        for i in range(self.num_agents):
            for p, tp in zip(self.actors[i].parameters(), self.actors_target[i].parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.critics[i].parameters(), self.critics_target[i].parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

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
        self.agent_last_obs = [np.zeros(self.obs_dim, dtype=np.float32)
                               for _ in range(self.num_agents)]
        self.agent_last_action = [0.0] * self.num_agents
        super().clear()

    def reset_popularity(self):
        self.request_counts = np.zeros(self.num_files, dtype=np.float64)
        self.recent_requests.clear()

    def get_stats(self) -> Dict:
        avg_loss = float(np.mean(self.losses[-100:])) if self.losses else 0.0
        avg_sigma = float(np.mean([n.sigma for n in self.noises])) if TORCH_AVAILABLE else 0.0
        return {
            'epsilon': avg_sigma,
            'avg_loss': avg_loss,
            'training_steps': self.training_step,
            'buffer_size': len(self.replay_buffer) if self.replay_buffer else 0,
            'beta': 0.0,
            'algorithm': 'MADDPG',
        }

    def save_model(self, filepath: str):
        if TORCH_AVAILABLE:
            state = {}
            for i in range(self.num_agents):
                state[f'actor_{i}'] = self.actors[i].state_dict()
                state[f'critic_{i}'] = self.critics[i].state_dict()
                state[f'actor_target_{i}'] = self.actors_target[i].state_dict()
                state[f'critic_target_{i}'] = self.critics_target[i].state_dict()
            torch.save(state, filepath)

    def load_model(self, filepath: str):
        if TORCH_AVAILABLE:
            checkpoint = torch.load(filepath, map_location=self.device)
            for i in range(self.num_agents):
                self.actors[i].load_state_dict(checkpoint[f'actor_{i}'])
                self.critics[i].load_state_dict(checkpoint[f'critic_{i}'])
                self.actors_target[i].load_state_dict(checkpoint[f'actor_target_{i}'])
                self.critics_target[i].load_state_dict(checkpoint[f'critic_target_{i}'])
