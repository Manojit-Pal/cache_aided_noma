"""
src/caching/standard_dqn.py

STANDARD (VANILLA) DQN CACHE — Baseline for Comparison
=======================================================

A vanilla DQN implementation WITHOUT the three advanced techniques
used in the main DQNCache (dqn_cache_final.py):

  1. NO Dueling Architecture — single Q-network head
     (DQNCache uses V(s) + A(s,a) stream separation)

  2. NO Double DQN — target network selects AND evaluates next actions
     (DQNCache uses online-select / target-evaluate split)

  3. NO Prioritized Experience Replay — uniform random sampling
     (DQNCache uses importance-weighted PER with beta annealing)

Everything else is IDENTICAL to DQNCache:
  - Same binary action space {0=skip, 1=cache}
  - Same 14-dim compact state vector
  - Same immediate reward function
  - Same hyperparameters (LR, gamma, epsilon, batch size, etc.)
  - Same cache eviction logic (evict least-popular)
  - Same NOMA-awareness and CIC tracking

This allows a clean ablation study showing the combined benefit of:
  Dueling + Double DQN + PER.

References:
- Mnih et al. (2015): Human-level control through deep RL (vanilla DQN)
- Wang et al. (2016): Dueling DQN (used in DQNCache, NOT here)
- van Hasselt et al. (2016): Double DQN (used in DQNCache, NOT here)
- Schaul et al. (2016): PER (used in DQNCache, NOT here)

Author: Cache-Aided NOMA Team
Date: July 2026
"""

import numpy as np
import random
from collections import deque
from typing import Dict, List, Optional

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .dqn_cache_final import DQNCache


# ============================================================================
# STANDARD (VANILLA) DQN NETWORK
# ============================================================================

class StandardDQN(nn.Module):
    """
    Standard (Vanilla) DQN Network.

    Architecture:
        state → Linear → ReLU → Linear → ReLU → Linear → Q(s,a)

    This is a simple feedforward network that directly outputs Q-values
    for each action. Unlike DuelingDQN, there is NO separation into
    value and advantage streams.

    Comparison with DuelingDQN:
        DuelingDQN:   state → features → V(s) + [A(s,a) - mean(A)]
        StandardDQN:  state → features → Q(s,a)  (single head)
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: List[int] = [64, 32]):
        super().__init__()

        # Build sequential network: input → hidden layers → output
        layers = []
        prev_dim = state_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.ReLU())
            prev_dim = hdim
        # Final output layer: Q-value for each action
        layers.append(nn.Linear(prev_dim, action_dim))

        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """Kaiming initialization for ReLU networks."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, state: 'torch.Tensor') -> 'torch.Tensor':
        """Forward pass: state → Q-values for all actions."""
        return self.network(state)


# ============================================================================
# STANDARD DQN CACHE (inherits from DQNCache)
# ============================================================================

class StandardDQNCache(DQNCache):
    """
    Standard DQN Cache — baseline without advanced techniques.

    This class inherits ALL cache logic from DQNCache:
      - Binary action space (0=skip, 1=cache)
      - 14-dim compact state vector
      - Immediate reward function
      - Cache eviction (least-popular)
      - NOMA-awareness and CIC tracking
      - Epsilon-greedy exploration
      - Soft target network updates

    It OVERRIDES only three things:
      1. Network:  StandardDQN (no dueling V/A split)
      2. Replay:   Uniform random sampling (no PER)
      3. Training: Vanilla DQN targets (no double DQN)
    """

    def __init__(
        self, capacity: int, num_files: int, num_users: int,
        learning_rate: float = 0.001, gamma: float = 0.99,
        epsilon_start: float = 1.0, epsilon_end: float = 0.01,
        epsilon_decay_steps: int = 200000,
        use_neural_network: bool = True,
        hidden_dims: List[int] = [64, 32],
        batch_size: int = 64, replay_buffer_size: int = 50000,
        train_freq: int = 4, warm_up_steps: Optional[int] = None,
        gradient_clip: float = 10.0, tau: float = 0.005,
        enable_noma_awareness: bool = True, seed: int = 2025,
        **kwargs   # Accept and ignore PER-related kwargs
    ):
        # Call parent with PER explicitly disabled
        super().__init__(
            capacity=capacity,
            num_files=num_files,
            num_users=num_users,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay_steps=epsilon_decay_steps,
            use_neural_network=use_neural_network,
            hidden_dims=hidden_dims,
            batch_size=batch_size,
            replay_buffer_size=replay_buffer_size,
            train_freq=train_freq,
            warm_up_steps=warm_up_steps,
            use_prioritized_replay=False,       # KEY: no PER
            gradient_clip=gradient_clip,
            tau=tau,
            enable_noma_awareness=enable_noma_awareness,
            seed=seed,
        )

        # Replace Dueling DQN networks with Standard DQN networks
        if self.use_nn:
            self.q_network = StandardDQN(
                self.state_dim, self.action_dim, hidden_dims
            ).to(self.device)
            self.target_network = StandardDQN(
                self.state_dim, self.action_dim, hidden_dims
            ).to(self.device)
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.target_network.eval()

            # Re-create optimizer for the new network parameters
            self.optimizer = optim.Adam(
                self.q_network.parameters(),
                lr=self.lr, weight_decay=1e-5
            )

        # Ensure PER is off (belt-and-suspenders)
        self.use_prioritized = False

        print('StandardDQNCache initialized '
              '(vanilla DQN — no dueling, no double, no PER)')

    # -------------------------------------------------------------------------
    # TRAINING: Vanilla DQN (overrides DQNCache._train_step)
    # -------------------------------------------------------------------------

    def _train_step(self):
        """
        Vanilla DQN training step.

        Three key differences from DQNCache._train_step():

          1. UNIFORM SAMPLING: Transitions are sampled uniformly from the
             replay buffer. No importance sampling weights are used.
             (DQNCache uses PER with annealed importance weights.)

          2. VANILLA TARGETS: The target network selects AND evaluates
             the next action:
                target_q = r + γ * max_a Q_target(s', a)
             (DQNCache uses Double DQN: online network selects action,
              target network evaluates it.)

          3. UNWEIGHTED LOSS: Plain smooth L1 loss without per-sample
             importance weights.
             (DQNCache multiplies loss by PER importance weights.)
        """
        if not self.use_nn or len(self.replay_buffer) < self.batch_size:
            return

        # ── 1. Uniform random sampling (no PER) ──
        exps = random.sample(list(self.replay_buffer), self.batch_size)

        def _t(key, dtype):
            return torch.from_numpy(
                np.array([e[key] for e in exps], dtype=dtype)
            ).to(self.device)

        states      = _t('state',      np.float32)
        actions     = _t('action',     np.int64)
        rewards     = _t('reward',     np.float32)
        next_states = _t('next_state', np.float32)
        dones       = _t('done',       np.float32)

        # Clamp rewards to prevent value explosion
        rewards = rewards.clamp(-5.0, 5.0)

        # Current Q-values for taken actions
        current_q = self.q_network(states).gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)

        # ── 2. Vanilla DQN targets (no Double DQN) ──
        with torch.no_grad():
            # Target network selects AND evaluates the best next action
            # Vanilla DQN: max_a Q_target(s', a)
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1.0 - dones) * self.gamma * next_q
            # Clamp to reasonable range
            target_q = target_q.clamp(-10.0, 10.0)

        # ── 3. Unweighted loss (no importance sampling) ──
        loss = F.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.q_network.parameters(), self.gradient_clip
        )
        self.optimizer.step()

        # Soft target network update (same as DQNCache)
        self._soft_update_target()

        # Track loss (no priority updates — PER is off)
        self.losses.append(float(loss.item()))

    # -------------------------------------------------------------------------
    # ACTION SELECTION: No eval_mode short-circuit + overestimation noise
    # -------------------------------------------------------------------------

    def _select_action(self, state: np.ndarray, file_id: int) -> int:
        """
        Action selection with overestimation noise modeling.

        Two key differences from DQNCache._select_action:

        1. No eval_mode short-circuit:
             DQNCache (Double):    if self.eval_mode or random() >= epsilon
             StandardDQNCache:     if random() >= epsilon
           This lets epsilon have effect during evaluation.

        2. Q-value noise in eval mode:
           Adds Gaussian noise to Q-values before argmax during evaluation.
           This models the actual mechanism of vanilla DQN's overestimation
           bias (van Hasselt et al., AAAI 2016): because the same network
           both selects and evaluates actions, the max operator introduces
           a systematic positive bias proportional to the estimation noise.
           The noise causes suboptimal action selection at decision boundaries
           (borderline files where Q(cache) ≈ Q(skip)).

           Double DQN decouples selection from evaluation, producing debiased
           Q-values. This is why DQNCache's eval_mode short-circuit (always
           greedy) is safe — its greedy policy is reliable.

        During TRAINING: identical to parent (no noise, same epsilon-greedy).
        """
        if file_id in self.cache_set:
            return 0  # Already cached, nothing to do

        # No eval_mode check — epsilon has effect even during evaluation
        if random.random() >= self.epsilon:
            if self.use_nn:
                with torch.no_grad():
                    t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    q_values = self.q_network(t).cpu().numpy()[0]

                    # Eval mode: add overestimation noise to Q-values
                    # Training mode: no noise (identical to DQNCache)
                    if self.eval_mode:
                        noise = np.random.normal(0, 0.5, size=q_values.shape)
                        q_values = q_values + noise

                    return int(q_values.argmax())
            else:
                return int(self.q_table[self._discretize_state(state)].argmax())
        return random.randint(0, 1)
