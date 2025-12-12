# src/simulation/stable_dqn_sim.py
"""
Stable NOMA-Aware DQN Cache Simulator with Train-Test-Eval Workflow

This module implements a comprehensive DQN training and evaluation pipeline for
Cache-Aided NOMA systems with proper separation of:
1. Training Phase: DQN learns optimal caching policies
2. Testing Phase: Validate DQN performance during training
3. Evaluation Phase: Fair comparison against baseline policies

Key Features:
- ✅ NOMA-aware state representation (channel gains, SIC/CIC status)
- ✅ Cache-Aided Interference Cancellation (CIC) rewards
- ✅ Proper train/test/eval split (following ML best practices)
- ✅ Baseline comparison (TopK, LRU, LFU, Random)
- ✅ Comprehensive metrics tracking
- ✅ Model checkpointing and loading

Research References:
- "Power Allocation in Cache-Aided NOMA Systems: Optimization and Deep
  Reinforcement Learning Approaches" (arXiv:1909.11074)
- "Cache-Aided NOMA Mobile Edge Computing: A Reinforcement Learning
  Approach" (arXiv:1906.08812)
- "Deep Q-Learning-Based Content Caching With Update Strategy"
  (IEEE, Jiang et al., 2019)
- "Train-Test Split for Evaluating ML Algorithms" (scikit-learn methodology)

Author: Cache-Aided NOMA Team
Date: December 12, 2025
Version: 2.0 (NOMA-Aware with Train-Test-Eval)
"""

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Any
import os
import json
from datetime import datetime

# Project imports
from src import config
from src.utils import set_seed, sample_zipf_catalog

# NOMA module imports
from src.noma import (
    generate_user_positions,
    compute_channel_gains,
    pair_users,
    allocate_power,
    simulate_sic_process,
    sinr_threshold_from_rate,
    rate_from_sinr
)

# Caching module imports
from src.caching import (
    create_cache,
    CacheBase,
    StaticTopKCache,
    LRUCache,
    LFUCache,
    RandomCache
)

# Try to import DQN cache
try:
    from src.caching import DQNCache
    HAS_DQN = True
except ImportError:
    HAS_DQN = False
    print("⚠️  DQN cache not available - cannot run this simulator")
    print("    Please install PyTorch and ensure dqn_cache_final.py exists")


# ============================================================================
# NOMA-AWARE DQN TRAINER
# ============================================================================

class NOMADQNTrainer:
    """
    NOMA-Aware DQN Training System.
    
    Implements proper ML workflow:
    1. Training: Learn from workload
    2. Testing: Validate during training (early stopping)
    3. Evaluation: Final assessment (never seen before)
    
    NOMA-Specific Features:
    - State includes channel gains, SIC status, cache status
    - Rewards account for CIC opportunities
    - Metrics track NOMA performance (outage, SIC success, CIC benefit)
    """
    
    def __init__(self, cfg, verbose: bool = True):
        """
        Initialize NOMA-aware DQN trainer.
        
        Args:
            cfg: Configuration object
            verbose: Whether to print progress
        """
        self.cfg = cfg
        self.verbose = verbose
        
        # Create save directory
        self.save_dir = 'models/dqn_cache'
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Training history
        self.train_history = []
        self.test_history = []
        
        # Best model tracking
        self.best_hit_rate = 0.0
        self.best_episode = 0
        
        if self.verbose:
            self._print_header()
    
    def _print_header(self):
        """Print training session header."""
        print("\n" + "#"*80)
        print("#" + " "*20 + "NOMA-AWARE DQN TRAINING SYSTEM" + " "*21 + "#")
        print("#"*80)
        print(f"\nConfiguration:")
        print(f"  Cache Size: {self.cfg.CACHE_SIZE}")
        print(f"  Num Files: {self.cfg.NUM_FILES}")
        print(f"  Num Users: {self.cfg.NUM_USERS}")
        print(f"  NOMA Pairing: {self.cfg.PAIRING_METHOD}")
        print(f"  Power Allocation: {self.cfg.POWER_ALLOC_METHOD}")
        print(f"  CIC Enabled: {self.cfg.ENABLE_CIC}")
        print(f"  Target Rate: {self.cfg.TARGET_RATE_BPS:.2f} bps/Hz")
        print(f"\nDQN Hyperparameters:")
        print(f"  Learning Rate: {self.cfg.RL_LEARNING_RATE}")
        print(f"  Gamma: {self.cfg.RL_GAMMA}")
        print(f"  Epsilon: {self.cfg.RL_EPSILON_START} → {self.cfg.RL_EPSILON_END}")
        print(f"  Batch Size: {self.cfg.RL_BATCH_SIZE}")
        print(f"  Replay Buffer: {self.cfg.RL_REPLAY_BUFFER_SIZE}")
        print(f"  Prioritized Replay: {self.cfg.RL_USE_PRIORITIZED_REPLAY}")
        print("\n" + "="*80 + "\n")
    
    def create_dqn_cache(self, seed: Optional[int] = None) -> DQNCache:
        """
        Create a fresh DQN cache instance.
        
        Args:
            seed: Random seed for reproducibility
        
        Returns:
            DQNCache instance
        """
        if seed is None:
            seed = self.cfg.RANDOM_SEED
        
        return DQNCache(
            capacity=self.cfg.CACHE_SIZE,
            num_files=self.cfg.NUM_FILES,
            num_users=self.cfg.NUM_USERS,
            
            # Learning parameters
            learning_rate=self.cfg.RL_LEARNING_RATE,
            gamma=self.cfg.RL_GAMMA,
            epsilon_start=self.cfg.RL_EPSILON_START,
            epsilon_end=self.cfg.RL_EPSILON_END,
            epsilon_decay_steps=self.cfg.RL_EPSILON_DECAY_STEPS,
            
            # Network architecture
            use_neural_network=self.cfg.RL_USE_NEURAL_NETWORK,
            hidden_dims=self.cfg.RL_HIDDEN_DIMS,
            
            # Training parameters
            batch_size=self.cfg.RL_BATCH_SIZE,
            replay_buffer_size=self.cfg.RL_REPLAY_BUFFER_SIZE,
            train_freq=self.cfg.RL_TRAIN_FREQUENCY,
            warm_up_steps=self.cfg.RL_WARM_UP_STEPS,
            
            # Prioritized replay
            use_prioritized_replay=self.cfg.RL_USE_PRIORITIZED_REPLAY,
            priority_alpha=self.cfg.RL_PRIORITY_ALPHA,
            priority_beta_start=self.cfg.RL_PRIORITY_BETA_START,
            priority_beta_end=self.cfg.RL_PRIORITY_BETA_END,
            priority_beta_frames=self.cfg.RL_PRIORITY_BETA_FRAMES,
            
            # Stability
            gradient_clip=self.cfg.RL_GRADIENT_CLIP,
            tau=self.cfg.RL_TAU,
            
            # NOMA awareness
            enable_noma_awareness=True,
            
            seed=seed
        )
    
    def run_episode(self, cache: DQNCache, seed: int, 
                   episode_done: bool = False,
                   phase: str = 'train') -> Dict:
        """
        Run a single NOMA-aware episode.
        
        Args:
            cache: DQN cache instance
            seed: Random seed
            episode_done: Whether this is the last episode
            phase: 'train', 'test', or 'eval'
        
        Returns:
            Dictionary of metrics
        """
        set_seed(seed)
        
        # Initialize metrics
        metrics = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'noma_transmissions': 0,
            'noma_successes': 0,
            'noma_failures': 0,
            'outages': 0,
            'sic_attempts': 0,
            'sic_successes': 0,
            'cic_opportunities': 0,
            'cic_enabled_weak': 0,
            'cic_enabled_strong': 0,
            'total_throughput': 0.0,
            'total_energy': 0.0,
        }
        
        # ====================================================================
        # STEP 1: CHANNEL GENERATION (NOMA-Aware)
        # ====================================================================
        
        user_positions = generate_user_positions(
            self.cfg.NUM_USERS,
            self.cfg.CELL_RADIUS,
            seed=seed
        )
        
        channel_gains = compute_channel_gains(
            user_positions,
            exponent=self.cfg.PATHLOSS_EXPONENT,
            fading_type=self.cfg.FADING_TYPE,
            K_factor_db=self.cfg.RICIAN_K_FACTOR_DB,
            los_probability=self.cfg.LOS_PROBABILITY
        )
        
        # ====================================================================
        # STEP 2: GENERATE REQUESTS (Zipf Distribution)
        # ====================================================================
        
        total_requests = self.cfg.NUM_USERS * self.cfg.REQUESTS_PER_USER
        file_requests = sample_zipf_catalog(
            self.cfg.NUM_FILES,
            self.cfg.ZIPF_ALPHA,
            size=total_requests
        )
        
        requesting_users = np.random.choice(
            self.cfg.NUM_USERS,
            size=total_requests,
            replace=True
        )
        
        # ====================================================================
        # STEP 3: PROCESS REQUESTS WITH NOMA-AWARE CACHING
        # ====================================================================
        
        # Group by user
        user_requests = defaultdict(list)
        for file_id, user_id in zip(file_requests, requesting_users):
            user_requests[user_id].append(file_id)
        
        # Populate TopK caches if needed
        if isinstance(cache, StaticTopKCache) and len(cache) == 0:
            cnt = Counter(file_requests)
            ranking = [item for item, _ in cnt.most_common()]
            cache.populate(ranking)
        
        # Track cache misses (need NOMA transmission)
        miss_users = []
        miss_files = {}
        
        for user_id in range(self.cfg.NUM_USERS):
            if user_id not in user_requests:
                continue
            
            # Take first request
            file_id = user_requests[user_id][0]
            metrics['total_requests'] += 1
            
            # Check cache
            hit = cache.is_hit(file_id, update_stats=True)
            
            if hit:
                metrics['cache_hits'] += 1
                metrics['total_throughput'] += self.cfg.CACHE_DELIVERY_RATE
            else:
                metrics['cache_misses'] += 1
                miss_users.append(user_id)
                miss_files[user_id] = file_id
        
        if len(miss_users) == 0:
            # All hits!
            return self._compile_metrics(metrics, cache, phase)
        
        # ====================================================================
        # STEP 4: NOMA PAIRING
        # ====================================================================
        
        pairs, leftover = pair_users(
            miss_users,
            channel_gains,
            method=self.cfg.PAIRING_METHOD
        )
        
        sinr_threshold = sinr_threshold_from_rate(self.cfg.TARGET_RATE_BPS)
        
        # ====================================================================
        # STEP 5: SIMULATE NOMA TRANSMISSIONS WITH SIC/CIC
        # ====================================================================
        
        for weak_user, strong_user in pairs:
            self._process_noma_pair(
                cache=cache,
                weak_user=weak_user,
                strong_user=strong_user,
                weak_file=miss_files[weak_user],
                strong_file=miss_files[strong_user],
                channel_gains=channel_gains,
                sinr_threshold=sinr_threshold,
                metrics=metrics,
                episode_done=episode_done,
                phase=phase
            )
        
        # Handle leftover user
        if leftover is not None:
            self._process_single_user(
                cache=cache,
                user_id=leftover,
                file_id=miss_files[leftover],
                channel_gains=channel_gains,
                sinr_threshold=sinr_threshold,
                metrics=metrics,
                episode_done=episode_done,
                phase=phase
            )
        
        return self._compile_metrics(metrics, cache, phase)
    
    def _process_noma_pair(self, cache: DQNCache, weak_user: int, strong_user: int,
                       weak_file: int, strong_file: int,
                       channel_gains: np.ndarray, sinr_threshold: float,
                       metrics: Dict, episode_done: bool, phase: str):
        """
        Process NOMA pair transmission with SIC/CIC (NOMA-AWARE).
    
        ✅ FIXED (Dec 12, 2025): Correct CIC detection logic
    
        Cache-Aided Interference Cancellation (CIC) Logic:
        - Weak user can use CIC if STRONG user's file is cached (cancels strong's interference)
        - Strong user can use CIC if WEAK user's file is cached (perfect SIC decoding)
        - Power allocation uses OWN file cache status (priority assignment)
        - SIC/CIC simulation uses PAIRED file cache status (interference cancellation)

        Research References:
        - IEEE TWC 2022: "Cache-Aided NOMA Mobile Edge Computing"
        - arXiv:1712.09557: "Cache-Aided Non-Orthogonal Multiple Access"
        - IEEE JSAC 2019: "Power Allocation in Cache-Aided NOMA Systems"
        """
        gain_w = channel_gains[weak_user]
        gain_s = channel_gains[strong_user]

        metrics['noma_transmissions'] += 1

        # ========================================================================
        # ✅ CRITICAL FIX: Separate cache checks for power allocation vs CIC
        # ========================================================================

        # Cache status for OWN requested files (affects power allocation priority)
        weak_file_cached = cache.is_hit(weak_file, update_stats=False)
        strong_file_cached = cache.is_hit(strong_file, update_stats=False)

        # Cache status for PAIRED user's files (enables CIC capability)
        # KEY INSIGHT: User can cancel interference if interferer's content is cached
        weak_can_use_cic = cache.is_hit(strong_file, update_stats=False)  # Weak has strong's file
        strong_can_use_cic = cache.is_hit(weak_file, update_stats=False)  # Strong has weak's file

        # ========================================================================
        # Power Allocation (uses OWN file cache status)
        # ========================================================================
        # Cached users may get higher/lower power based on cache-aware strategy
        p_weak, p_strong, feasible, _ = allocate_power(
            gain_w=gain_w,
            gain_s=gain_s,
            cfg=self.cfg,
            method=self.cfg.POWER_ALLOC_METHOD,
            weak_cached=weak_file_cached,  # ✅ Weak user's own file status
            strong_cached=strong_file_cached,  # ✅ Strong user's own file status
            grid_points=self.cfg.POWER_ALLOC_GRID
        )
            
        # ========================================================================
        # Simulate SIC/CIC (uses PAIRED file cache status)
        # ========================================================================
        # CIC is enabled when user has interfering signal cached
        sic_results = simulate_sic_process(
            P_tx=self.cfg.TX_POWER,
            p_weak=p_weak,
            p_strong=p_strong,
            gain_w=gain_w,
            gain_s=gain_s,
            noise=self.cfg.NOISE_POWER,
            target_sinr=sinr_threshold,
            imperfection_factor=self.cfg.SIC_IMPERFECTION,
            weak_cached=weak_can_use_cic,  # ✅ Weak's CIC capability
            strong_cached=strong_can_use_cic  # ✅ Strong's CIC capability
        )

        weak_success = sic_results['weak_success']
        strong_success = sic_results['strong_success']

        # ========================================================================
        # Update SIC Metrics
        # ========================================================================
        metrics['sic_attempts'] += 1
        if sic_results['can_decode_weak']:
            metrics['sic_successes'] += 1

        # ========================================================================
        # ✅ FIXED CIC Tracking (uses correct cache status)
        # ========================================================================
        # Track when CIC is actually beneficial (user succeeds with CIC help)
        if weak_can_use_cic:
            metrics['cic_opportunities'] += 1
            if weak_success:
                metrics['cic_enabled_weak'] += 1

        if strong_can_use_cic:
            metrics['cic_opportunities'] += 1
            if strong_success:
                metrics['cic_enabled_strong'] += 1

        # ========================================================================
        # Transmission Outcomes
        # ========================================================================
        if weak_success and strong_success:
            metrics['noma_successes'] += 1
        elif weak_success or strong_success:
            metrics['noma_successes'] += 1
            metrics['outages'] += 1  # One user failed
        else:
            metrics['noma_failures'] += 1
            metrics['outages'] += 2  # Both failed

        # ========================================================================
        # Throughput and Energy
        # ========================================================================
        metrics['total_throughput'] += sic_results['sum_rate']
        metrics['total_energy'] += self.cfg.TX_POWER * (p_weak + p_strong)

        # ========================================================================
        # DQN Learning (only during training phase)
        # ========================================================================
        if phase == 'train' and hasattr(cache, 'request'):
            # Weak user request (for DQN learning)
            cache.request(
                item=weak_file,
                user_id=weak_user,
                channel_gain=gain_w,
                paired_user=strong_user,
                paired_file=strong_file,
                noma_success=weak_success,
                outage=not weak_success,
                sinr_weak=sic_results['sinr_w'],
                sinr_strong=sic_results['sinr_s_after'],
                episode_done=episode_done
            )

            # Strong user request (for DQN learning)
            cache.request(
                item=strong_file,
                user_id=strong_user,
                channel_gain=gain_s,
               paired_user=weak_user,
               paired_file=weak_file,
               noma_success=strong_success,
               outage=not strong_success,
               sinr_weak=sic_results['sinr_w'],
               sinr_strong=sic_results['sinr_s_after'],
               episode_done=episode_done
           )

    
    def _process_single_user(self, cache: DQNCache, user_id: int, file_id: int,
                            channel_gains: np.ndarray, sinr_threshold: float,
                            metrics: Dict, episode_done: bool, phase: str):
        """
        Process single user transmission.
        """
        gain = channel_gains[user_id]
        
        sinr = self.cfg.TX_POWER * gain / self.cfg.NOISE_POWER
        success = sinr >= sinr_threshold
        
        if success:
            metrics['noma_successes'] += 1
            metrics['total_throughput'] += rate_from_sinr(sinr)
        else:
            metrics['noma_failures'] += 1
            metrics['outages'] += 1
        
        metrics['noma_transmissions'] += 1
        metrics['total_energy'] += self.cfg.TX_POWER
        
        # DQN Learning (only during training)
        if phase == 'train' and hasattr(cache, 'request'):
            cache.request(
                item=file_id,
                user_id=user_id,
                channel_gain=gain,
                noma_success=success,
                outage=not success,
                episode_done=episode_done
            )
    
    def _compile_metrics(self, metrics: Dict, cache, phase: str) -> Dict:
        """
        Compile final metrics for episode.
        """
        total_req = max(metrics['total_requests'], 1)
        total_noma = max(metrics['noma_transmissions'], 1)
        total_noma_users = max(total_noma * 2, 1)
        total_sic = max(metrics['sic_attempts'], 1)
        
        result = {
            # Cache performance
            'hit_rate': metrics['cache_hits'] / total_req,
            'miss_rate': metrics['cache_misses'] / total_req,
            
            # NOMA performance
            'outage_probability': metrics['outages'] / total_noma_users,
            'noma_success_rate': metrics['noma_successes'] / total_noma,
            
            # SIC performance
            'sic_success_rate': metrics['sic_successes'] / total_sic,
            
            # CIC performance (NOVEL)
            'cic_benefit_rate': (
                (metrics['cic_enabled_weak'] + metrics['cic_enabled_strong']) / total_noma_users 
                if total_noma_users > 0 else 0.0
            ),

            # Throughput
            'avg_throughput': metrics['total_throughput'] / total_req,
            'spectral_efficiency': metrics['total_throughput'] / total_noma,
            
            # Energy
            'energy_per_bit': metrics['total_energy'] / max(metrics['total_throughput'], 1),
            
            # Raw counts
            'total_requests': metrics['total_requests'],
            'cache_hits': metrics['cache_hits'],
            'noma_transmissions': metrics['noma_transmissions'],
            'outages': metrics['outages'],
            'cic_events': metrics['cic_opportunities'],
            
            # Phase
            'phase': phase,
        }
        
        # Add DQN stats
        if hasattr(cache, 'get_stats'):
            dqn_stats = cache.get_stats()
            result.update(dqn_stats)
        
        return result
    
    def train(self, num_episodes: int, test_interval: int = 50,
             save_best: bool = True) -> Tuple[DQNCache, pd.DataFrame]:
        """
        Train DQN cache with periodic testing.
        
        Args:
            num_episodes: Number of training episodes
            test_interval: Test every N episodes
            save_best: Save best model checkpoint
        
        Returns:
            Tuple of (trained_cache, training_history_df)
        """
        if not HAS_DQN:
            raise ImportError("DQN not available")
        
        print(f"\n{'='*80}")
        print(f"TRAINING PHASE: {num_episodes} episodes")
        print(f"{'='*80}\n")
        
        # Create DQN cache
        cache = self.create_dqn_cache()
        
        # Training loop
        for episode in range(num_episodes):
            seed = self.cfg.RANDOM_SEED + episode
            episode_done = (episode == num_episodes - 1)
            
            # Training episode
            result = self.run_episode(
                cache, 
                seed, 
                episode_done=episode_done,
                phase='train'
            )
            result['episode'] = episode
            self.train_history.append(result)
            
            # Periodic testing
            if (episode + 1) % test_interval == 0:
                test_result = self._test_cache(cache, episode)
                self.test_history.append(test_result)
                
                # Save best model
                if save_best and test_result['hit_rate'] > self.best_hit_rate:
                    self.best_hit_rate = test_result['hit_rate']
                    self.best_episode = episode
                    self._save_checkpoint(cache, episode, test_result)
                
                if self.verbose:
                    self._print_progress(episode, result, test_result)
            elif self.verbose and (episode + 1) % 10 == 0:
                print(f"  Episode {episode+1}/{num_episodes}: "
                      f"Hit={result['hit_rate']:.3f}, "
                      f"ε={result.get('epsilon', 0):.3f}, "
                      f"Loss={result.get('avg_loss', 0):.4f}")
        
        print(f"\n✅ Training complete!")
        print(f"   Best hit rate: {self.best_hit_rate:.4f} (episode {self.best_episode})")
        
        # Save final model
        self._save_checkpoint(cache, num_episodes - 1, 
                            self.train_history[-1], final=True)
        
        return cache, pd.DataFrame(self.train_history)
    
    def _test_cache(self, cache: DQNCache, episode: int) -> Dict:
        """
        Test cache on validation set.
        """
        # Set evaluation mode (no learning, no exploration)
        cache.set_eval_mode(True)
        
        # Run test episode
        seed = self.cfg.RANDOM_SEED + 100000 + episode  # Different seed space
        result = self.run_episode(cache, seed, episode_done=False, phase='test')
        result['episode'] = episode
        
        # Restore training mode
        cache.set_eval_mode(False)
        
        return result
    
    def _print_progress(self, episode: int, train_result: Dict, test_result: Dict):
        """
        Print training progress.
        """
        print(f"\n  Episode {episode+1}:")
        print(f"    TRAIN: Hit={train_result['hit_rate']:.4f}, "
              f"Outage={train_result['outage_probability']:.4f}, "
              f"CIC={train_result['cic_benefit_rate']:.4f}")
        print(f"    TEST:  Hit={test_result['hit_rate']:.4f}, "
              f"Outage={test_result['outage_probability']:.4f}, "
              f"CIC={test_result['cic_benefit_rate']:.4f}")
        print(f"    DQN:   ε={train_result.get('epsilon', 0):.4f}, "
              f"Loss={train_result.get('avg_loss', 0):.4f}, "
              f"β={train_result.get('beta', 0):.4f}")
    
    def _save_checkpoint(self, cache: DQNCache, episode: int, 
                        result: Dict, final: bool = False):
        """
        Save model checkpoint.
        """
        if final:
            filename = f'dqn_cache_final.pth'
        else:
            filename = f'dqn_cache_best_ep{episode}.pth'
        
        filepath = os.path.join(self.save_dir, filename)
        cache.save_model(filepath)
        
        # Save metadata
        metadata = {
            'episode': episode,
            'hit_rate': result['hit_rate'],
            'outage_probability': result['outage_probability'],
            'cic_benefit_rate': result['cic_benefit_rate'],
            'timestamp': datetime.now().isoformat()
        }
        
        meta_file = filepath.replace('.pth', '_meta.json')
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        if self.verbose and not final:
            print(f"    ✅ Checkpoint saved: {filename}")


# ============================================================================
# EVALUATION SYSTEM
# ============================================================================

class CachePolicyEvaluator:
    """
    Evaluate and compare different caching policies.
    
    Supports:
    - Trained DQN (loaded from checkpoint)
    - Baseline policies (TopK, LRU, LFU, Random)
    - Fair comparison on held-out evaluation set
    """
    
    def __init__(self, cfg, verbose: bool = True):
        self.cfg = cfg
        self.verbose = verbose
        self.trainer = NOMADQNTrainer(cfg, verbose=False)
    
    def evaluate_policy(self, policy: str, num_runs: int,
                       pretrained_cache: Optional[DQNCache] = None) -> pd.DataFrame:
        """
        Evaluate a single caching policy.
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Evaluating {policy.upper()} Policy ({num_runs} runs)")
            print(f"{'='*70}")
        
        # Create cache
        if policy == 'dqn':
            if pretrained_cache is None:
                raise ValueError("DQN policy requires pretrained_cache")
            cache = pretrained_cache
            cache.set_eval_mode(True)  # No exploration, no learning
        else:
            cache = create_cache(policy, capacity=self.cfg.CACHE_SIZE)
        
        # Run evaluations
        results = []
        for run in range(num_runs):
            # Use different seed space for evaluation (never seen before)
            seed = self.cfg.RANDOM_SEED + 200000 + run
            
            result = self.trainer.run_episode(
                cache, 
                seed, 
                episode_done=False,
                phase='eval'
            )
            result['policy'] = policy
            result['run'] = run
            results.append(result)
            
            if self.verbose and (run + 1) % 10 == 0:
                print(f"  Run {run+1}/{num_runs}: "
                      f"Hit={result['hit_rate']:.3f}, "
                      f"Outage={result['outage_probability']:.3f}")
        
        return pd.DataFrame(results)
    
    def compare_all_policies(self, num_runs: int,
                            pretrained_dqn: Optional[DQNCache] = None) -> pd.DataFrame:
        """
        Compare all caching policies.
        """
        print(f"\n{'#'*80}")
        print(f"#" + " "*20 + "CACHING POLICY COMPARISON" + " "*23 + "#")
        print(f"\n{'#'*80}")
        
        policies = ['topk', 'lru', 'lfu', 'random']
        if pretrained_dqn is not None:
            policies.append('dqn')
        
        all_results = []
        
        for policy in policies:
            df = self.evaluate_policy(
                policy, 
                num_runs,
                pretrained_cache=pretrained_dqn if policy == 'dqn' else None
            )
            all_results.append(df)
        
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Print summary
        self._print_comparison_summary(combined_df)
        
        return combined_df
    
    def _print_comparison_summary(self, df: pd.DataFrame):
        """
        Print comparison summary statistics.
        """
        print(f"\n{'='*80}")
        print("COMPARISON SUMMARY")
        print(f"{'='*80}\n")
        
        summary = df.groupby('policy').agg({
            'hit_rate': ['mean', 'std'],
            'outage_probability': ['mean', 'std'],
            'cic_benefit_rate': ['mean', 'std'],
            'spectral_efficiency': ['mean', 'std']
        }).round(4)
        
        print(summary)
        print()


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_curves(train_df: pd.DataFrame, test_df: pd.DataFrame,
                        save_path: Optional[str] = None):
    """
    Plot DQN training curves.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('DQN Training Progress', fontsize=16)
    
    # Hit rate
    axes[0, 0].plot(train_df['episode'], train_df['hit_rate'], label='Train')
    if len(test_df) > 0:
        axes[0, 0].plot(test_df['episode'], test_df['hit_rate'], 
                       label='Test', marker='o')
    axes[0, 0].set_title('Cache Hit Rate')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Hit Rate')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Epsilon decay
    if 'epsilon' in train_df.columns:
        axes[0, 1].plot(train_df['episode'], train_df['epsilon'])
        axes[0, 1].set_title('Exploration Rate')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Epsilon')
        axes[0, 1].grid(True)
    
    # Loss
    if 'avg_loss' in train_df.columns:
        axes[0, 2].plot(train_df['episode'], train_df['avg_loss'])
        axes[0, 2].set_title('Average Loss')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].grid(True)
    
    # Outage probability
    axes[1, 0].plot(train_df['episode'], train_df['outage_probability'], 
                   label='Train')
    if len(test_df) > 0:
        axes[1, 0].plot(test_df['episode'], test_df['outage_probability'],
                       label='Test', marker='o')
    axes[1, 0].set_title('Outage Probability')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Outage Probability')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # CIC benefit rate
    axes[1, 1].plot(train_df['episode'], train_df['cic_benefit_rate'],
                   label='Train')
    if len(test_df) > 0:
        axes[1, 1].plot(test_df['episode'], test_df['cic_benefit_rate'],
                       label='Test', marker='o')
    axes[1, 1].set_title('CIC Benefit Rate')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('CIC Rate')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Spectral efficiency
    axes[1, 2].plot(train_df['episode'], train_df['spectral_efficiency'],
                   label='Train')
    if len(test_df) > 0:
        axes[1, 2].plot(test_df['episode'], test_df['spectral_efficiency'],
                       label='Test', marker='o')
    axes[1, 2].set_title('Spectral Efficiency')
    axes[1, 2].set_xlabel('Episode')
    axes[1, 2].set_ylabel('Efficiency (bps/Hz)')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Training curves saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_policy_comparison(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Plot comparison of caching policies.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Caching Policy Comparison', fontsize=16)
    
    metrics = [
        ('hit_rate', 'Cache Hit Rate'),
        ('outage_probability', 'Outage Probability'),
        ('cic_benefit_rate', 'CIC Benefit Rate'),
        ('spectral_efficiency', 'Spectral Efficiency'),
        ('energy_per_bit', 'Energy per Bit'),
        ('avg_throughput', 'Average Throughput')
    ]
    
    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        sns.boxplot(data=df, x='policy', y=metric, ax=ax)
        ax.set_title(title)
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Comparison plot saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    if not HAS_DQN:
        print("❌ Error: DQN not available. Exiting.")
        exit(1)
    
    from src import config as cfg
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    t0 = time.time()
    
    # ========================================================================
    # PHASE 1: TRAIN DQN
    # ========================================================================
    
    trainer = NOMADQNTrainer(cfg)
    
    trained_cache, train_history = trainer.train(
        num_episodes=cfg.RL_TRAINING_EPISODES,
        test_interval=50,
        save_best=True
    )
    
    # Save training history
    train_history.to_csv('results/dqn_training_history.csv', index=False)
    test_history = pd.DataFrame(trainer.test_history)
    if len(test_history) > 0:
        test_history.to_csv('results/dqn_test_history.csv', index=False)
    
    # Plot training curves
    plot_training_curves(
        train_history, 
        test_history,
        save_path='results/dqn_training_curves.png'
    )
    
    # ========================================================================
    # PHASE 2: EVALUATE & COMPARE
    # ========================================================================
    
    evaluator = CachePolicyEvaluator(cfg)
    
    comparison_results = evaluator.compare_all_policies(
        num_runs=cfg.NUM_RUNS,
        pretrained_dqn=trained_cache
    )
    
    # Save results
    comparison_results.to_csv('results/policy_comparison.csv', index=False)
    
    # Plot comparison
    plot_policy_comparison(
        comparison_results,
        save_path='results/policy_comparison.png'
    )
    
    # ========================================================================
    # PHASE 3: FINAL SUMMARY
    # ========================================================================
    
    print(f"\n{'#'*80}")
    print(f"#" + " "*25 + "SIMULATION COMPLETE" + " "*24 + "#")
    print(f"\n{'#'*80}")
    print(f"\nTotal time: {time.time() - t0:.2f}s")
    print(f"\nResults saved to:")
    print(f"  • results/dqn_training_history.csv")
    print(f"  • results/dqn_training_curves.png")
    print(f"  • results/policy_comparison.csv")
    print(f"  • results/policy_comparison.png")
    print(f"  • models/dqn_cache/dqn_cache_final.pth")
    print(f"\n✅ All files integrated properly with NOMA characteristics!\n")
