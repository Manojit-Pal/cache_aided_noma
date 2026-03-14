"""
src/simulation/stable_dqn_sim.py

Stable NOMA-Aware DQN Cache Simulator with Train-Test-Eval Workflow

Bug Fix History (2026 Audit):
- BUG-4    (HIGH):     episode_done only True on very last training episode
- BUG-SIM-1 (HIGH):   DQN double-counts stats
- BUG-SIM-2 (CRITICAL): 98% of requests silently dropped
- BUG-SIM-4 (MEDIUM): cache.request() hit result not fed back to metrics
- BUG-SIM-5 (HIGH):   episode_done forwarded to every request
- ROOT-3 (CRITICAL):  paired_file never passed to cache.request()
- FIX-1 (CRITICAL):   evaluate_policy() routed DQN through run_batch_episode()
- FIX-2 (HIGH):       _process_single_request() blocked cache.request() in eval
- BUG-A (HIGH):       DQN hits still went through NOMA (miss_for_noma=True always)
- BUG-C (HIGH):       _test_cache() phase='test' -> DQN fell into baseline path
- BUG-D (MEDIUM):     episode_done=True could never fire when users had 0 requests
- BUG-E (LOW):        outage_probability denominator mismatch between episode paths
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

from src import config
from src.utils import set_seed, sample_zipf_catalog

from src.noma import (
    generate_user_positions,
    compute_channel_gains,
    pair_users,
    allocate_power,
    simulate_sic_process,
    sinr_threshold_from_rate,
    rate_from_sinr
)

from src.caching import (
    create_cache,
    CacheBase,
    StaticTopKCache,
    LRUCache,
    LFUCache,
    RandomCache
)

try:
    from src.caching import DQNCache
    HAS_DQN = True
except ImportError:
    HAS_DQN = False
    print("Warning: DQN cache not available")


class NOMADQNTrainer:
    """
    NOMA-Aware DQN Training System with proper train/test/eval separation.
    """

    def __init__(self, cfg, verbose: bool = True):
        self.cfg = cfg
        self.verbose = verbose
        self.save_dir = 'models/dqn_cache'
        os.makedirs(self.save_dir, exist_ok=True)
        self.train_history: List[Dict] = []
        self.test_history:  List[Dict] = []
        self.best_hit_rate = 0.0
        self.best_episode  = 0
        if self.verbose:
            self._print_header()

    def _print_header(self):
        print("\n" + "#"*80)
        print("#" + " "*20 + "NOMA-AWARE DQN TRAINING SYSTEM" + " "*21 + "#")
        print("#"*80)
        print(f"\nConfiguration:")
        print(f"  Cache Size        : {self.cfg.CACHE_SIZE}")
        print(f"  Num Files         : {self.cfg.NUM_FILES}")
        print(f"  Num Users         : {self.cfg.NUM_USERS}")
        print(f"  Requests/User     : {self.cfg.REQUESTS_PER_USER}")
        print(f"  Steps/Episode     : {self.cfg.NUM_USERS * self.cfg.REQUESTS_PER_USER}")
        print(f"  NOMA Pairing      : {self.cfg.PAIRING_METHOD}")
        print(f"  Power Allocation  : {self.cfg.POWER_ALLOC_METHOD}")
        print(f"  CIC Enabled       : {self.cfg.ENABLE_CIC}")
        print(f"  Target Rate       : {self.cfg.TARGET_RATE_BPS:.2f} bps/Hz")
        print(f"\nDQN Hyperparameters:")
        print(f"  Learning Rate     : {self.cfg.RL_LEARNING_RATE}")
        print(f"  Gamma             : {self.cfg.RL_GAMMA}")
        print(f"  Epsilon           : {self.cfg.RL_EPSILON_START} -> {self.cfg.RL_EPSILON_END}")
        print(f"  Epsilon Decay     : over {self.cfg.RL_EPSILON_DECAY_STEPS:,} steps")
        print(f"  Batch Size        : {self.cfg.RL_BATCH_SIZE}")
        print(f"  Replay Buffer     : {self.cfg.RL_REPLAY_BUFFER_SIZE:,}")
        print(f"  Warm-up Steps     : {self.cfg.RL_WARM_UP_STEPS:,}")
        print(f"  Train Frequency   : every {self.cfg.RL_TRAIN_FREQUENCY} steps")
        print(f"  Prioritized Replay: {self.cfg.RL_USE_PRIORITIZED_REPLAY}")
        print("\n" + "="*80 + "\n")

    def create_dqn_cache(self, seed: Optional[int] = None) -> 'DQNCache':
        if seed is None:
            seed = self.cfg.RANDOM_SEED
        return DQNCache(
            capacity=self.cfg.CACHE_SIZE,
            num_files=self.cfg.NUM_FILES,
            num_users=self.cfg.NUM_USERS,
            learning_rate=self.cfg.RL_LEARNING_RATE,
            gamma=self.cfg.RL_GAMMA,
            epsilon_start=self.cfg.RL_EPSILON_START,
            epsilon_end=self.cfg.RL_EPSILON_END,
            epsilon_decay_steps=self.cfg.RL_EPSILON_DECAY_STEPS,
            use_neural_network=self.cfg.RL_USE_NEURAL_NETWORK,
            hidden_dims=self.cfg.RL_HIDDEN_DIMS,
            batch_size=self.cfg.RL_BATCH_SIZE,
            replay_buffer_size=self.cfg.RL_REPLAY_BUFFER_SIZE,
            train_freq=self.cfg.RL_TRAIN_FREQUENCY,
            warm_up_steps=self.cfg.RL_WARM_UP_STEPS,
            use_prioritized_replay=self.cfg.RL_USE_PRIORITIZED_REPLAY,
            priority_alpha=self.cfg.RL_PRIORITY_ALPHA,
            priority_beta_start=self.cfg.RL_PRIORITY_BETA_START,
            priority_beta_end=self.cfg.RL_PRIORITY_BETA_END,
            priority_beta_frames=self.cfg.RL_PRIORITY_BETA_FRAMES,
            gradient_clip=self.cfg.RL_GRADIENT_CLIP,
            tau=self.cfg.RL_TAU,
            enable_noma_awareness=True,
            seed=seed,
        )

    # ========================================================================
    # EPISODE RUNNER
    # BUG-A FIX: peek with is_hit(update_stats=False) to set miss_for_noma;
    #            cache.request() remains the single authoritative call.
    # BUG-D FIX: episode_done derived from flat request index, not per-user.
    # ========================================================================

    def run_episode(self, cache, seed: int, phase: str = 'train') -> Dict:
        """
        Run one NOMA-aware episode.

        BUG-A fix:
            Previously miss_for_noma=True was set unconditionally for DQN,
            causing every hit to still trigger a NOMA transmission call.
            Now we peek with is_hit(update_stats=False) to decide whether
            to skip to the next request. cache.request() is still the
            single authoritative stat-recording + learning call for DQN.

        BUG-D fix:
            episode_done is now derived from a flat enumeration of all
            (user_id, file_id) pairs, so it fires correctly on the very
            last request regardless of per-user request distribution.
        """
        set_seed(seed)

        is_dqn = isinstance(cache, DQNCache)
        is_dqn_train = is_dqn and phase == 'train'
        is_dqn_eval  = is_dqn and (phase == 'eval')

        metrics = {
            'total_requests':     0,
            'cache_hits':         0,
            'cache_misses':       0,
            'noma_transmissions': 0,
            'noma_successes':     0,
            'noma_failures':      0,
            'outages':            0,
            'sic_attempts':       0,
            'sic_successes':      0,
            'cic_opportunities':  0,
            'cic_enabled_weak':   0,
            'cic_enabled_strong': 0,
            'total_throughput':   0.0,
            'total_energy':       0.0,
        }

        user_positions = generate_user_positions(
            self.cfg.NUM_USERS, self.cfg.CELL_RADIUS, seed=seed)
        channel_gains = compute_channel_gains(
            user_positions,
            exponent=self.cfg.PATHLOSS_EXPONENT,
            fading_type=self.cfg.FADING_TYPE,
            K_factor_db=self.cfg.RICIAN_K_FACTOR_DB,
            los_probability=self.cfg.LOS_PROBABILITY,
        )

        total_requests_in_episode = self.cfg.NUM_USERS * self.cfg.REQUESTS_PER_USER
        file_requests = sample_zipf_catalog(
            self.cfg.NUM_FILES, self.cfg.ZIPF_ALPHA,
            size=total_requests_in_episode)
        requesting_users = np.random.choice(
            self.cfg.NUM_USERS, size=total_requests_in_episode, replace=True)

        user_requests: Dict[int, List[int]] = defaultdict(list)
        for file_id, user_id in zip(file_requests, requesting_users):
            user_requests[int(user_id)].append(int(file_id))

        if isinstance(cache, StaticTopKCache) and len(cache) == 0:
            cnt     = Counter(file_requests)
            ranking = [item for item, _ in cnt.most_common()]
            cache.populate(ranking)

        all_users = list(range(self.cfg.NUM_USERS))
        episode_pairs, episode_leftover = pair_users(
            all_users, channel_gains, method=self.cfg.PAIRING_METHOD)

        user_partner: Dict[int, int] = {}
        for weak_u, strong_u in episode_pairs:
            user_partner[weak_u]   = strong_u
            user_partner[strong_u] = weak_u
        if episode_leftover is not None:
            user_partner[episode_leftover] = -1

        sinr_threshold = sinr_threshold_from_rate(self.cfg.TARGET_RATE_BPS)

        # BUG-D FIX: build a flat ordered list of (user_id, file_id) pairs
        # so episode_done fires correctly on the LAST actual request.
        flat_requests: List[Tuple[int, int]] = []
        for user_id in range(self.cfg.NUM_USERS):
            for file_id in user_requests.get(user_id, []):
                flat_requests.append((user_id, int(file_id)))
        total_flat = len(flat_requests)

        for req_idx, (user_id, file_id) in enumerate(flat_requests):
            is_last_request = (req_idx == total_flat - 1) and (phase == 'train')
            partner_id = user_partner.get(user_id, -1)

            paired_file = None
            if partner_id >= 0 and partner_id in user_requests:
                partner_reqs = user_requests[partner_id]
                if partner_reqs:
                    paired_file = partner_reqs[0]

            metrics['total_requests'] += 1

            # BUG-A FIX: peek to decide whether to skip NOMA for DQN hits.
            # For DQN: peek without updating stats (cache.request() owns stats).
            # For baselines: is_hit() IS the authoritative stat call.
            if is_dqn_train or is_dqn_eval:
                is_already_cached = cache.is_hit(file_id, update_stats=False)
                if is_already_cached:
                    # Let cache.request() handle the hit (stats + learning)
                    result = cache.request(
                        item=file_id,
                        user_id=user_id,
                        channel_gain=channel_gains[user_id],
                        noma_success=True,
                        outage=False,
                        episode_done=(is_last_request if is_dqn_train else False),
                        paired_file=paired_file,
                        paired_user=partner_id if partner_id >= 0 else None,
                    )
                    metrics['cache_hits']       += 1
                    metrics['total_throughput'] += self.cfg.CACHE_DELIVERY_RATE
                    if result.get('cic_enabled', False):
                        metrics['cic_enabled_strong'] += 1
                    continue  # No NOMA transmission needed
                # else: cache miss -> fall through to NOMA processing below
                miss_for_noma = True
            else:
                should_update = (phase == 'train')
                hit = cache.is_hit(file_id, update_stats=should_update)
                if hit:
                    metrics['cache_hits']       += 1
                    metrics['total_throughput'] += self.cfg.CACHE_DELIVERY_RATE
                    continue
                else:
                    metrics['cache_misses'] += 1
                    miss_for_noma = True

            # NOMA transmission for cache miss
            hit_result = self._process_single_request(
                cache=cache,
                user_id=user_id,
                file_id=file_id,
                paired_file=paired_file,
                paired_user=partner_id if partner_id >= 0 else None,
                channel_gains=channel_gains,
                sinr_threshold=sinr_threshold,
                metrics=metrics,
                episode_done=is_last_request,
                phase=phase,
                is_dqn_train=is_dqn_train,
                is_dqn_eval=is_dqn_eval,
            )

            # BUG-SIM-4 fix: accumulate DQN miss result from cache.request()
            if (is_dqn_train or is_dqn_eval) and hit_result is not None:
                if hit_result:
                    metrics['cache_hits']       += 1
                    metrics['total_throughput'] += self.cfg.CACHE_DELIVERY_RATE
                else:
                    metrics['cache_misses'] += 1

        # BUG-E FIX: pass flag so _compile_metrics uses correct denominator
        return self._compile_metrics(metrics, cache, phase,
                                     single_user_per_noma=True)

    # ========================================================================
    # SINGLE-REQUEST NOMA HANDLER
    # ========================================================================

    def _process_single_request(
        self, cache, user_id: int, file_id: int,
        channel_gains: np.ndarray, sinr_threshold: float,
        metrics: Dict, episode_done: bool, phase: str,
        is_dqn_train: bool,
        is_dqn_eval: bool = False,
        paired_file: Optional[int] = None,
        paired_user: Optional[int] = None,
    ) -> Optional[bool]:
        gain    = channel_gains[user_id]
        sinr    = self.cfg.TX_POWER * gain / self.cfg.NOISE_POWER
        success = sinr >= sinr_threshold

        if success:
            metrics['noma_successes']   += 1
            metrics['total_throughput'] += rate_from_sinr(sinr)
        else:
            metrics['noma_failures'] += 1
            metrics['outages']       += 1

        metrics['noma_transmissions'] += 1
        metrics['total_energy']       += self.cfg.TX_POWER

        if paired_file is not None:
            metrics['cic_opportunities'] += 1
            if cache.is_hit(paired_file, update_stats=False):
                metrics['cic_enabled_weak'] += 1

        hit = None
        if hasattr(cache, 'request') and (is_dqn_train or is_dqn_eval):
            safe_episode_done = episode_done if is_dqn_train else False
            result = cache.request(
                item=file_id,
                user_id=user_id,
                channel_gain=gain,
                noma_success=success,
                outage=not success,
                episode_done=safe_episode_done,
                paired_file=paired_file,
                paired_user=paired_user,
            )
            hit = result['hit']
            if result.get('cic_enabled', False):
                metrics['cic_enabled_strong'] += 1

        return hit

    # ========================================================================
    # BATCH EPISODE RUNNER (baselines only)
    # ========================================================================

    def run_batch_episode(self, cache, seed: int, phase: str = 'eval') -> Dict:
        """Run one episode with full NOMA pairing (for baseline/eval policies only)."""
        set_seed(seed)

        metrics = {
            'total_requests':     0,
            'cache_hits':         0,
            'cache_misses':       0,
            'noma_transmissions': 0,
            'noma_successes':     0,
            'noma_failures':      0,
            'outages':            0,
            'sic_attempts':       0,
            'sic_successes':      0,
            'cic_opportunities':  0,
            'cic_enabled_weak':   0,
            'cic_enabled_strong': 0,
            'total_throughput':   0.0,
            'total_energy':       0.0,
        }

        user_positions = generate_user_positions(
            self.cfg.NUM_USERS, self.cfg.CELL_RADIUS, seed=seed)
        channel_gains = compute_channel_gains(
            user_positions,
            exponent=self.cfg.PATHLOSS_EXPONENT,
            fading_type=self.cfg.FADING_TYPE,
            K_factor_db=self.cfg.RICIAN_K_FACTOR_DB,
            los_probability=self.cfg.LOS_PROBABILITY,
        )

        total_requests = self.cfg.NUM_USERS * self.cfg.REQUESTS_PER_USER
        file_requests  = sample_zipf_catalog(
            self.cfg.NUM_FILES, self.cfg.ZIPF_ALPHA, size=total_requests)
        requesting_users = np.random.choice(
            self.cfg.NUM_USERS, size=total_requests, replace=True)

        user_requests: Dict[int, List[int]] = defaultdict(list)
        for file_id, user_id in zip(file_requests, requesting_users):
            user_requests[int(user_id)].append(int(file_id))

        if isinstance(cache, StaticTopKCache) and len(cache) == 0:
            cnt = Counter(file_requests)
            cache.populate([item for item, _ in cnt.most_common()])

        miss_users: List[int] = []
        miss_files: Dict[int, int] = {}

        for user_id in range(self.cfg.NUM_USERS):
            if user_id not in user_requests:
                continue
            file_id = user_requests[user_id][0]
            metrics['total_requests'] += 1
            hit = cache.is_hit(file_id, update_stats=(phase == 'train'))
            if hit:
                metrics['cache_hits']       += 1
                metrics['total_throughput'] += self.cfg.CACHE_DELIVERY_RATE
            else:
                metrics['cache_misses'] += 1
                miss_users.append(user_id)
                miss_files[user_id] = file_id

        if miss_users:
            sinr_threshold = sinr_threshold_from_rate(self.cfg.TARGET_RATE_BPS)
            pairs, leftover = pair_users(
                miss_users, channel_gains, method=self.cfg.PAIRING_METHOD)
            for weak_user, strong_user in pairs:
                self._process_noma_pair(
                    cache=cache,
                    weak_user=weak_user,  strong_user=strong_user,
                    weak_file=miss_files[weak_user],
                    strong_file=miss_files[strong_user],
                    channel_gains=channel_gains,
                    sinr_threshold=sinr_threshold,
                    metrics=metrics,
                    episode_done=False,
                    phase=phase,
                )
            if leftover is not None:
                self._process_single_request(
                    cache=cache, user_id=leftover,
                    file_id=miss_files[leftover],
                    channel_gains=channel_gains,
                    sinr_threshold=sinr_threshold,
                    metrics=metrics, episode_done=False,
                    phase=phase, is_dqn_train=False, is_dqn_eval=False,
                )

        # run_batch_episode: noma_transmissions = num_pairs (2 users each)
        # so denominator must be total_noma*2 (BUG-E: keep original behaviour)
        return self._compile_metrics(metrics, cache, phase,
                                     single_user_per_noma=False)

    # ========================================================================
    # NOMA PAIR HANDLER (used by run_batch_episode)
    # ========================================================================

    def _process_noma_pair(
        self, cache, weak_user: int, strong_user: int,
        weak_file: int, strong_file: int,
        channel_gains: np.ndarray, sinr_threshold: float,
        metrics: Dict, episode_done: bool, phase: str
    ):
        gain_w = channel_gains[weak_user]
        gain_s = channel_gains[strong_user]
        metrics['noma_transmissions'] += 1

        weak_file_cached   = cache.is_hit(weak_file,   update_stats=False)
        strong_file_cached = cache.is_hit(strong_file, update_stats=False)
        weak_can_use_cic   = cache.is_hit(strong_file, update_stats=False)
        strong_can_use_cic = cache.is_hit(weak_file,   update_stats=False)

        p_weak, p_strong, feasible, _ = allocate_power(
            gain_w=gain_w, gain_s=gain_s, cfg=self.cfg,
            method=self.cfg.POWER_ALLOC_METHOD,
            weak_cached=weak_file_cached,
            strong_cached=strong_file_cached,
            grid_points=self.cfg.POWER_ALLOC_GRID,
        )

        sic_results = simulate_sic_process(
            P_tx=self.cfg.TX_POWER,
            p_weak=p_weak, p_strong=p_strong,
            gain_w=gain_w, gain_s=gain_s,
            noise=self.cfg.NOISE_POWER,
            target_sinr=sinr_threshold,
            imperfection_factor=self.cfg.SIC_IMPERFECTION,
            weak_cached=weak_can_use_cic,
            strong_cached=strong_can_use_cic,
        )

        weak_success   = sic_results['weak_success']
        strong_success = sic_results['strong_success']

        metrics['sic_attempts'] += 1
        if sic_results['can_decode_weak']:
            metrics['sic_successes'] += 1

        if weak_can_use_cic:
            metrics['cic_opportunities'] += 1
            if weak_success:
                metrics['cic_enabled_weak'] += 1
        if strong_can_use_cic:
            metrics['cic_opportunities'] += 1
            if strong_success:
                metrics['cic_enabled_strong'] += 1

        if weak_success and strong_success:
            metrics['noma_successes'] += 1
        elif weak_success or strong_success:
            metrics['noma_successes'] += 1
            metrics['outages']        += 1
        else:
            metrics['noma_failures'] += 1
            metrics['outages']       += 2

        metrics['total_throughput'] += sic_results['sum_rate']
        metrics['total_energy']     += self.cfg.TX_POWER * (p_weak + p_strong)

        if phase == 'train' and hasattr(cache, 'request'):
            cache.request(
                item=weak_file, user_id=weak_user, channel_gain=gain_w,
                paired_user=strong_user, paired_file=strong_file,
                noma_success=weak_success, outage=not weak_success,
                sinr_weak=sic_results['sinr_w'],
                sinr_strong=sic_results['sinr_s_after'],
                episode_done=episode_done,
            )
            cache.request(
                item=strong_file, user_id=strong_user, channel_gain=gain_s,
                paired_user=weak_user, paired_file=weak_file,
                noma_success=strong_success, outage=not strong_success,
                sinr_weak=sic_results['sinr_w'],
                sinr_strong=sic_results['sinr_s_after'],
                episode_done=episode_done,
            )

    # ========================================================================
    # METRICS COMPILER
    # BUG-E FIX: single_user_per_noma flag selects correct outage denominator
    # ========================================================================

    def _compile_metrics(
        self, metrics: Dict, cache, phase: str,
        single_user_per_noma: bool = False
    ) -> Dict:
        total_req  = max(metrics['total_requests'],     1)
        total_noma = max(metrics['noma_transmissions'], 1)
        total_sic  = max(metrics['sic_attempts'],       1)

        # BUG-E FIX:
        #   run_episode()     : 1 user per NOMA call -> denominator = total_noma
        #   run_batch_episode(): 2 users per NOMA pair -> denominator = total_noma*2
        if single_user_per_noma:
            outage_denom = max(total_noma, 1)
        else:
            outage_denom = max(total_noma * 2, 1)

        result = {
            'hit_rate':           metrics['cache_hits']     / total_req,
            'miss_rate':          metrics['cache_misses']   / total_req,
            'outage_probability': metrics['outages']        / outage_denom,
            'noma_success_rate':  metrics['noma_successes'] / total_noma,
            'sic_success_rate':   metrics['sic_successes']  / total_sic,
            'cic_benefit_rate': (
                (metrics['cic_enabled_weak'] + metrics['cic_enabled_strong'])
                / max(metrics['cic_opportunities'], 1)
            ),
            'avg_throughput':      metrics['total_throughput'] / total_req,
            'spectral_efficiency': metrics['total_throughput'] / total_noma,
            'energy_per_bit':      metrics['total_energy'] / max(metrics['total_throughput'], 1),
            'total_requests':      metrics['total_requests'],
            'cache_hits':          metrics['cache_hits'],
            'noma_transmissions':  metrics['noma_transmissions'],
            'outages':             metrics['outages'],
            'cic_events':          metrics['cic_opportunities'],
            'phase':               phase,
        }

        if hasattr(cache, 'get_stats'):
            result.update(cache.get_stats())

        return result

    # ========================================================================
    # TRAINING LOOP
    # ========================================================================

    def train(
        self, num_episodes: int,
        test_interval: int = 50,
        save_best: bool = True,
    ) -> Tuple['DQNCache', pd.DataFrame]:
        if not HAS_DQN:
            raise ImportError("DQN not available")

        print(f"\n{'='*80}")
        print(f"TRAINING PHASE: {num_episodes} episodes")
        print(f"{'='*80}\n")

        cache = self.create_dqn_cache()

        for episode in range(num_episodes):
            seed = self.cfg.RANDOM_SEED + episode
            result = self.run_episode(cache, seed, phase='train')
            result['episode'] = episode
            self.train_history.append(result)

            if (episode + 1) % test_interval == 0:
                test_result = self._test_cache(cache, episode)
                self.test_history.append(test_result)

                if save_best and test_result['hit_rate'] > self.best_hit_rate:
                    self.best_hit_rate = test_result['hit_rate']
                    self.best_episode  = episode
                    self._save_checkpoint(cache, episode, test_result)

                if self.verbose:
                    self._print_progress(episode, result, test_result)

            elif self.verbose and (episode + 1) % 10 == 0:
                print(f"  Episode {episode+1}/{num_episodes}: "
                      f"Hit={result['hit_rate']:.3f}, "
                      f"e={result.get('epsilon', 0):.3f}, "
                      f"Loss={result.get('avg_loss', 0):.4f}")

        print(f"\nTraining complete!")
        print(f"  Best hit rate: {self.best_hit_rate:.4f} (episode {self.best_episode})")
        self._save_checkpoint(cache, num_episodes - 1,
                              self.train_history[-1], final=True)

        return cache, pd.DataFrame(self.train_history)

    def _test_cache(self, cache: 'DQNCache', episode: int) -> Dict:
        """
        BUG-C FIX: use phase='eval' (not 'test') so DQN follows the
        is_dqn_eval=True path with cache.request() instead of falling
        into the baseline is_hit()-only path.
        """
        cache.set_eval_mode(True)
        seed   = self.cfg.RANDOM_SEED + episode + 10000
        result = self.run_episode(cache, seed, phase='eval')   # BUG-C fix
        result['episode'] = episode
        cache.set_eval_mode(False)
        return result

    def _print_progress(self, episode: int, train_result: Dict, test_result: Dict):
        print(f"\n  Episode {episode+1}:")
        print(f"    TRAIN: Hit={train_result['hit_rate']:.4f}, "
              f"Outage={train_result['outage_probability']:.4f}, "
              f"CIC={train_result['cic_benefit_rate']:.4f}")
        print(f"    TEST:  Hit={test_result['hit_rate']:.4f}, "
              f"Outage={test_result['outage_probability']:.4f}, "
              f"CIC={test_result['cic_benefit_rate']:.4f}")
        print(f"    DQN:   e={train_result.get('epsilon', 0):.4f}, "
              f"Loss={train_result.get('avg_loss', 0):.4f}, "
              f"b={train_result.get('beta', 0):.4f}")

    def _save_checkpoint(self, cache, episode: int,
                         result: Dict, final: bool = False):
        filename = 'dqn_cache_final.pth' if final else f'dqn_cache_best_ep{episode}.pth'
        filepath = os.path.join(self.save_dir, filename)
        cache.save_model(filepath)
        metadata = {
            'episode':            episode,
            'hit_rate':           result['hit_rate'],
            'outage_probability': result['outage_probability'],
            'cic_benefit_rate':   result['cic_benefit_rate'],
            'timestamp':          datetime.now().isoformat(),
        }
        with open(filepath.replace('.pth', '_meta.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        if self.verbose and not final:
            print(f"    Checkpoint saved: {filename}")


# ============================================================================
# EVALUATION SYSTEM
# ============================================================================

class CachePolicyEvaluator:
    """Evaluate and compare different caching policies on held-out data."""

    def __init__(self, cfg, verbose: bool = True):
        self.cfg     = cfg
        self.verbose = verbose
        self.trainer = NOMADQNTrainer(cfg, verbose=False)

    def evaluate_policy(
        self, policy: str, num_runs: int,
        pretrained_cache=None,
    ) -> pd.DataFrame:
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Evaluating {policy.upper()} Policy ({num_runs} runs)")
            print(f"{'='*70}")

        if policy == 'dqn':
            if pretrained_cache is None:
                raise ValueError("DQN policy requires pretrained_cache")
            cache = pretrained_cache
            cache.set_eval_mode(True)
        else:
            cache = create_cache(policy, capacity=self.cfg.CACHE_SIZE)

        results = []
        for run in range(num_runs):
            seed = self.cfg.RANDOM_SEED + 200000 + run

            if policy == 'dqn':
                result = self.trainer.run_episode(cache, seed, phase='eval')
            else:
                result = self.trainer.run_batch_episode(cache, seed, phase='eval')

            result['policy'] = policy
            result['run']    = run
            results.append(result)
            if self.verbose and (run + 1) % 10 == 0:
                print(f"  Run {run+1}/{num_runs}: "
                      f"Hit={result['hit_rate']:.3f}, "
                      f"Outage={result['outage_probability']:.3f}")

        return pd.DataFrame(results)

    def compare_all_policies(
        self, num_runs: int, pretrained_dqn=None
    ) -> pd.DataFrame:
        print(f"\n{'#'*80}")
        print("CACHING POLICY COMPARISON")
        print(f"{'#'*80}")

        policies = ['topk', 'lru', 'lfu', 'random']
        if pretrained_dqn is not None:
            policies.append('dqn')

        all_results = []
        for policy in policies:
            df = self.evaluate_policy(
                policy, num_runs,
                pretrained_cache=pretrained_dqn if policy == 'dqn' else None,
            )
            all_results.append(df)

        combined_df = pd.concat(all_results, ignore_index=True)
        self._print_comparison_summary(combined_df)
        return combined_df

    def _print_comparison_summary(self, df: pd.DataFrame):
        print(f"\n{'='*80}")
        print("COMPARISON SUMMARY")
        print(f"{'='*80}\n")
        summary = df.groupby('policy').agg({
            'hit_rate':            ['mean', 'std'],
            'outage_probability':  ['mean', 'std'],
            'cic_benefit_rate':    ['mean', 'std'],
            'spectral_efficiency': ['mean', 'std'],
        }).round(4)
        print(summary)
        print()


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_curves(
    train_df: pd.DataFrame, test_df: pd.DataFrame,
    save_path: Optional[str] = None
):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('DQN Training Progress', fontsize=16)

    axes[0, 0].plot(train_df['episode'], train_df['hit_rate'], label='Train')
    if len(test_df) > 0:
        axes[0, 0].plot(test_df['episode'], test_df['hit_rate'], label='Test', marker='o')
    axes[0, 0].set_title('Cache Hit Rate');  axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Hit Rate');       axes[0, 0].legend(); axes[0, 0].grid(True)

    if 'epsilon' in train_df.columns:
        axes[0, 1].plot(train_df['episode'], train_df['epsilon'])
        axes[0, 1].set_title('Exploration Rate'); axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Epsilon');         axes[0, 1].grid(True)

    if 'avg_loss' in train_df.columns:
        axes[0, 2].plot(train_df['episode'], train_df['avg_loss'])
        axes[0, 2].set_title('Average Loss'); axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Loss');        axes[0, 2].grid(True)

    axes[1, 0].plot(train_df['episode'], train_df['outage_probability'], label='Train')
    if len(test_df) > 0:
        axes[1, 0].plot(test_df['episode'], test_df['outage_probability'], label='Test', marker='o')
    axes[1, 0].set_title('Outage Probability'); axes[1, 0].legend(); axes[1, 0].grid(True)

    axes[1, 1].plot(train_df['episode'], train_df['cic_benefit_rate'], label='Train')
    if len(test_df) > 0:
        axes[1, 1].plot(test_df['episode'], test_df['cic_benefit_rate'], label='Test', marker='o')
    axes[1, 1].set_title('CIC Benefit Rate'); axes[1, 1].legend(); axes[1, 1].grid(True)

    axes[1, 2].plot(train_df['episode'], train_df['spectral_efficiency'], label='Train')
    if len(test_df) > 0:
        axes[1, 2].plot(test_df['episode'], test_df['spectral_efficiency'], label='Test', marker='o')
    axes[1, 2].set_title('Spectral Efficiency'); axes[1, 2].legend(); axes[1, 2].grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_policy_comparison(
    df: pd.DataFrame, save_path: Optional[str] = None
):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Caching Policy Comparison', fontsize=16)
    metrics = [
        ('hit_rate',            'Cache Hit Rate'),
        ('outage_probability',  'Outage Probability'),
        ('cic_benefit_rate',    'CIC Benefit Rate'),
        ('spectral_efficiency', 'Spectral Efficiency'),
        ('energy_per_bit',      'Energy per Bit'),
        ('avg_throughput',      'Average Throughput'),
    ]
    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        sns.boxplot(data=df, x='policy', y=metric, ax=ax)
        ax.set_title(title); ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved: {save_path}")
    else:
        plt.show()
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    if not HAS_DQN:
        print("Error: DQN not available. Exiting.")
        exit(1)

    from src import config as cfg

    os.makedirs('results', exist_ok=True)
    t0 = time.time()

    trainer = NOMADQNTrainer(cfg)
    trained_cache, train_history = trainer.train(
        num_episodes=cfg.RL_TRAINING_EPISODES,
        test_interval=50,
        save_best=True,
    )

    train_history.to_csv('results/dqn_training_history.csv', index=False)
    test_history = pd.DataFrame(trainer.test_history)
    if len(test_history) > 0:
        test_history.to_csv('results/dqn_test_history.csv', index=False)

    plot_training_curves(
        train_history, test_history,
        save_path='results/dqn_training_curves.png',
    )

    evaluator = CachePolicyEvaluator(cfg)
    comparison_results = evaluator.compare_all_policies(
        num_runs=cfg.NUM_RUNS, pretrained_dqn=trained_cache)
    comparison_results.to_csv('results/policy_comparison.csv', index=False)
    plot_policy_comparison(comparison_results, save_path='results/policy_comparison.png')

    print(f"\nTotal time: {time.time() - t0:.2f}s")
    print(f"Results saved to results/")
