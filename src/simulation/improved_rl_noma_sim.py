# src/simulation/improved_rl_noma_sim.py

import os
import time
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple

from src.utils import set_seed, sample_zipf_catalog
from src.noma import channel_model
from src.noma.power_allocation import allocate_power_gridsearch
from src.noma.noma_base import simulate_noma_pair
from scipy.special import erfc

# Import improved cache
try:
    from src.caching.improved_dqn_noma_cache import ImprovedDQNNomaCache
    USE_IMPROVED = True
except ImportError:
    print("⚠️ Using fallback RL cache")

# Optional TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except Exception:
    SummaryWriter = None
    TENSORBOARD_AVAILABLE = False


def compute_ber_bpsk(sinr):
    """Compute BER for BPSK modulation."""
    return 0.5 * erfc(np.sqrt(np.maximum(sinr, 0.0)))


def run_training_phase(
    cache,
    cfg,
    num_training_steps=50000,
    checkpoint_dir="./checkpoints",
    checkpoint_interval=10000,
    tb_writer=None,
    seed: int = None
):
    """
    FIXED TRAINING PHASE with:
    - Proper episode boundaries
    - Better reward tracking
    - Episode-based metrics
    """
    print(f"\n{'='*70}")
    print("TRAINING PHASE")
    print(f"{'='*70}")
    print(f"Training steps: {num_training_steps}")
    print(f"Epsilon decay: {cache.epsilon_start} → {cache.epsilon_end}")

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Generate training environment
    user_pos = channel_model.generate_user_positions(
        cfg.NUM_USERS, cfg.CELL_RADIUS, seed=seed if seed is not None else cfg.RANDOM_SEED
    )
    distances = user_pos[:, 2]
    pl = np.array([
        channel_model.pathloss(d, cfg.PATHLOSS_EXPONENT, cfg.MIN_DISTANCE)
        for d in distances
    ])

    # Training metrics
    episode_rewards = []
    episode_lengths = []
    current_episode_reward = 0.0
    current_episode_length = 0
    running_hits = 0
    running_requests = 0
    
    # FIXED: Episode management
    STEPS_PER_EPISODE = 1000  # Define episode length
    episode_num = 0

    print("\nTraining progress:")
    print_interval = max(1, num_training_steps // 20)
    t0 = time.time()

    for step in range(num_training_steps):
        # Update channel periodically (simulate mobility)
        if step % 50 == 0:
            small_scale = channel_model.rayleigh_gain(cfg.NUM_USERS)
            channel_gains = pl * small_scale

        # Generate request
        file_id = sample_zipf_catalog(cfg.NUM_FILES, cfg.ZIPF_ALPHA, size=1)[0]
        user_id = np.random.choice(cfg.NUM_USERS)

        # FIXED: Check if this is the last step of episode
        is_episode_end = ((step + 1) % STEPS_PER_EPISODE == 0)

        # Check cache
        cache_hit = cache.is_hit(file_id)
        running_requests += 1
        
        if cache_hit:
            running_hits += 1
            # Cache hit - observe with episode_done flag
            cache.observe_request(
                user_id=user_id,
                file_id=file_id,
                cache_hit=True,
                channel_gain=channel_gains[user_id],
                episode_done=is_episode_end  # FIXED: Mark episode boundaries
            )
            # Use cache's normalized reward
            reward = cache.last_reward if hasattr(cache, 'last_reward') else 1.0
            current_episode_reward += float(reward)
        else:
            # Cache miss - NOMA transmission
            partner_user = np.random.choice([u for u in range(cfg.NUM_USERS) if u != user_id])

            if channel_gains[user_id] < channel_gains[partner_user]:
                weak_user, strong_user = user_id, partner_user
            else:
                weak_user, strong_user = partner_user, user_id

            gain_weak = channel_gains[weak_user]
            gain_strong = channel_gains[strong_user]

            # Power allocation and transmission
            p_w, p_s, feasible, _ = allocate_power_gridsearch(
                gain_weak, gain_strong, cfg, grid_points=cfg.POWER_ALLOC_GRID
            )

            weak_success, strong_success, sinr_w, _, sinr_s_after = \
                simulate_noma_pair(gain_weak, gain_strong, cfg, p_w, p_s)

            noma_success = weak_success if user_id == weak_user else strong_success
            user_sinr = sinr_w if user_id == weak_user else sinr_s_after
            ber = compute_ber_bpsk(user_sinr)
            outage = not noma_success

            # Learn from transmission with episode_done flag
            cache.observe_request(
                user_id=user_id,
                file_id=file_id,
                cache_hit=False,
                noma_success=noma_success,
                channel_gain=channel_gains[user_id],
                sinr_weak=sinr_w,
                sinr_strong=sinr_s_after,
                ber=ber,
                outage=outage,
                episode_done=is_episode_end  # FIXED: Mark episode boundaries
            )

            # Use cache's normalized reward
            reward = cache.last_reward if hasattr(cache, 'last_reward') else (-10.0 if outage else -1.0)
            current_episode_reward += float(reward)

        current_episode_length += 1

        # FIXED: Episode end handling
        if is_episode_end:
            episode_rewards.append(current_episode_reward)
            episode_lengths.append(current_episode_length)
            episode_num += 1
            
            # Log episode to TensorBoard
            if tb_writer is not None:
                tb_writer.add_scalar("train/episode_reward", current_episode_reward, episode_num)
                tb_writer.add_scalar("train/episode_length", current_episode_length, episode_num)
            
            current_episode_reward = 0.0
            current_episode_length = 0

        # Periodic checkpointing
        if (step + 1) % checkpoint_interval == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"ckpt_step_{step+1}.pt")
            ckpt_latest = os.path.join(checkpoint_dir, "latest.pt")
            try:
                cache.save_model(ckpt_path)
                cache.save_model(ckpt_latest)
            except Exception as e:
                print(f"⚠️ Failed to save checkpoint: {e}")

        # TensorBoard logging
        if tb_writer is not None and (step % 100 == 0):
            stats = cache.get_stats()
            tb_writer.add_scalar("train/epsilon", stats.get("epsilon", 0.0), step+1)
            tb_writer.add_scalar("train/cumulative_reward", stats.get("cumulative_reward", 0.0), step+1)
            tb_writer.add_scalar("train/avg_loss", stats.get("avg_loss", 0.0), step+1)
            hit_rate = running_hits / running_requests if running_requests > 0 else 0.0
            tb_writer.add_scalar("train/hit_rate_running", hit_rate, step+1)

        # Print progress
        if (step + 1) % print_interval == 0:
            stats = cache.get_stats()
            recent_reward = np.mean(episode_rewards[-5:]) if episode_rewards else 0
            elapsed = time.time() - t0
            hit_rate = running_hits / running_requests if running_requests > 0 else 0.0
            
            print(f"  Step {step+1}/{num_training_steps} | "
                  f"Epsilon: {stats.get('epsilon', 0):.3f} | "
                  f"Avg Reward: {recent_reward:.1f} | "
                  f"Loss: {stats.get('avg_loss', 0):.6f} | "
                  f"HitRate: {hit_rate:.3f} | "
                  f"Elapsed: {elapsed:.1f}s")
            t0 = time.time()
            running_hits = 0
            running_requests = 0

        # Periodic TB flush
        if tb_writer is not None and (step % 500 == 0):
            tb_writer.flush()

    # Final checkpoint
    try:
        ckpt_latest = os.path.join(checkpoint_dir, "latest.pt")
        cache.save_model(ckpt_latest)
    except Exception:
        pass

    print(f"\n✅ Training complete!")
    print(f"   Final epsilon: {cache.epsilon:.3f}")
    print(f"   Total training steps: {cache.training_step}")
    print(f"   Avg episode reward: {np.mean(episode_rewards):.2f}" if episode_rewards else "   No episodes completed")

    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'final_stats': cache.get_stats()
    }


def run_evaluation_phase(cache, cfg, seed, num_eval_requests=5000):
    """
    FIXED EVALUATION PHASE using set_eval_mode().
    """
    set_seed(seed)

    # FIXED: Use proper evaluation mode
    cache.set_eval_mode(True)

    # Generate evaluation environment
    user_pos = channel_model.generate_user_positions(
        cfg.NUM_USERS, cfg.CELL_RADIUS, seed=seed
    )
    distances = user_pos[:, 2]
    pl = np.array([
        channel_model.pathloss(d, cfg.PATHLOSS_EXPONENT, cfg.MIN_DISTANCE)
        for d in distances
    ])
    small_scale = channel_model.rayleigh_gain(cfg.NUM_USERS)
    channel_gains = pl * small_scale

    # Metrics
    total_requests = 0
    total_hits = 0
    total_transmissions = 0
    total_outages = 0
    ber_weak_list = []
    ber_strong_list = []
    noma_success_count = 0

    # Simulate evaluation requests
    for step in range(num_eval_requests):
        if step % 50 == 0:
            small_scale = channel_model.rayleigh_gain(cfg.NUM_USERS)
            channel_gains = pl * small_scale

        file_id = sample_zipf_catalog(cfg.NUM_FILES, cfg.ZIPF_ALPHA, size=1)[0]
        user_id = np.random.choice(cfg.NUM_USERS)

        total_requests += 1
        cache_hit = cache.is_hit(file_id)

        if cache_hit:
            total_hits += 1

        else:
            total_transmissions += 1

            # NOMA transmission
            partner_user = np.random.choice([u for u in range(cfg.NUM_USERS) if u != user_id])

            if channel_gains[user_id] < channel_gains[partner_user]:
                weak_user, strong_user = user_id, partner_user
            else:
                weak_user, strong_user = partner_user, user_id

            gain_weak = channel_gains[weak_user]
            gain_strong = channel_gains[strong_user]

            p_w, p_s, feasible, _ = allocate_power_gridsearch(
                gain_weak, gain_strong, cfg, grid_points=cfg.POWER_ALLOC_GRID
            )

            weak_success, strong_success, sinr_w, _, sinr_s_after = \
                simulate_noma_pair(gain_weak, gain_strong, cfg, p_w, p_s)

            noma_success = weak_success if user_id == weak_user else strong_success

            ber_weak = compute_ber_bpsk(sinr_w)
            ber_strong = compute_ber_bpsk(sinr_s_after)
            ber_weak_list.append(ber_weak)
            ber_strong_list.append(ber_strong)

            if not noma_success:
                total_outages += 1
            else:
                noma_success_count += 1

    # FIXED: Restore training mode
    cache.set_eval_mode(False)

    # Calculate metrics
    hit_rate = total_hits / total_requests if total_requests > 0 else 0.0
    outage_rate = total_outages / total_transmissions if total_transmissions > 0 else 0.0
    noma_success_rate = noma_success_count / total_transmissions if total_transmissions > 0 else 0.0

    return {
        'hit_rate': hit_rate,
        'outage_rate': outage_rate,
        'noma_success_rate': noma_success_rate,
        'avg_ber_weak': np.mean(ber_weak_list) if ber_weak_list else 0.0,
        'avg_ber_strong': np.mean(ber_strong_list) if ber_strong_list else 0.0,
        'total_hits': total_hits,
        'total_transmissions': total_transmissions,
        'total_outages': total_outages
    }


def run_complete_rl_experiment(
    cfg,
    seed,
    num_training_steps=50000,
    num_eval_requests=5000,
    checkpoint_dir="./checkpoints",
    tb_log_dir="./tb_logs",
    fresh_start=True  # FIXED: Add option for fresh training
):
    """
    FIXED: Complete RL experiment with proper checkpoint management.
    """
    set_seed(seed)

    # FIXED: Clear checkpoint dir for fresh start if requested
    seed_ckpt_dir = os.path.join(checkpoint_dir, f"seed_{seed}")
    if fresh_start and os.path.exists(seed_ckpt_dir):
        try:
            shutil.rmtree(seed_ckpt_dir)
            print(f"🗑️  Cleared old checkpoints: {seed_ckpt_dir}")
        except Exception as e:
            print(f"⚠️ Failed to clear checkpoints: {e}")
    
    os.makedirs(seed_ckpt_dir, exist_ok=True)

    # Initialize improved RL cache
    cache = ImprovedDQNNomaCache(
        capacity=cfg.CACHE_SIZE,
        num_files=cfg.NUM_FILES,
        num_users=cfg.NUM_USERS,
        learning_rate=cfg.RL_LEARNING_RATE,
        gamma=cfg.RL_GAMMA,
        epsilon_start=cfg.RL_EPSILON_START,
        epsilon_end=cfg.RL_EPSILON_END,
        epsilon_decay_steps=cfg.RL_EPSILON_DECAY_STEPS,  # FIXED: Use config value
        use_neural_network=cfg.RL_USE_NEURAL_NETWORK,
        batch_size=cfg.RL_BATCH_SIZE,
        replay_buffer_size=cfg.RL_REPLAY_BUFFER_SIZE,
        seed=seed
    )

    # TensorBoard writer
    writer = None
    if TENSORBOARD_AVAILABLE:
        run_tb_dir = os.path.join(tb_log_dir, f"run_seed_{seed}_{int(time.time())}")
        os.makedirs(run_tb_dir, exist_ok=True)
        writer = SummaryWriter(run_tb_dir)
        print(f"📊 TensorBoard logs -> {run_tb_dir}")

    # Phase 1: Training
    training_results = run_training_phase(
        cache, cfg, num_training_steps,
        checkpoint_dir=seed_ckpt_dir,
        checkpoint_interval=max(2000, num_training_steps // 10),
        tb_writer=writer,
        seed=seed
    )

    # Phase 2: Evaluation
    eval_results = run_evaluation_phase(cache, cfg, seed, num_eval_requests)

    # Final stats
    final_stats = cache.get_stats()

    # Log eval to TensorBoard
    if writer is not None:
        writer.add_scalar("eval/hit_rate", eval_results['hit_rate'], final_stats.get("training_step", 0))
        writer.add_scalar("eval/outage_rate", eval_results['outage_rate'], final_stats.get("training_step", 0))
        writer.add_scalar("eval/noma_success_rate", eval_results['noma_success_rate'], final_stats.get("training_step", 0))
        writer.flush()
        writer.close()

    result = {
        'seed': seed,
        'policy': 'improved_dqn_noma',
        'use_neural_network': final_stats.get('use_neural_network', False),

        # Training metrics
        'training_steps': num_training_steps,
        'final_epsilon': final_stats.get('epsilon', 0.0),
        'cumulative_reward': final_stats.get('cumulative_reward', 0.0),
        'avg_training_loss': final_stats.get('avg_loss', 0.0),

        # Evaluation metrics
        'hit_rate': eval_results['hit_rate'],
        'outage_rate': eval_results['outage_rate'],
        'noma_success_rate': eval_results['noma_success_rate'],
        'avg_ber_weak': eval_results['avg_ber_weak'],
        'avg_ber_strong': eval_results['avg_ber_strong'],
        'avg_ber': (eval_results['avg_ber_weak'] + eval_results['avg_ber_strong']) / 2.0,

        # Other stats
        'total_hits': eval_results['total_hits'],
        'total_transmissions': eval_results['total_transmissions'],
        'total_outages': eval_results['total_outages'],
        'cache_size': final_stats.get('cache_size', 0),
    }

    return result, cache, training_results


def run_improved_rl_monte_carlo(cfg, num_runs=50, num_training_steps=50000):
    """
    FIXED: Run multiple experiments with proper checkpoint management.
    """
    print("\n" + "="*70)
    print("IMPROVED RL-DQN-NOMA SIMULATION")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Neural Network: {'Yes (PyTorch)' if USE_IMPROVED else 'No (Q-table fallback)'}")
    print(f"  Files: {cfg.NUM_FILES}")
    print(f"  Cache Size: {cfg.CACHE_SIZE}")
    print(f"  Users: {cfg.NUM_USERS}")
    print(f"  Training Steps: {num_training_steps}")
    print(f"  Epsilon Decay Steps: {cfg.RL_EPSILON_DECAY_STEPS}")
    print(f"  Runs: {num_runs}")

    results = []
    all_training_curves = []

    base_ckpt_dir = "./checkpoints"
    tb_root_dir = "./tb_logs"
    os.makedirs(base_ckpt_dir, exist_ok=True)
    os.makedirs(tb_root_dir, exist_ok=True)

    for run in range(num_runs):
        print(f"\n{'='*70}")
        print(f"RUN {run+1}/{num_runs}")
        print(f"{'='*70}")

        seed = cfg.RANDOM_SEED + run

        result, trained_cache, training_results = run_complete_rl_experiment(
            cfg, seed,
            num_training_steps=num_training_steps,
            num_eval_requests=5000,
            checkpoint_dir=base_ckpt_dir,
            tb_log_dir=tb_root_dir,
            fresh_start=True  # FIXED: Always start fresh for each run
        )

        results.append(result)
        all_training_curves.append(training_results['episode_rewards'])

        print(f"\n📊 Run {run+1} Results:")
        print(f"   Hit Rate:      {result['hit_rate']:.4f}")
        print(f"   Outage Rate:   {result['outage_rate']:.4f}")
        print(f"   NOMA Success:  {result['noma_success_rate']:.4f}")
        print(f"   BER:           {result['avg_ber']:.6f}")
        print(f"   Final Epsilon: {result['final_epsilon']:.4f}")

    df = pd.DataFrame(results)

    # Print summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"Mean Hit Rate:     {df['hit_rate'].mean():.4f} ± {df['hit_rate'].std():.4f}")
    print(f"Mean Outage Rate:  {df['outage_rate'].mean():.4f} ± {df['outage_rate'].std():.4f}")
    print(f"Mean NOMA Success: {df['noma_success_rate'].mean():.4f} ± {df['noma_success_rate'].std():.4f}")
    print(f"Mean BER:          {df['avg_ber'].mean():.6f} ± {df['avg_ber'].std():.6f}")
    print("="*70)

    # Plot learning curves
    plot_learning_curves(all_training_curves, save_path='./improved_rl_learning_curves.png')

    return df, all_training_curves


def plot_learning_curves(training_curves, save_path='learning_curves.png'):
    """Plot learning curves showing training progress."""
    if not training_curves or not any(training_curves):
        print("⚠️ No training curves to plot.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Individual curves (plot first 5 only for clarity)
    for i, curve in enumerate(training_curves[:5]):
        if len(curve) > 0:
            ax1.plot(curve, alpha=0.3, label=f'Run {i+1}')

    # Average curve
    min_len = min(len(c) for c in training_curves if len(c) > 0)
    if min_len == 0:
        print("⚠️ No valid curves to average.")
        return
        
    truncated = [c[:min_len] for c in training_curves if len(c) >= min_len]
    if len(truncated) == 0:
        print("⚠️ No curves to truncate.")
        return
        
    avg_curve = np.mean(truncated, axis=0)
    std_curve = np.std(truncated, axis=0)

    episodes = np.arange(len(avg_curve))
    ax1.plot(avg_curve, 'r-', linewidth=2, label='Average')
    ax1.fill_between(episodes, avg_curve - std_curve, avg_curve + std_curve,
                     alpha=0.2, color='red')

    ax1.set_xlabel('Episode (1000 steps each)')
    ax1.set_ylabel('Episode Reward')
    ax1.set_title('Learning Progress: Episode Rewards')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Moving average
    window = 5
    if len(avg_curve) >= window:
        moving_avg = np.convolve(avg_curve, np.ones(window)/window, mode='valid')
        ax2.plot(moving_avg, 'b-', linewidth=2, label=f'{window}-episode MA')

    ax2.plot(avg_curve, 'r--', alpha=0.5, label='Raw Average')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Episode Reward')
    ax2.set_title('Smoothed Learning Curve')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved learning curves: {save_path}")
    plt.close()


if __name__ == "__main__":
    from src import config as cfg

    df, curves = run_improved_rl_monte_carlo(
        cfg,
        num_runs=cfg.NUM_RUNS,
        num_training_steps=cfg.RL_TRAINING_STEPS
    )

    df.to_csv('results_improved_rl_dqn_noma.csv', index=False)
    print("\n✅ Results saved to: results_improved_rl_dqn_noma.csv")
    