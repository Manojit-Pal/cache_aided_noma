"""
config.py

Comprehensive Configuration for Cache-Aided NOMA System with DQN
=================================================================

This file contains all configurable parameters for:
- DQN Cache implementation (with all bug fixes applied)
- NOMA system parameters
- Simulation settings
- Network topology

All parameters are research-backed with default values from:
- Mnih et al. (2015): DQN
- Schaul et al. (2016): Prioritized Experience Replay
- Wang et al. (2016): Dueling DQN
- Lillicrap et al. (2016): Soft target updates

Author: Cache-Aided NOMA Team
Date: December 2025
Version: 2.0 (Post bug fixes #1-6)
"""

import numpy as np
from typing import Dict, Any


# ====================================================================================
# SYSTEM-LEVEL PARAMETERS
# ====================================================================================

class SystemConfig:
    """High-level system configuration."""
    
    # Random seed for reproducibility
    SEED = 2025
    
    # Device selection
    DEVICE = "auto"  # Options: "auto", "cuda", "cpu"
    
    # Logging
    VERBOSE = True
    LOG_INTERVAL = 100  # Log every N requests
    
    # File paths
    RESULTS_DIR = "results/"
    MODELS_DIR = "models/"
    LOGS_DIR = "logs/"


# ====================================================================================
# NOMA SYSTEM PARAMETERS
# ====================================================================================

class NOMAConfig:
    """NOMA physical layer and system parameters."""
    
    # ============================================================================
    # Users and Files
    # ============================================================================
    NUM_USERS = 10
    NUM_FILES = 100
    FILE_SIZE = 1.0  # MB (normalized)
    
    # ============================================================================
    # Channel Model
    # ============================================================================
    CHANNEL_MODEL = "rayleigh"  # Options: "rayleigh", "rician", "awgn"
    PATH_LOSS_EXPONENT = 3.5
    
    # User distribution
    MIN_DISTANCE = 10.0  # meters
    MAX_DISTANCE = 100.0  # meters
    
    # ============================================================================
    # Power Allocation
    # ============================================================================
    TOTAL_POWER = 1.0  # Watts (normalized)
    POWER_ALLOCATION_SCHEME = "ftpa"  # Options: "ftpa", "equal", "adaptive"
    
    # FTPA (Fractional Transmit Power Allocation) parameters
    POWER_RATIO_WEAK = 0.8  # 80% power to weak user
    POWER_RATIO_STRONG = 0.2  # 20% power to strong user
    
    # ============================================================================
    # SIC (Successive Interference Cancellation)
    # ============================================================================
    ENABLE_SIC = True
    SIC_ERROR_THRESHOLD = 1e-3  # BER threshold for successful SIC
    
    # ============================================================================
    # CIC (Cache-aided Interference Cancellation)
    # ============================================================================
    ENABLE_CIC = True
    CIC_GAIN_FACTOR = 1.5  # SINR improvement factor when cache helps
    
    # ============================================================================
    # Noise and Interference
    # ============================================================================
    NOISE_POWER = 1e-10  # Watts
    TEMPERATURE = 290  # Kelvin
    BANDWIDTH = 1e6  # Hz (1 MHz)
    
    # ============================================================================
    # SINR Thresholds
    # ============================================================================
    SINR_THRESHOLD_WEAK = 0.0  # dB (minimum for weak user)
    SINR_THRESHOLD_STRONG = 5.0  # dB (minimum for strong user)
    OUTAGE_SINR = -5.0  # dB (below this = outage)


# ====================================================================================
# DQN CACHE PARAMETERS (Updated with Bug Fixes #1-6)
# ====================================================================================

class DQNConfig:
    """
    Deep Q-Network cache configuration.
    
    All parameters have been validated against research:
    - Mnih et al. (2015): Playing Atari with Deep RL
    - Schaul et al. (2016): Prioritized Experience Replay
    - Wang et al. (2016): Dueling Network Architectures
    - Lillicrap et al. (2016): Continuous Control with Deep RL
    """
    
    # ============================================================================
    # Cache Parameters
    # ============================================================================
    CACHE_CAPACITY = 10  # Number of files that can be cached
    ENABLE_NOMA_AWARENESS = True  # Use NOMA-specific features in state
    
    # ============================================================================
    # Network Architecture
    # ============================================================================
    USE_NEURAL_NETWORK = True  # False = Q-table fallback
    HIDDEN_DIMS = [128, 64]  # Dueling DQN hidden layer sizes
    
    # State dimension: 2*CACHE_CAPACITY + 6
    # - LRU counters (CACHE_CAPACITY)
    # - LFU counters (CACHE_CAPACITY)
    # - Requested file popularity (1)
    # - Cache occupancy (1)
    # - Mean channel gain (1)
    # - Std channel gain (1)
    # - CIC success rate (1)
    # - NOMA success rate (1)
    
    # ============================================================================
    # Learning Hyperparameters
    # ============================================================================
    LEARNING_RATE = 0.0001  # Adam optimizer learning rate
    GAMMA = 0.95  # Discount factor for future rewards
    
    # ============================================================================
    # Exploration Strategy (Epsilon-Greedy)
    # ============================================================================
    EPSILON_START = 1.0  # Initial exploration rate
    EPSILON_END = 0.01  # Final exploration rate (1% random)
    EPSILON_DECAY_STEPS = 25000  # Steps to decay from start to end
    
    # ============================================================================
    # Experience Replay
    # ============================================================================
    BATCH_SIZE = 64  # Mini-batch size for training
    REPLAY_BUFFER_SIZE = 50000  # Maximum experiences stored
    TRAIN_FREQ = 4  # Train every N steps
    
    # ============================================================================
    # BUG FIX #6: Warm-up Period
    # ============================================================================
    # Research: Wait for diverse experiences before training
    # Default: max(10 * BATCH_SIZE, 1000) = 640 experiences
    WARM_UP_STEPS = None  # None = auto (10 * BATCH_SIZE)
    # Set explicitly if needed:
    # WARM_UP_STEPS = 1000
    
    # ============================================================================
    # BUG FIX #2: Prioritized Experience Replay (Schaul et al., 2016)
    # ============================================================================
    USE_PRIORITIZED_REPLAY = True
    
    # Priority exponent: controls how much prioritization is used
    # α = 0: uniform sampling (no prioritization)
    # α = 1: full prioritization (proportional to TD-error)
    PRIORITY_ALPHA = 0.6  # Standard value from paper
    
    # Importance sampling exponent: compensates for bias
    # β anneals from BETA_START to BETA_END over training
    PRIORITY_BETA_START = 0.4  # Start with more bias (faster learning)
    PRIORITY_BETA_END = 1.0  # End with no bias (accurate estimates)
    PRIORITY_BETA_FRAMES = 100000  # Anneal over 100k samples
    
    # ============================================================================
    # BUG FIX #4: Target Network Updates (Lillicrap et al., 2016)
    # ============================================================================
    # Soft update: θ_target ← τ*θ_policy + (1-τ)*θ_target
    # Happens EVERY training step (not every 1000 steps)
    TAU = 0.005  # Soft update coefficient (0.5% of policy per step)
    
    # Note: TARGET_UPDATE_FREQ removed - soft updates happen every training step
    
    # ============================================================================
    # Stability and Regularization
    # ============================================================================
    GRADIENT_CLIP = 10.0  # Max gradient norm (prevents exploding gradients)
    WEIGHT_DECAY = 1e-5  # L2 regularization in Adam optimizer
    
    # ============================================================================
    # Reward Function (NOMA-Aware)
    # ============================================================================
    # Reward structure:
    REWARD_CACHE_HIT = 10.0  # Best outcome
    REWARD_CIC_ENABLED = 2.0  # Good outcome (cache helps NOMA)
    REWARD_NOMA_SUCCESS = -1.0  # Acceptable outcome
    REWARD_NOMA_FAILURE = -5.0  # Bad outcome
    REWARD_OUTAGE = -10.0  # Worst outcome
    
    # BER-based modifiers
    REWARD_EXCELLENT_BER = 1.0  # Bonus for BER < 1e-4
    REWARD_POOR_BER = -2.0  # Penalty for BER > 1e-2
    
    # ============================================================================
    # Popularity Tracking (BUG FIX #1: Correct EMA)
    # ============================================================================
    POPULARITY_DECAY = 0.9  # EMA decay factor
    # Update: p[i] = DECAY * p[i] + (1-DECAY) for accessed item
    # Others decay through normalization


# ====================================================================================
# SIMULATION PARAMETERS
# ====================================================================================

class SimulationConfig:
    """Simulation and evaluation parameters."""
    
    # ============================================================================
    # Training
    # ============================================================================
    NUM_EPISODES = 1000  # Training episodes
    REQUESTS_PER_EPISODE = 1000  # Requests per episode
    
    # ============================================================================
    # Evaluation
    # ============================================================================
    EVAL_INTERVAL = 10  # Evaluate every N episodes
    EVAL_EPISODES = 10  # Number of episodes for evaluation
    
    # ============================================================================
    # Request Generation
    # ============================================================================
    REQUEST_DISTRIBUTION = "zipf"  # Options: "zipf", "uniform", "temporal"
    ZIPF_ALPHA = 0.8  # Zipf exponent (higher = more skewed)
    
    # Temporal patterns (for "temporal" distribution)
    ENABLE_TEMPORAL_PATTERNS = False
    PATTERN_PERIOD = 100  # Requests before pattern repeats
    
    # ============================================================================
    # NOMA Pairing Strategy
    # ============================================================================
    PAIRING_STRATEGY = "distance_based"  # Options: "distance_based", "random", "optimal"
    MAX_PAIRING_DISTANCE = 50.0  # Max distance between paired users
    
    # ============================================================================
    # Checkpointing
    # ============================================================================
    SAVE_INTERVAL = 50  # Save model every N episodes
    SAVE_BEST_ONLY = True  # Only save if performance improves
    
    # ============================================================================
    # Metrics to Track
    # ============================================================================
    TRACK_METRICS = [
        "cache_hit_ratio",
        "average_sinr_weak",
        "average_sinr_strong",
        "cic_enabled_ratio",
        "sic_success_ratio",
        "outage_probability",
        "average_reward",
        "epsilon",
        "loss",
        "replay_buffer_size",
        "per_beta"  # Prioritized replay beta value
    ]


# ====================================================================================
# BASELINE ALGORITHMS (for comparison)
# ====================================================================================

class BaselineConfig:
    """Configuration for baseline cache policies."""
    
    BASELINES = [
        "lru",      # Least Recently Used
        "lfu",      # Least Frequently Used
        "fifo",     # First In First Out
        "random",   # Random eviction
        "optimal"   # Optimal (requires future knowledge)
    ]
    
    # Run baselines for comparison
    ENABLE_BASELINE_COMPARISON = True


# ====================================================================================
# HELPER FUNCTIONS
# ====================================================================================

def get_config() -> Dict[str, Any]:
    """
    Get complete configuration as a dictionary.
    
    Returns:
        Dictionary with all configuration parameters
    """
    return {
        "system": {
            "seed": SystemConfig.SEED,
            "device": SystemConfig.DEVICE,
            "verbose": SystemConfig.VERBOSE,
            "log_interval": SystemConfig.LOG_INTERVAL,
            "results_dir": SystemConfig.RESULTS_DIR,
            "models_dir": SystemConfig.MODELS_DIR,
            "logs_dir": SystemConfig.LOGS_DIR,
        },
        "noma": {
            "num_users": NOMAConfig.NUM_USERS,
            "num_files": NOMAConfig.NUM_FILES,
            "file_size": NOMAConfig.FILE_SIZE,
            "channel_model": NOMAConfig.CHANNEL_MODEL,
            "path_loss_exponent": NOMAConfig.PATH_LOSS_EXPONENT,
            "min_distance": NOMAConfig.MIN_DISTANCE,
            "max_distance": NOMAConfig.MAX_DISTANCE,
            "total_power": NOMAConfig.TOTAL_POWER,
            "power_allocation_scheme": NOMAConfig.POWER_ALLOCATION_SCHEME,
            "power_ratio_weak": NOMAConfig.POWER_RATIO_WEAK,
            "power_ratio_strong": NOMAConfig.POWER_RATIO_STRONG,
            "enable_sic": NOMAConfig.ENABLE_SIC,
            "sic_error_threshold": NOMAConfig.SIC_ERROR_THRESHOLD,
            "enable_cic": NOMAConfig.ENABLE_CIC,
            "cic_gain_factor": NOMAConfig.CIC_GAIN_FACTOR,
            "noise_power": NOMAConfig.NOISE_POWER,
            "temperature": NOMAConfig.TEMPERATURE,
            "bandwidth": NOMAConfig.BANDWIDTH,
            "sinr_threshold_weak": NOMAConfig.SINR_THRESHOLD_WEAK,
            "sinr_threshold_strong": NOMAConfig.SINR_THRESHOLD_STRONG,
            "outage_sinr": NOMAConfig.OUTAGE_SINR,
        },
        "dqn": {
            "cache_capacity": DQNConfig.CACHE_CAPACITY,
            "enable_noma_awareness": DQNConfig.ENABLE_NOMA_AWARENESS,
            "use_neural_network": DQNConfig.USE_NEURAL_NETWORK,
            "hidden_dims": DQNConfig.HIDDEN_DIMS,
            "learning_rate": DQNConfig.LEARNING_RATE,
            "gamma": DQNConfig.GAMMA,
            "epsilon_start": DQNConfig.EPSILON_START,
            "epsilon_end": DQNConfig.EPSILON_END,
            "epsilon_decay_steps": DQNConfig.EPSILON_DECAY_STEPS,
            "batch_size": DQNConfig.BATCH_SIZE,
            "replay_buffer_size": DQNConfig.REPLAY_BUFFER_SIZE,
            "train_freq": DQNConfig.TRAIN_FREQ,
            "warm_up_steps": DQNConfig.WARM_UP_STEPS,
            "use_prioritized_replay": DQNConfig.USE_PRIORITIZED_REPLAY,
            "priority_alpha": DQNConfig.PRIORITY_ALPHA,
            "priority_beta_start": DQNConfig.PRIORITY_BETA_START,
            "priority_beta_end": DQNConfig.PRIORITY_BETA_END,
            "priority_beta_frames": DQNConfig.PRIORITY_BETA_FRAMES,
            "tau": DQNConfig.TAU,
            "gradient_clip": DQNConfig.GRADIENT_CLIP,
            "weight_decay": DQNConfig.WEIGHT_DECAY,
            "rewards": {
                "cache_hit": DQNConfig.REWARD_CACHE_HIT,
                "cic_enabled": DQNConfig.REWARD_CIC_ENABLED,
                "noma_success": DQNConfig.REWARD_NOMA_SUCCESS,
                "noma_failure": DQNConfig.REWARD_NOMA_FAILURE,
                "outage": DQNConfig.REWARD_OUTAGE,
                "excellent_ber": DQNConfig.REWARD_EXCELLENT_BER,
                "poor_ber": DQNConfig.REWARD_POOR_BER,
            },
            "popularity_decay": DQNConfig.POPULARITY_DECAY,
        },
        "simulation": {
            "num_episodes": SimulationConfig.NUM_EPISODES,
            "requests_per_episode": SimulationConfig.REQUESTS_PER_EPISODE,
            "eval_interval": SimulationConfig.EVAL_INTERVAL,
            "eval_episodes": SimulationConfig.EVAL_EPISODES,
            "request_distribution": SimulationConfig.REQUEST_DISTRIBUTION,
            "zipf_alpha": SimulationConfig.ZIPF_ALPHA,
            "enable_temporal_patterns": SimulationConfig.ENABLE_TEMPORAL_PATTERNS,
            "pattern_period": SimulationConfig.PATTERN_PERIOD,
            "pairing_strategy": SimulationConfig.PAIRING_STRATEGY,
            "max_pairing_distance": SimulationConfig.MAX_PAIRING_DISTANCE,
            "save_interval": SimulationConfig.SAVE_INTERVAL,
            "save_best_only": SimulationConfig.SAVE_BEST_ONLY,
            "track_metrics": SimulationConfig.TRACK_METRICS,
        },
        "baseline": {
            "baselines": BaselineConfig.BASELINES,
            "enable_baseline_comparison": BaselineConfig.ENABLE_BASELINE_COMPARISON,
        },
    }


def print_config():
    """
    Print configuration in a readable format.
    """
    config = get_config()
    
    print("="*70)
    print("CACHE-AIDED NOMA CONFIGURATION")
    print("="*70)
    
    for section, params in config.items():
        print(f"\n[{section.upper()}]")
        print("-" * 70)
        for key, value in params.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")
    
    print("\n" + "="*70)
    print("✅ Configuration loaded successfully!")
    print("="*70)


def validate_config():
    """
    Validate configuration parameters.
    
    Raises:
        ValueError: If any parameter is invalid
    """
    errors = []
    
    # Validate NOMA parameters
    if NOMAConfig.NUM_USERS <= 0:
        errors.append("NUM_USERS must be positive")
    if NOMAConfig.NUM_FILES <= 0:
        errors.append("NUM_FILES must be positive")
    if not (0 < NOMAConfig.POWER_RATIO_WEAK < 1):
        errors.append("POWER_RATIO_WEAK must be in (0, 1)")
    if not (0 < NOMAConfig.POWER_RATIO_STRONG < 1):
        errors.append("POWER_RATIO_STRONG must be in (0, 1)")
    if abs(NOMAConfig.POWER_RATIO_WEAK + NOMAConfig.POWER_RATIO_STRONG - 1.0) > 1e-6:
        errors.append("Power ratios must sum to 1.0")
    
    # Validate DQN parameters
    if DQNConfig.CACHE_CAPACITY <= 0:
        errors.append("CACHE_CAPACITY must be positive")
    if DQNConfig.CACHE_CAPACITY > NOMAConfig.NUM_FILES:
        errors.append("CACHE_CAPACITY cannot exceed NUM_FILES")
    if not (0 < DQNConfig.LEARNING_RATE < 1):
        errors.append("LEARNING_RATE must be in (0, 1)")
    if not (0 < DQNConfig.GAMMA <= 1):
        errors.append("GAMMA must be in (0, 1]")
    if not (0 < DQNConfig.EPSILON_END <= DQNConfig.EPSILON_START <= 1):
        errors.append("Epsilon values must satisfy: 0 < END <= START <= 1")
    if DQNConfig.BATCH_SIZE <= 0:
        errors.append("BATCH_SIZE must be positive")
    if not (0 < DQNConfig.PRIORITY_ALPHA <= 1):
        errors.append("PRIORITY_ALPHA must be in (0, 1]")
    if not (0 < DQNConfig.PRIORITY_BETA_START <= 1):
        errors.append("PRIORITY_BETA_START must be in (0, 1]")
    if DQNConfig.PRIORITY_BETA_END != 1.0:
        errors.append("PRIORITY_BETA_END should be 1.0 (per Schaul et al.)")
    if not (0 < DQNConfig.TAU < 1):
        errors.append("TAU must be in (0, 1)")
    
    # Validate simulation parameters
    if SimulationConfig.NUM_EPISODES <= 0:
        errors.append("NUM_EPISODES must be positive")
    if SimulationConfig.REQUESTS_PER_EPISODE <= 0:
        errors.append("REQUESTS_PER_EPISODE must be positive")
    
    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
    
    print("✅ Configuration validated successfully!")


# ====================================================================================
# USAGE EXAMPLE
# ====================================================================================

if __name__ == "__main__":
    # Print configuration
    print_config()
    
    # Validate configuration
    try:
        validate_config()
    except ValueError as e:
        print(f"❌ {e}")
        exit(1)
    
    # Get configuration dictionary
    config = get_config()
    print(f"\n✅ Configuration ready for use!")
    print(f"   Total parameters: {sum(len(v) if isinstance(v, dict) else 1 for v in config.values())}")
