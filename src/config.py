"""
src/config.py

Simulation configuration parameters for Cache-Aided NOMA with DQN.

Bug Fixes Applied (2026):
  BUG-5        : RL_STEPS_PER_EPISODE / RL_TRAINING_STEPS / RL_EPSILON_DECAY_STEPS
                 corrected after BUG-SIM-2 fix — actual steps/episode = 10,000
  BUG-7        : RL_REWARD_CIC_ENABLED 7.0 → 2.0 (prevents perverse incentive)
  BUG-8        : Added set_debug_config() small-scale preset
  BUG-CONFIG-1 : set_quick_test_config() step counts corrected
  BUG-CONFIG-2 : set_full_experiment_config() step counts corrected
  BUG-CONFIG-3 : RL_PRIORITY_BETA_FRAMES 100K → 20M (full training duration)
  ROOT-1       : RL_REPLAY_BUFFER_SIZE 50K → 500K (was 20% overwritten/episode)
  ROOT-2       : RL_WARM_UP_STEPS None(→1000) → 15,000 (1.5 full episodes)
  ROOT-4       : RL_TRAIN_FREQUENCY 4 → 10 (reduce overfitting to recent data)
"""

# Random seed for reproducibility
RANDOM_SEED = 2025

# ------------------------------
# Content catalog
# ------------------------------
NUM_FILES = 2000        # total unique files in catalog
ZIPF_ALPHA = 1.0        # Zipf skew parameter (1.0 = strong skew)

# ------------------------------
# Users & requests
# ------------------------------
NUM_USERS = 200         # number of users in the cell
REQUESTS_PER_USER = 50  # number of requests per user per episode

# ------------------------------
# Cache
# ------------------------------
CACHE_SIZE = 200        # ~10% of catalog size
CACHE_POLICY = "stable_dqn"
CACHE_DELIVERY_RATE = 10.0   # bits/s/Hz

# ------------------------------
# Novel algorithm parameters
# ------------------------------
NOMA_AWARE_ALPHA_POP = 0.3
NOMA_AWARE_ALPHA_CHANNEL = 0.2
CACHE_UPDATE_INTERVAL = 100
TIME_SLOTS = 1000

# ------------------------------
# Multi-objective weights
# ------------------------------
MO_OBJECTIVES = ['hit_rate', 'outage', 'energy']
MO_HIT_WEIGHT = 0.4
MO_OUTAGE_WEIGHT = 0.4
MO_ENERGY_WEIGHT = 0.2

# ------------------------------
# Monte Carlo runs
# ------------------------------
NUM_RUNS = 100

# ------------------------------
# NOMA & channel params
# ------------------------------
TX_POWER = 2.0
NOISE_POWER = 1e-9
CELL_RADIUS = 500.0
PATHLOSS_EXPONENT = 3.5
MIN_DISTANCE = 1.0

# User pairing
PAIR_USERS = True
PAIRING_METHOD = "extreme"  # Options: 'extreme', 'random', 'sequential'

# Power allocation
POWER_COEFF_WEAK = 0.8
POWER_COEFF_STRONG = 0.2
POWER_ALLOC_METHOD = "cache_aware"  # Options: 'gridsearch', 'closedform', 'cache_aware', 'sumrate_max', 'energy_efficient'
POWER_ALLOC_GRID = 101
USE_CLOSED_FORM_ALLOC = False  # DEPRECATED: use POWER_ALLOC_METHOD

# SIC parameters
SIC_IMPERFECTION = 0.05  # Residual interference factor (ζ)
TARGET_RATE_BPS = 0.3    # Target data rate in bps/Hz

# Channel modeling parameters
FADING_TYPE = "mixed"     # Options: 'rayleigh', 'rician', 'mixed'
RICIAN_K_FACTOR_DB = 10.0
LOS_PROBABILITY = 0.4
ENABLE_MOBILITY = False
DOPPLER_FREQ = 10.0

# Cache-aware NOMA parameters
ENABLE_CIC = True
CIC_PERFECT = True
CACHE_HIT_ENABLES_CIC = True

# Performance thresholds
BER_THRESHOLD = 1e-3
OUTAGE_SINR_MARGIN = 2.0


# ============================================================================
# STABLE DQN CACHE PARAMETERS
# ============================================================================

# ------------------------------
# Training Configuration
# BUG-5 FIX: Steps per episode = NUM_USERS * REQUESTS_PER_USER = 10,000
# ------------------------------
RL_TRAINING_EPISODES   = 2000        # number of training episodes
RL_STEPS_PER_EPISODE   = 10_000      # BUG-5 FIX: NUM_USERS(200) x REQ_PER_USER(50)
RL_TRAINING_STEPS      = 20_000_000  # BUG-5 FIX: 2000 x 10,000

# ------------------------------
# Epsilon-Greedy Exploration
# BUG-5 FIX: Decay steps corrected to 50% of actual total training steps
# ------------------------------
RL_EPSILON_START        = 1.0
RL_EPSILON_END          = 0.01
RL_EPSILON_DECAY_STEPS  = 10_000_000  # BUG-5 FIX: 50% of 20M real steps
RL_EVAL_EPSILON         = 0.0

# ------------------------------
# Neural Network Architecture
# ------------------------------
RL_USE_NEURAL_NETWORK = True
RL_HIDDEN_DIMS = [128, 128]   # Dueling DQN shared feature layers

# ------------------------------
# Learning Hyperparameters
# ------------------------------
RL_LEARNING_RATE      = 0.0001   # Adam optimizer
RL_GAMMA              = 0.99     # Discount factor
RL_BATCH_SIZE         = 64

# ROOT-1 FIX: Buffer was 50,000 (only 5 episodes worth).
# Each episode writes 10,000 steps → 20% of buffer overwritten/episode.
# After just 5 episodes the entire buffer was fresh data → catastrophic
# forgetting. New size = 500,000 = 50 episodes worth of experience.
# Rule of thumb: buffer >= 10-50x steps_per_episode.
RL_REPLAY_BUFFER_SIZE = 500_000   # ROOT-1 FIX: was 50,000

# ------------------------------
# Training Stability
# ------------------------------
RL_GRADIENT_CLIP      = 10.0   # Max norm for gradient clipping
RL_TAU                = 0.005  # Soft target network update rate (every step)
RL_TARGET_UPDATE_FREQ = 1000   # DEPRECATED: kept for backward compat

# ROOT-4 FIX: train_freq was 4 → 2,500 gradient steps/episode on a 50K buffer
# = buffer sampled ~5x per episode → overfitting to most recent data.
# New: 1,000 gradient steps/episode on 500K buffer = sampled 0.2x/episode.
RL_TRAIN_FREQUENCY    = 10    # ROOT-4 FIX: was 4

# ------------------------------
# Warm-Up Period
# ROOT-2 FIX: Was None → auto-resolves to max(10*64, 1000) = 1,000.
# Warm-up ended after 10% of episode 1 with zero-diversity buffer.
# New: 15,000 = 1.5 full episodes of random exploration before any
# gradient step. Ensures the buffer has diverse (s,a,r,s') transitions
# from multiple random policies before learning begins.
# ------------------------------
RL_WARM_UP_STEPS = 15_000    # ROOT-2 FIX: was None (→ 1,000)

# ------------------------------
# Reward Function Parameters
# BUG-7 FIX: RL_REWARD_CIC_ENABLED 7.0 → 2.0
# ------------------------------
RL_REWARD_CACHE_HIT          =  10.0
RL_REWARD_CIC_ENABLED        =   2.0  # BUG-7 FIX: was 7.0
RL_REWARD_CACHE_MISS_SUCCESS =  -1.0
RL_REWARD_NOMA_FAILURE       =  -5.0
RL_REWARD_OUTAGE             = -10.0
RL_REWARD_POOR_BER           =  -2.0
RL_REWARD_GOOD_BER           =   1.0

# BER thresholds for reward shaping
RL_BER_THRESHOLD_GOOD = 1e-4
RL_BER_THRESHOLD_POOR = 1e-2

# ------------------------------
# Prioritized Experience Replay (Schaul et al., ICLR 2016)
# BUG-CONFIG-3 FIX: RL_PRIORITY_BETA_FRAMES 100K → 20M
# ------------------------------
RL_USE_PRIORITIZED_REPLAY  = True
RL_PRIORITY_ALPHA          = 0.6
RL_PRIORITY_BETA_START     = 0.4
RL_PRIORITY_BETA_END       = 1.0
RL_PRIORITY_BETA_FRAMES    = 20_000_000  # BUG-CONFIG-3 FIX: was 100,000

# ------------------------------
# Popularity Tracking
# ------------------------------
RL_POPULARITY_DECAY = 0.9

# ------------------------------
# Evaluation Configuration
# ------------------------------
RL_EVAL_REQUESTS       = 5000
RL_SEPARATE_TRAIN_EVAL = True

# ------------------------------
# Checkpointing & Logging
# ------------------------------
RL_SAVE_CHECKPOINTS  = True
RL_CHECKPOINT_FREQ   = 10000
RL_CHECKPOINT_DIR    = "./checkpoints/"
RL_TRACK_LEARNING    = True
RL_VERBOSE           = True
RL_PLOT_LEARNING_CURVES = True
RL_SAVE_TRAINING_LOGS   = True

# ------------------------------
# Comparison Experiments
# ------------------------------
COMPARE_POLICIES = ["topk", "lru", "lfu", "stable_dqn"]


# ============================================================================
# CONFIGURATION PRESETS
# ============================================================================

def set_debug_config():
    """
    Tiny-scale config for rapid convergence verification (~2-3 min on CPU).

    Use this FIRST after any code change. DQN hit rate should clearly
    rise above TopK within ~10 episodes if everything is working.

    Scale:
        NUM_FILES=100, CACHE_SIZE=10 (10%), NUM_USERS=20, REQ_PER_USER=20
        Steps/episode = 400
        Total steps   = 50 * 400 = 20,000
    """
    global NUM_FILES, CACHE_SIZE, NUM_USERS, REQUESTS_PER_USER
    global RL_TRAINING_EPISODES, RL_STEPS_PER_EPISODE, RL_TRAINING_STEPS
    global RL_EPSILON_DECAY_STEPS, NUM_RUNS, RL_EVAL_REQUESTS
    global RL_REPLAY_BUFFER_SIZE, RL_WARM_UP_STEPS, RL_PRIORITY_BETA_FRAMES
    global RL_TRAIN_FREQUENCY

    NUM_FILES          = 100
    CACHE_SIZE         = 10
    NUM_USERS          = 20
    REQUESTS_PER_USER  = 20

    _steps_per_ep = NUM_USERS * REQUESTS_PER_USER   # 400
    RL_TRAINING_EPISODES    = 50
    RL_STEPS_PER_EPISODE    = _steps_per_ep            # 400
    RL_TRAINING_STEPS       = 50 * _steps_per_ep       # 20,000
    RL_EPSILON_DECAY_STEPS  = 50 * _steps_per_ep // 2  # 10,000
    NUM_RUNS                = 5
    RL_EVAL_REQUESTS        = 500
    # ROOT-1/2/4 ratios preserved at debug scale:
    RL_REPLAY_BUFFER_SIZE   = 20_000   # 50x steps_per_ep
    RL_WARM_UP_STEPS        = 600      # 1.5x steps_per_ep
    RL_TRAIN_FREQUENCY      = 10
    RL_PRIORITY_BETA_FRAMES = RL_TRAINING_STEPS

    print("[DEBUG] Config enabled")
    print(f"  Catalog   : {NUM_FILES} files, cache={CACHE_SIZE} ({CACHE_SIZE/NUM_FILES*100:.0f}%)")
    print(f"  Users     : {NUM_USERS} x {REQUESTS_PER_USER} req = {_steps_per_ep} steps/ep")
    print(f"  Training  : {RL_TRAINING_EPISODES} ep x {RL_STEPS_PER_EPISODE} = {RL_TRAINING_STEPS:,} total")
    print(f"  Buffer    : {RL_REPLAY_BUFFER_SIZE:,} (50x steps/ep)")
    print(f"  Warm-up   : {RL_WARM_UP_STEPS} steps (1.5x steps/ep)")
    print(f"  Expected  : ~2-3 minutes on CPU")


def set_quick_test_config():
    """
    Quick test configuration (~15-20 minutes on CPU).
    BUG-CONFIG-1 FIX + ROOT-1/2/4 ratios applied.
    """
    global RL_TRAINING_EPISODES, RL_STEPS_PER_EPISODE, RL_TRAINING_STEPS
    global RL_EPSILON_DECAY_STEPS, NUM_RUNS, RL_EVAL_REQUESTS
    global RL_PRIORITY_BETA_FRAMES, RL_REPLAY_BUFFER_SIZE, RL_WARM_UP_STEPS
    global RL_TRAIN_FREQUENCY

    _steps_per_ep = NUM_USERS * REQUESTS_PER_USER
    RL_TRAINING_EPISODES    = 100
    RL_STEPS_PER_EPISODE    = _steps_per_ep
    RL_TRAINING_STEPS       = 100 * _steps_per_ep
    RL_EPSILON_DECAY_STEPS  = 100 * _steps_per_ep // 2
    NUM_RUNS                = 10
    RL_EVAL_REQUESTS        = 2000
    RL_REPLAY_BUFFER_SIZE   = 50 * _steps_per_ep   # 500,000
    RL_WARM_UP_STEPS        = int(1.5 * _steps_per_ep)  # 15,000
    RL_TRAIN_FREQUENCY      = 10
    RL_PRIORITY_BETA_FRAMES = RL_TRAINING_STEPS

    print("[QUICK TEST] Config enabled")
    print(f"  Training  : {RL_TRAINING_EPISODES} ep x {RL_STEPS_PER_EPISODE:,} = {RL_TRAINING_STEPS:,} total")
    print(f"  Buffer    : {RL_REPLAY_BUFFER_SIZE:,}")
    print(f"  Expected  : ~15-20 minutes on CPU")


def set_full_experiment_config():
    """
    Full experiment configuration for paper results.
    BUG-CONFIG-2 FIX + ROOT-1/2/4 ratios applied.
    """
    global RL_TRAINING_EPISODES, RL_STEPS_PER_EPISODE, RL_TRAINING_STEPS
    global RL_EPSILON_DECAY_STEPS, NUM_RUNS, RL_EVAL_REQUESTS
    global RL_PRIORITY_BETA_FRAMES, RL_REPLAY_BUFFER_SIZE, RL_WARM_UP_STEPS
    global RL_TRAIN_FREQUENCY

    _steps_per_ep = NUM_USERS * REQUESTS_PER_USER
    RL_TRAINING_EPISODES    = 2000
    RL_STEPS_PER_EPISODE    = _steps_per_ep
    RL_TRAINING_STEPS       = 2000 * _steps_per_ep
    RL_EPSILON_DECAY_STEPS  = 2000 * _steps_per_ep // 2
    NUM_RUNS                = 100
    RL_EVAL_REQUESTS        = 5000
    RL_REPLAY_BUFFER_SIZE   = 50 * _steps_per_ep   # 500,000
    RL_WARM_UP_STEPS        = int(1.5 * _steps_per_ep)  # 15,000
    RL_TRAIN_FREQUENCY      = 10
    RL_PRIORITY_BETA_FRAMES = RL_TRAINING_STEPS

    print("[FULL EXPERIMENT] Config enabled")
    print(f"  Training  : {RL_TRAINING_EPISODES} ep x {RL_STEPS_PER_EPISODE:,} = {RL_TRAINING_STEPS:,} total")
    print(f"  Buffer    : {RL_REPLAY_BUFFER_SIZE:,}")
    print(f"  Expected  : ~4-6 hrs CPU, ~1 hr GPU")


def set_aggressive_learning_config():
    """Aggressive learning config (faster convergence, less stable)."""
    global RL_LEARNING_RATE, RL_BATCH_SIZE, RL_HIDDEN_DIMS, RL_EPSILON_DECAY_STEPS
    RL_LEARNING_RATE       = 0.0005
    RL_BATCH_SIZE          = 128
    RL_HIDDEN_DIMS         = [256, 128]
    RL_EPSILON_DECAY_STEPS = RL_TRAINING_STEPS // 3
    print("[AGGRESSIVE] Config enabled — may be less stable")


def set_conservative_learning_config():
    """Conservative learning config (more stable, slower convergence)."""
    global RL_LEARNING_RATE, RL_BATCH_SIZE, RL_GRADIENT_CLIP, RL_TAU
    RL_LEARNING_RATE  = 0.00005
    RL_BATCH_SIZE     = 32
    RL_GRADIENT_CLIP  = 5.0
    RL_TAU            = 0.001
    print("[CONSERVATIVE] Config enabled — more stable, slower")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_noma_config():
    return {
        'TX_POWER': TX_POWER, 'NOISE_POWER': NOISE_POWER,
        'CELL_RADIUS': CELL_RADIUS, 'PATHLOSS_EXPONENT': PATHLOSS_EXPONENT,
        'MIN_DISTANCE': MIN_DISTANCE, 'PAIR_USERS': PAIR_USERS,
        'PAIRING_METHOD': PAIRING_METHOD,
        'POWER_COEFF_WEAK': POWER_COEFF_WEAK, 'POWER_COEFF_STRONG': POWER_COEFF_STRONG,
        'POWER_ALLOC_METHOD': POWER_ALLOC_METHOD,
        'SIC_IMPERFECTION': SIC_IMPERFECTION, 'TARGET_RATE_BPS': TARGET_RATE_BPS,
        'FADING_TYPE': FADING_TYPE, 'RICIAN_K_FACTOR_DB': RICIAN_K_FACTOR_DB,
        'LOS_PROBABILITY': LOS_PROBABILITY,
        'ENABLE_CIC': ENABLE_CIC, 'CIC_PERFECT': CIC_PERFECT,
    }


def get_rl_config():
    import sys
    module = sys.modules[__name__]
    return {attr: getattr(module, attr)
            for attr in dir(module) if attr.startswith('RL_')}


def print_noma_config():
    print("\n" + "="*70)
    print("NOMA SYSTEM CONFIGURATION")
    print("="*70)
    print(f"\n  TX Power            : {TX_POWER}W")
    print(f"  Noise Power         : {NOISE_POWER}W")
    print(f"  Cell Radius         : {CELL_RADIUS}m")
    print(f"  Path Loss Exponent  : {PATHLOSS_EXPONENT}")
    print(f"  Fading Type         : {FADING_TYPE}")
    print(f"  Pairing Method      : {PAIRING_METHOD}")
    print(f"  Power Alloc Method  : {POWER_ALLOC_METHOD}")
    print(f"  SIC Imperfection    : {SIC_IMPERFECTION}")
    print(f"  Target Rate         : {TARGET_RATE_BPS} bps/Hz")
    print(f"  CIC Enabled         : {ENABLE_CIC}")
    print("\n" + "="*70 + "\n")


def print_rl_config():
    print("\n" + "="*70)
    print("STABLE DQN CACHE CONFIGURATION")
    print("="*70)
    actual_steps = NUM_USERS * REQUESTS_PER_USER
    print(f"  Episodes            : {RL_TRAINING_EPISODES}")
    print(f"  Steps / episode     : {RL_STEPS_PER_EPISODE:,}  (actual={actual_steps:,})")
    print(f"  Total steps         : {RL_TRAINING_STEPS:,}")
    print(f"  Replay buffer       : {RL_REPLAY_BUFFER_SIZE:,}  ({RL_REPLAY_BUFFER_SIZE//RL_STEPS_PER_EPISODE}x steps/ep)")
    print(f"  Warm-up             : {RL_WARM_UP_STEPS:,} steps")
    print(f"  Train frequency     : every {RL_TRAIN_FREQUENCY} steps")
    print(f"  Epsilon decay       : {RL_EPSILON_START} -> {RL_EPSILON_END} over {RL_EPSILON_DECAY_STEPS:,} steps")
    print(f"  PER beta frames     : {RL_PRIORITY_BETA_FRAMES:,}")
    print("\n" + "="*70 + "\n")


def validate_config():
    """Validate configuration parameters. Returns True if all OK."""
    issues = []

    if not (0 < CACHE_SIZE <= NUM_FILES):
        issues.append(f"CACHE_SIZE ({CACHE_SIZE}) must be in (0, NUM_FILES={NUM_FILES}]")
    if not (0 < RL_GAMMA <= 1):
        issues.append(f"RL_GAMMA ({RL_GAMMA}) must be in (0, 1]")
    if not (0 < RL_LEARNING_RATE < 1):
        issues.append(f"RL_LEARNING_RATE ({RL_LEARNING_RATE}) out of range")
    if RL_EPSILON_START < RL_EPSILON_END:
        issues.append(f"RL_EPSILON_START must be >= RL_EPSILON_END")
    if RL_BATCH_SIZE > RL_REPLAY_BUFFER_SIZE:
        issues.append(f"RL_BATCH_SIZE ({RL_BATCH_SIZE}) > RL_REPLAY_BUFFER_SIZE ({RL_REPLAY_BUFFER_SIZE})")

    actual_steps_per_ep = NUM_USERS * REQUESTS_PER_USER
    expected_total      = RL_TRAINING_EPISODES * actual_steps_per_ep
    if RL_TRAINING_STEPS != expected_total:
        issues.append(
            f"RL_TRAINING_STEPS ({RL_TRAINING_STEPS:,}) != "
            f"EPISODES x ACTUAL_STEPS ({expected_total:,})")
    if RL_STEPS_PER_EPISODE != actual_steps_per_ep:
        issues.append(
            f"RL_STEPS_PER_EPISODE ({RL_STEPS_PER_EPISODE:,}) != "
            f"NUM_USERS x REQUESTS_PER_USER ({actual_steps_per_ep:,})")

    if not (0 < POWER_COEFF_WEAK < 1):
        issues.append(f"POWER_COEFF_WEAK ({POWER_COEFF_WEAK}) must be in (0,1)")
    if not (0 < POWER_COEFF_STRONG < 1):
        issues.append(f"POWER_COEFF_STRONG ({POWER_COEFF_STRONG}) must be in (0,1)")
    if abs(POWER_COEFF_WEAK + POWER_COEFF_STRONG - 1.0) > 1e-6:
        issues.append("POWER_COEFF_WEAK + POWER_COEFF_STRONG must equal 1.0")
    if FADING_TYPE not in ['rayleigh', 'rician', 'mixed']:
        issues.append(f"Invalid FADING_TYPE: '{FADING_TYPE}'")
    if PAIRING_METHOD not in ['extreme', 'random', 'sequential']:
        issues.append(f"Invalid PAIRING_METHOD: '{PAIRING_METHOD}'")
    if RL_USE_PRIORITIZED_REPLAY:
        if not (0 < RL_PRIORITY_ALPHA <= 1):
            issues.append(f"RL_PRIORITY_ALPHA must be in (0,1]")
        if not (0 < RL_PRIORITY_BETA_START <= 1):
            issues.append(f"RL_PRIORITY_BETA_START must be in (0,1]")
        if RL_PRIORITY_BETA_END != 1.0:
            issues.append(f"RL_PRIORITY_BETA_END should be 1.0")
    if not (0 < RL_TAU < 1):
        issues.append(f"RL_TAU must be in (0,1)")
    if RL_REWARD_CIC_ENABLED >= RL_REWARD_CACHE_HIT:
        issues.append(
            f"RL_REWARD_CIC_ENABLED ({RL_REWARD_CIC_ENABLED}) must be < "
            f"RL_REWARD_CACHE_HIT ({RL_REWARD_CACHE_HIT})")
    if RL_REWARD_CIC_ENABLED <= RL_REWARD_NOMA_FAILURE:
        issues.append(
            f"RL_REWARD_CIC_ENABLED ({RL_REWARD_CIC_ENABLED}) must be > "
            f"RL_REWARD_NOMA_FAILURE ({RL_REWARD_NOMA_FAILURE})")

    if issues:
        print("[WARN] Configuration Issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("[OK] Configuration validated successfully")
        return True


# ============================================================================
# AUTO-VALIDATION ON IMPORT
# ============================================================================

if __name__ == "__main__":
    print_noma_config()
    print_rl_config()
    validate_config()
else:
    validate_config()