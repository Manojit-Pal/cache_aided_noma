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
# (was 200 — the old value assumed only one request per user was processed;
#  BUG-SIM-2 fix in stable_dqn_sim.py now processes all 50 requests/user)
# ------------------------------
RL_TRAINING_EPISODES   = 2000        # number of training episodes
RL_STEPS_PER_EPISODE   = 10_000      # ✅ BUG-5 FIX: NUM_USERS(200) × REQ_PER_USER(50)
RL_TRAINING_STEPS      = 20_000_000  # ✅ BUG-5 FIX: 2000 × 10,000

# Training rationale:
# - 2000 episodes, each with 10,000 requests → 20M total steps
# - Matches research standard (arXiv:1712.08132, DRLCache)
# - DO NOT manually set RL_TRAINING_STEPS to any other value;
#   validate_config() will catch any mismatch.

# ------------------------------
# Epsilon-Greedy Exploration
# BUG-5 FIX: Decay steps corrected to 50% of actual total training steps
# Was 200,000 → epsilon hit 0.01 after only 1% of real training.
# Now 10,000,000 → epsilon decays correctly over first half of training.
# ------------------------------
RL_EPSILON_START        = 1.0
RL_EPSILON_END          = 0.01
RL_EPSILON_DECAY_STEPS  = 10_000_000  # ✅ BUG-5 FIX: 50% of 20M real steps
RL_EVAL_EPSILON         = 0.0

# Epsilon schedule:
#   Steps 0 → 10M  (episodes   0–1000): ε = 1.0 → 0.01  (exploration)
#   Steps 10M→ 20M (episodes 1000–2000): ε = 0.01         (exploitation)

# ------------------------------
# Neural Network Architecture
# ------------------------------
RL_USE_NEURAL_NETWORK = True
RL_HIDDEN_DIMS = [128, 128]   # Dueling DQN shared feature layers

# ------------------------------
# Learning Hyperparameters
# ------------------------------
RL_LEARNING_RATE = 0.0001     # Adam optimizer; research consensus
RL_GAMMA         = 0.99       # Discount factor; values long-term hits
RL_BATCH_SIZE    = 64
RL_REPLAY_BUFFER_SIZE = 50000

# ------------------------------
# Training Stability
# ------------------------------
RL_GRADIENT_CLIP      = 10.0   # Max norm for gradient clipping
RL_TAU                = 0.005  # Soft target network update rate (every step)
RL_TARGET_UPDATE_FREQ = 1000   # DEPRECATED: kept for backward compat
RL_TRAIN_FREQUENCY    = 4      # Train every N environment steps

# ------------------------------
# Warm-Up Period
# ------------------------------
RL_WARM_UP_STEPS = None   # None → auto: max(10 × BATCH_SIZE, 1000) = 1000

# ------------------------------
# Reward Function Parameters
# BUG-7 FIX: RL_REWARD_CIC_ENABLED 7.0 → 2.0
# Hierarchy must be strictly ordered to avoid perverse incentives:
#   hit=+10  >>  CIC=+2  >  miss=-1  >>  failure=-5  >>  outage=-10
# The old value of 7.0 created a gap of 8 between CIC and regular miss,
# causing the DQN to optimise CIC (a side-effect it can't control) over
# actual cache hits. dqn_cache_final._compute_reward() already hardcodes
# +2.0; this value is now aligned to prevent confusion.
# ------------------------------
RL_REWARD_CACHE_HIT         =  10.0  # cache hit  (best)
RL_REWARD_CIC_ENABLED       =   2.0  # ✅ BUG-7 FIX: was 7.0 (perverse incentive)
RL_REWARD_CACHE_MISS_SUCCESS =  -1.0  # miss + NOMA success (no CIC)
RL_REWARD_NOMA_FAILURE       =  -5.0  # miss + NOMA failure
RL_REWARD_OUTAGE             = -10.0  # miss + outage (worst)
RL_REWARD_POOR_BER           =  -2.0  # additional penalty for high BER
RL_REWARD_GOOD_BER           =   1.0  # bonus for good BER

# BER thresholds for reward shaping
RL_BER_THRESHOLD_GOOD = 1e-4
RL_BER_THRESHOLD_POOR = 1e-2

# ------------------------------
# Prioritized Experience Replay (Schaul et al., ICLR 2016)
# BUG-CONFIG-3 FIX: RL_PRIORITY_BETA_FRAMES 100K → 20M
# Beta was fully annealing (0.4→1.0) after only 0.5% of training.
# Should cover the full training run per the original PER paper.
# ------------------------------
RL_USE_PRIORITIZED_REPLAY  = True
RL_PRIORITY_ALPHA          = 0.6    # priority exponent (0=uniform, 1=full)
RL_PRIORITY_BETA_START     = 0.4
RL_PRIORITY_BETA_END       = 1.0
RL_PRIORITY_BETA_FRAMES    = 20_000_000  # ✅ BUG-CONFIG-3 FIX: was 100,000

# ------------------------------
# Popularity Tracking
# ------------------------------
RL_POPULARITY_DECAY = 0.9  # EMA decay; full-vector decay applied in dqn_cache_final

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
    ✅ NEW (BUG-8 FIX): Tiny-scale config for rapid convergence verification.

    Use this FIRST after any code change to confirm the DQN is actually
    learning (hit rate should clearly rise above TopK within ~10 episodes).
    Takes ~2-3 minutes on CPU.

    Scale:
        NUM_FILES=100, CACHE_SIZE=10 (10%), NUM_USERS=20, REQ_PER_USER=20
        Steps/episode = 20 * 20 = 400
        Total steps   = 50 * 400 = 20,000
    """
    global NUM_FILES, CACHE_SIZE, NUM_USERS, REQUESTS_PER_USER
    global RL_TRAINING_EPISODES, RL_STEPS_PER_EPISODE, RL_TRAINING_STEPS
    global RL_EPSILON_DECAY_STEPS, NUM_RUNS, RL_EVAL_REQUESTS
    global RL_REPLAY_BUFFER_SIZE, RL_WARM_UP_STEPS, RL_PRIORITY_BETA_FRAMES

    NUM_FILES          = 100
    CACHE_SIZE         = 10
    NUM_USERS          = 20
    REQUESTS_PER_USER  = 20

    _steps_per_ep = NUM_USERS * REQUESTS_PER_USER   # 400
    RL_TRAINING_EPISODES   = 50
    RL_STEPS_PER_EPISODE   = _steps_per_ep           # 400
    RL_TRAINING_STEPS      = 50 * _steps_per_ep      # 20,000
    RL_EPSILON_DECAY_STEPS = 50 * _steps_per_ep // 2 # 10,000  (50% of total)
    NUM_RUNS               = 5
    RL_EVAL_REQUESTS       = 500
    RL_REPLAY_BUFFER_SIZE  = 5000
    RL_WARM_UP_STEPS       = 200
    RL_PRIORITY_BETA_FRAMES = RL_TRAINING_STEPS       # anneal over full run

    print("🐛 Debug Config Enabled (BUG-8 fix)")
    print(f"   Catalog      : {NUM_FILES} files, cache={CACHE_SIZE} ({CACHE_SIZE/NUM_FILES*100:.0f}%)")
    print(f"   Users        : {NUM_USERS} users × {REQUESTS_PER_USER} req = {_steps_per_ep} steps/ep")
    print(f"   Training     : {RL_TRAINING_EPISODES} episodes × {RL_STEPS_PER_EPISODE} steps = {RL_TRAINING_STEPS:,} total")
    print(f"   ε decay      : over first {RL_EPSILON_DECAY_STEPS:,} steps (50%)")
    print(f"   Expected time: ~2-3 minutes on CPU")
    print(f"   Pass criteria: DQN hit_rate > TopK within ~10 episodes")


def set_quick_test_config():
    """
    Quick test configuration (~15-20 minutes on CPU).
    BUG-CONFIG-1 FIX: step counts updated to match actual steps/episode=10,000.
    """
    global RL_TRAINING_EPISODES, RL_STEPS_PER_EPISODE, RL_TRAINING_STEPS
    global RL_EPSILON_DECAY_STEPS, NUM_RUNS, RL_EVAL_REQUESTS
    global RL_PRIORITY_BETA_FRAMES

    _steps_per_ep = NUM_USERS * REQUESTS_PER_USER   # 10,000 (uses current globals)
    RL_TRAINING_EPISODES   = 100
    RL_STEPS_PER_EPISODE   = _steps_per_ep            # ✅ BUG-CONFIG-1 FIX
    RL_TRAINING_STEPS      = 100 * _steps_per_ep      # ✅ 1,000,000 (was 20,000)
    RL_EPSILON_DECAY_STEPS = 100 * _steps_per_ep // 2 # ✅ 500,000  (was 10,000)
    NUM_RUNS               = 10
    RL_EVAL_REQUESTS       = 2000
    RL_PRIORITY_BETA_FRAMES = RL_TRAINING_STEPS

    print("⚡ Quick Test Config Enabled")
    print(f"   Training: {RL_TRAINING_EPISODES} episodes × {RL_STEPS_PER_EPISODE:,} steps = {RL_TRAINING_STEPS:,} total")
    print(f"   ε decay : over first {RL_EPSILON_DECAY_STEPS:,} steps (50%)")
    print(f"   Eval    : {NUM_RUNS} runs × {RL_EVAL_REQUESTS} requests")
    print(f"   Expected time: ~15-20 minutes on CPU")


def set_full_experiment_config():
    """
    Full experiment configuration for paper results (~4-6 hours on CPU, ~1 hr on GPU).
    BUG-CONFIG-2 FIX: step counts updated to match actual steps/episode=10,000.
    """
    global RL_TRAINING_EPISODES, RL_STEPS_PER_EPISODE, RL_TRAINING_STEPS
    global RL_EPSILON_DECAY_STEPS, NUM_RUNS, RL_EVAL_REQUESTS
    global RL_PRIORITY_BETA_FRAMES

    _steps_per_ep = NUM_USERS * REQUESTS_PER_USER   # 10,000
    RL_TRAINING_EPISODES    = 2000
    RL_STEPS_PER_EPISODE    = _steps_per_ep           # ✅ BUG-CONFIG-2 FIX
    RL_TRAINING_STEPS       = 2000 * _steps_per_ep    # ✅ 20,000,000 (was 400,000)
    RL_EPSILON_DECAY_STEPS  = 2000 * _steps_per_ep // 2  # ✅ 10,000,000 (was 200,000)
    NUM_RUNS                = 100
    RL_EVAL_REQUESTS        = 5000
    RL_PRIORITY_BETA_FRAMES = RL_TRAINING_STEPS

    print("🎓 Full Experiment Config Enabled (Research Standard)")
    print(f"   Training: {RL_TRAINING_EPISODES} episodes × {RL_STEPS_PER_EPISODE:,} steps = {RL_TRAINING_STEPS:,} total")
    print(f"   ε decay : over first {RL_EPSILON_DECAY_STEPS:,} steps (50%)")
    print(f"   Eval    : {NUM_RUNS} runs × {RL_EVAL_REQUESTS} requests")
    print(f"   Expected time: ~4-6 hours on CPU, ~1 hour on GPU")


def set_aggressive_learning_config():
    """
    Aggressive learning configuration (faster convergence, less stable).
    """
    global RL_LEARNING_RATE, RL_BATCH_SIZE, RL_HIDDEN_DIMS, RL_EPSILON_DECAY_STEPS

    RL_LEARNING_RATE       = 0.0005
    RL_BATCH_SIZE          = 128
    RL_HIDDEN_DIMS         = [256, 128]
    RL_EPSILON_DECAY_STEPS = RL_TRAINING_STEPS // 3   # decay over first third

    print("⚡ Aggressive Learning Config Enabled")
    print("   ⚠️  May be less stable but converges faster")


def set_conservative_learning_config():
    """
    Conservative learning configuration (more stable, slower convergence).
    """
    global RL_LEARNING_RATE, RL_BATCH_SIZE, RL_GRADIENT_CLIP, RL_TAU

    RL_LEARNING_RATE  = 0.00005
    RL_BATCH_SIZE     = 32
    RL_GRADIENT_CLIP  = 5.0
    RL_TAU            = 0.001

    print("🐢 Conservative Learning Config Enabled")
    print("   ✅ More stable but requires longer training")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_noma_config():
    """Return NOMA-specific configuration as a dictionary."""
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
    """Return all RL_ parameters as a dictionary."""
    import sys
    module = sys.modules[__name__]
    return {attr: getattr(module, attr)
            for attr in dir(module) if attr.startswith('RL_')}


def print_noma_config():
    """Print NOMA configuration for verification."""
    print("\n" + "="*70)
    print("NOMA SYSTEM CONFIGURATION")
    print("="*70)
    print(f"\n📡 CHANNEL PARAMETERS:")
    print(f"  TX Power            : {TX_POWER}W")
    print(f"  Noise Power         : {NOISE_POWER}W")
    print(f"  Cell Radius         : {CELL_RADIUS}m")
    print(f"  Path Loss Exponent  : {PATHLOSS_EXPONENT}")
    print(f"  Fading Type         : {FADING_TYPE}")
    if FADING_TYPE in ['rician', 'mixed']:
        print(f"  Rician K-factor     : {RICIAN_K_FACTOR_DB} dB")
    if FADING_TYPE == 'mixed':
        print(f"  LoS Probability     : {LOS_PROBABILITY}")
    print(f"\n👥 USER PAIRING:")
    print(f"  Method              : {PAIRING_METHOD}")
    print(f"  Number of Users     : {NUM_USERS}")
    print(f"\n⚡ POWER ALLOCATION:")
    print(f"  Method              : {POWER_ALLOC_METHOD}")
    print(f"  p_weak              : {POWER_COEFF_WEAK}")
    print(f"  p_strong            : {POWER_COEFF_STRONG}")
    print(f"\n🔄 SIC PARAMETERS:")
    print(f"  Imperfection Factor : {SIC_IMPERFECTION}")
    print(f"  Target Rate         : {TARGET_RATE_BPS} bps/Hz")
    print(f"\n💾 CACHE-AIDED FEATURES:")
    print(f"  CIC Enabled         : {ENABLE_CIC}")
    print(f"  Perfect CIC         : {CIC_PERFECT}")
    print(f"  Cache Size          : {CACHE_SIZE}")
    print("\n" + "="*70 + "\n")


def print_rl_config():
    """Print RL configuration for verification."""
    print("\n" + "="*70)
    print("STABLE DQN CACHE CONFIGURATION")
    print("="*70)
    actual_steps = NUM_USERS * REQUESTS_PER_USER
    print(f"\n📋 TRAINING:")
    print(f"  Episodes            : {RL_TRAINING_EPISODES}")
    print(f"  Steps / episode     : {RL_STEPS_PER_EPISODE:,}  (NUM_USERS×REQ_PER_USER={actual_steps:,})")
    if RL_STEPS_PER_EPISODE != actual_steps:
        print(f"  ⚠️  MISMATCH: config says {RL_STEPS_PER_EPISODE}, actual = {actual_steps}")
    print(f"  Total steps         : {RL_TRAINING_STEPS:,}")
    print(f"\n📉 EPSILON DECAY:")
    print(f"  Start               : {RL_EPSILON_START}")
    print(f"  End                 : {RL_EPSILON_END}")
    print(f"  Decay over          : {RL_EPSILON_DECAY_STEPS:,} steps ({RL_EPSILON_DECAY_STEPS/max(RL_TRAINING_STEPS,1)*100:.0f}% of training)")
    print(f"\n🧠 NETWORK:")
    print(f"  Architecture        : {RL_HIDDEN_DIMS}")
    print(f"  Learning Rate       : {RL_LEARNING_RATE}")
    print(f"  Batch Size          : {RL_BATCH_SIZE}")
    print(f"  Replay Buffer       : {RL_REPLAY_BUFFER_SIZE:,}")
    print(f"  Gamma               : {RL_GAMMA}")
    print(f"\n⚖️  REWARD STRUCTURE:")
    print(f"  Cache hit           : +{RL_REWARD_CACHE_HIT}")
    print(f"  CIC enabled         : +{RL_REWARD_CIC_ENABLED}  (BUG-7 fix: was 7.0)")
    print(f"  Miss + success      :  {RL_REWARD_CACHE_MISS_SUCCESS}")
    print(f"  Miss + failure      :  {RL_REWARD_NOMA_FAILURE}")
    print(f"  Outage              :  {RL_REWARD_OUTAGE}")
    print(f"\n🎯 STABILITY:")
    print(f"  Gradient Clip       : {RL_GRADIENT_CLIP}")
    print(f"  Soft Update τ       : {RL_TAU}")
    print(f"  Train Frequency     : every {RL_TRAIN_FREQUENCY} steps")
    print(f"  Prioritized Replay  : {RL_USE_PRIORITIZED_REPLAY}")
    if RL_USE_PRIORITIZED_REPLAY:
        print(f"    α                 : {RL_PRIORITY_ALPHA}")
        print(f"    β                 : {RL_PRIORITY_BETA_START} → {RL_PRIORITY_BETA_END} over {RL_PRIORITY_BETA_FRAMES:,} frames")
    print("\n" + "="*70 + "\n")


def validate_config():
    """Validate configuration parameters and print any issues found."""
    issues = []

    if not (0 < CACHE_SIZE <= NUM_FILES):
        issues.append(f"CACHE_SIZE ({CACHE_SIZE}) must be in (0, NUM_FILES={NUM_FILES}]")
    if not (0 < RL_GAMMA <= 1):
        issues.append(f"RL_GAMMA ({RL_GAMMA}) must be in (0, 1]")
    if not (0 < RL_LEARNING_RATE < 1):
        issues.append(f"RL_LEARNING_RATE ({RL_LEARNING_RATE}) out of range")
    if RL_EPSILON_START < RL_EPSILON_END:
        issues.append(f"RL_EPSILON_START ({RL_EPSILON_START}) must be >= RL_EPSILON_END ({RL_EPSILON_END})")
    if RL_BATCH_SIZE > RL_REPLAY_BUFFER_SIZE:
        issues.append(f"RL_BATCH_SIZE ({RL_BATCH_SIZE}) > RL_REPLAY_BUFFER_SIZE ({RL_REPLAY_BUFFER_SIZE})")

    # ✅ Correct check: use actual steps, not the config constant
    actual_steps_per_ep = NUM_USERS * REQUESTS_PER_USER
    expected_total      = RL_TRAINING_EPISODES * actual_steps_per_ep
    if RL_TRAINING_STEPS != expected_total:
        issues.append(
            f"RL_TRAINING_STEPS ({RL_TRAINING_STEPS:,}) != "
            f"EPISODES×ACTUAL_STEPS ({RL_TRAINING_EPISODES}×{actual_steps_per_ep}={expected_total:,}). "
            f"Set RL_TRAINING_STEPS = {expected_total:,}"
        )
    if RL_STEPS_PER_EPISODE != actual_steps_per_ep:
        issues.append(
            f"RL_STEPS_PER_EPISODE ({RL_STEPS_PER_EPISODE:,}) != "
            f"NUM_USERS×REQUESTS_PER_USER ({actual_steps_per_ep:,}). "
            f"Set RL_STEPS_PER_EPISODE = {actual_steps_per_ep:,}"
        )

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
            issues.append(f"RL_PRIORITY_ALPHA ({RL_PRIORITY_ALPHA}) must be in (0,1]")
        if not (0 < RL_PRIORITY_BETA_START <= 1):
            issues.append(f"RL_PRIORITY_BETA_START ({RL_PRIORITY_BETA_START}) must be in (0,1]")
        if RL_PRIORITY_BETA_END != 1.0:
            issues.append(f"RL_PRIORITY_BETA_END should be 1.0 per Schaul et al. (got {RL_PRIORITY_BETA_END})")
    if not (0 < RL_TAU < 1):
        issues.append(f"RL_TAU ({RL_TAU}) must be in (0,1)")
    if RL_REWARD_CIC_ENABLED >= RL_REWARD_CACHE_HIT:
        issues.append(
            f"RL_REWARD_CIC_ENABLED ({RL_REWARD_CIC_ENABLED}) must be < "
            f"RL_REWARD_CACHE_HIT ({RL_REWARD_CACHE_HIT}) to avoid perverse incentive"
        )
    if RL_REWARD_CIC_ENABLED <= RL_REWARD_NOMA_FAILURE:
        issues.append(
            f"RL_REWARD_CIC_ENABLED ({RL_REWARD_CIC_ENABLED}) must be > "
            f"RL_REWARD_NOMA_FAILURE ({RL_REWARD_NOMA_FAILURE})"
        )

    if issues:
        print("⚠️  Configuration Issues Found:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print("✅ Configuration validated successfully")
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
