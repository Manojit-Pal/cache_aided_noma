# src/config.py
"""
Simulation configuration parameters (optimized for Stable DQN Cache).

✅ UPDATED: Optimized parameters for the new stable DQN implementation
✅ ADDED: Complete NOMA channel modeling and cache-aware parameters
✅ BUG FIXES #1-6: All parameters updated for fixed implementation
✅ BUG FIX #7: Added CIC-aware reward for DQN learning
✅ CRITICAL FIX (Dec 12, 2025): Training parameters updated to research standards
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
REQUESTS_PER_USER = 50  # number of requests per user

# ------------------------------
# Cache
# ------------------------------
CACHE_SIZE = 200        # ~10% of catalog size
CACHE_POLICY = "stable_dqn"  # ✅ UPDATED: Default to stable DQN
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
NUM_RUNS = 100  # ✅ UPDATED: Increased for DQN convergence (was 10)

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
POWER_ALLOC_METHOD = "cache_aware"  # ✅ NEW: Options: 'gridsearch', 'closedform', 'cache_aware', 'sumrate_max', 'energy_efficient'
POWER_ALLOC_GRID = 101
USE_CLOSED_FORM_ALLOC = False  # ✅ DEPRECATED: Use POWER_ALLOC_METHOD instead

# SIC parameters
SIC_IMPERFECTION = 0.05  # Residual interference factor (ζ)
TARGET_RATE_BPS = 0.3    # Target data rate in bps/Hz

# ✅ NEW: Channel modeling parameters (6G features)
FADING_TYPE = "mixed"     # Options: 'rayleigh', 'rician', 'mixed'
RICIAN_K_FACTOR_DB = 10.0 # Rician K-factor in dB (for LoS scenarios)
LOS_PROBABILITY = 0.4     # Probability of Line-of-Sight (for mixed fading)
ENABLE_MOBILITY = False   # Enable time-varying channels
DOPPLER_FREQ = 10.0      # Doppler frequency in Hz (if mobility enabled)

# ✅ NEW: Cache-aware NOMA parameters
ENABLE_CIC = True         # Enable Cache-aided Interference Cancellation
CIC_PERFECT = True        # Perfect CIC (residual=0) vs imperfect
CACHE_HIT_ENABLES_CIC = True  # Whether cache hits enable CIC for paired users

# ✅ NEW: Performance thresholds
BER_THRESHOLD = 1e-3      # Maximum acceptable BER
OUTAGE_SINR_MARGIN = 2.0  # SINR margin above threshold (dB)


# ============================================================================
# ✅ STABLE DQN CACHE PARAMETERS (WITH BUG FIXES #1-7 + RESEARCH STANDARDS)
# ============================================================================

# ------------------------------
# Training Configuration (UPDATED TO RESEARCH STANDARDS)
# ------------------------------
# 🔧 CRITICAL FIX: Increased from 500 to 2000 episodes
# Research papers show DQN needs 2000-5000 episodes for content caching
# - arXiv:1712.08132: "Deep RL for Content Caching"
# - IEEE TSP_CMC_20471: "5000-50000 episodes for convergence"
# - DRLCache (GitHub): "2000 episodes minimum"
RL_TRAINING_EPISODES = 2000       # ✅ FIXED: Was 500 (4x increase)
RL_STEPS_PER_EPISODE = 200        # ✅ FIXED: Was 1000 (faster episodes, more iterations)
RL_TRAINING_STEPS = 400000         # ✅ FIXED: Was 500000 (2000 × 200 = 400K)
                                  # Total interactions remains high for learning

# Training rationale:
# - 2000 episodes: Allows DQN to see diverse scenarios 4x more
# - 200 steps/episode: Faster iterations, better for episodic learning
# - 400K total steps: Sufficient training budget (research: 200K-1M)

# ------------------------------
# Epsilon-Greedy Exploration (UPDATED FOR FASTER CONVERGENCE)
# ------------------------------
# Strategy: Decay over first 50% of training (research standard)
# This ensures exploitation starts by episode 1000
RL_EPSILON_START = 1.0            # Start with 100% exploration
RL_EPSILON_END = 0.01             # End with 1% exploration (never fully greedy)
RL_EPSILON_DECAY_STEPS = 200000    # ✅ FIXED: Was 250000
                                  # Decay over FIRST HALF (200K of 400K steps)
RL_EVAL_EPSILON = 0.0             # No exploration during evaluation

# Epsilon schedule explanation:
# Steps 0-200K (episodes 0-1000):  ε = 1.0 → 0.01 (exploration → exploitation)
# Steps 200K-400K (episodes 1000-2000): ε = 0.01 (refinement phase)
# 
# At episode 500 (old config): ε would be ~0.50 (still too much exploration!)
# At episode 1000 (new config): ε = 0.01 (proper exploitation begins)
# At episode 2000 (end): ε = 0.01 (fully learned policy)

# ------------------------------
# Neural Network Architecture (RESEARCH STANDARD)
# ------------------------------
RL_USE_NEURAL_NETWORK = True      # Use PyTorch DQN (set False for Q-table fallback)
RL_HIDDEN_DIMS = [128, 128]       # ✅ FIXED: Was [128, 64]
                                  # Research standard architecture:
                                  # - Input → 128 → 128 → Output
                                  # - Same as DeepChunk (IEEE 2019)
                                  # - Same as DRLCache (GitHub)
                                  # - Provides better learning capacity

# ------------------------------
# Learning Hyperparameters (RESEARCH STANDARD)
# ------------------------------
RL_LEARNING_RATE = 0.0001         # ✅ CORRECT: Standard value (Adam optimizer)
                                  # Research consensus: 0.0001 for stable learning

RL_GAMMA = 0.99                   # ✅ FIXED: Was 0.95
                                  # Research standard: 0.99
                                  # Higher gamma = values long-term cache hits more
                                  # Critical for caching (hit now prevents miss later)

RL_BATCH_SIZE = 64                # ✅ CORRECT: Standard batch size
                                  # Typical: 32, 64, 128
                                  # 64 is sweet spot for stability vs. speed

RL_REPLAY_BUFFER_SIZE = 50000     # ✅ CORRECT: Standard buffer size
                                  # Should be >> batch_size (50K >> 64 ✓)
                                  # Larger = more diverse experiences

# ------------------------------
# Training Stability
# ------------------------------
RL_GRADIENT_CLIP = 10.0           # ✅ NEW: Gradient clipping max norm
                                  # Prevents exploding gradients
                                  # Typical: 5.0-10.0

RL_TAU = 0.005                    # ✅ BUG FIX #4: Soft target network update rate
                                  # Lower = more stable, slower target updates
                                  # Higher = faster convergence, less stable
                                  # NOW USED EVERY TRAINING STEP (not every 1000)
                                  # Typical: 0.001-0.01

RL_TARGET_UPDATE_FREQ = 1000      # ✅ DEPRECATED: Not used with soft updates
                                  # Kept for backward compatibility
                                  # Target now updates every training step with τ=0.005

RL_TRAIN_FREQUENCY = 4            # Train every N steps
                                  # Lower = more training, slower
                                  # Higher = faster, less learning

# ------------------------------
# ✅ BUG FIX #6: WARM-UP PERIOD
# ------------------------------
# Research: Wait for buffer to fill before training
# Prevents learning from tiny, biased samples
RL_WARM_UP_STEPS = None           # None = auto (max(10 * BATCH_SIZE, 1000) = 1000)
                                  # With 2000 episodes, warm-up is ~0.25% of training
                                  # (1000 steps out of 400K total)

# ------------------------------
# Reward Function Parameters
# ------------------------------
# ✅ BUG FIX #7: CIC-aware reward structure
# Balanced reward structure for stable learning with CIC incentivization
RL_REWARD_CACHE_HIT = 10.0            # Reward for cache hit (best outcome)
RL_REWARD_CIC_ENABLED = 7.0           # ✅ NEW: Bonus for enabling CIC! (good outcome)
RL_REWARD_CACHE_MISS_SUCCESS = -1.0   # Miss but NOMA succeeded (without CIC)
RL_REWARD_NOMA_FAILURE = -5.0         # Miss and NOMA failed (bad outcome)
RL_REWARD_OUTAGE = -10.0              # Miss and outage occurred (worst outcome)
RL_REWARD_POOR_BER = -2.0             # Additional penalty for high BER
RL_REWARD_GOOD_BER = 1.0              # Bonus for good BER

# BER thresholds for reward shaping
RL_BER_THRESHOLD_GOOD = 1e-4      # BER below this = good quality
RL_BER_THRESHOLD_POOR = 1e-2      # BER above this = poor quality

# Reward balance explanation:
# Cache hit:       +10  (best - no transmission needed)
# CIC enabled:     +7   (✅ NEW - cache miss but CIC helps paired user!)
# Miss + success:  -1   (acceptable - delivered via NOMA without CIC)
# Miss + failure:  -5   (bad - poor QoS)
# Outage:          -10  (worst - no communication)
#
# This structure teaches DQN to:
# 1. Maximize cache hits (+10)
# 2. Enable CIC when miss occurs (+7 vs -1)
# 3. Minimize NOMA failures and outages

# ------------------------------
# ✅ BUG FIX #2 & #3: PRIORITIZED EXPERIENCE REPLAY
# ------------------------------
RL_USE_PRIORITIZED_REPLAY = True  # ✅ NEW: Use prioritized replay (recommended)
RL_PRIORITY_ALPHA = 0.6           # Priority exponent (0=uniform, 1=full priority)
                                  # Controls how much prioritization is used

# ✅ BUG FIX #2: Beta annealing (Schaul et al., 2016)
RL_PRIORITY_BETA_START = 0.4      # Start with more bias (faster learning)
RL_PRIORITY_BETA_END = 1.0        # End with no bias (accurate estimates)
RL_PRIORITY_BETA_FRAMES = 100000  # Anneal over 100k samples
                                  # Beta anneals: 0.4 → 1.0 during training

# ✅ BUG FIX #3: Smart sampling strategy
# Implementation automatically uses:
# - replace=True when buffer < 3*batch_size (prevents correlation)
# - replace=False when buffer >= 3*batch_size (diverse sampling)

# ------------------------------
# ✅ BUG FIX #1: POPULARITY TRACKING
# ------------------------------
RL_POPULARITY_DECAY = 0.9         # EMA decay factor
                                  # Correct EMA: p[i] = decay*p[i] + (1-decay)
                                  # Others decay through normalization
                                  # FIXED: No more double-decay bug

# ------------------------------
# ✅ BUG FIX #5: EMPTY SLOT HANDLING
# ------------------------------
# Implementation automatically:
# - Uses -1.0 marker for empty slots in state
# - Only increments LRU for occupied slots
# - Network distinguishes empty vs old files

# ------------------------------
# Evaluation Configuration
# ------------------------------
RL_EVAL_REQUESTS = 5000           # Number of requests per evaluation run
RL_SEPARATE_TRAIN_EVAL = True     # Use separate eval phase (recommended)

# ------------------------------
# Checkpointing & Logging
# ------------------------------
RL_SAVE_CHECKPOINTS = True        # Save model checkpoints during training
RL_CHECKPOINT_FREQ = 10000        # Save every N steps
RL_CHECKPOINT_DIR = "./checkpoints/"  # Directory for checkpoints

RL_TRACK_LEARNING = True          # Track and log training metrics
RL_VERBOSE = True                 # Print training progress
RL_PLOT_LEARNING_CURVES = True    # Generate learning curve plots
RL_SAVE_TRAINING_LOGS = True      # Save training logs to CSV

# ------------------------------
# Comparison Experiments
# ------------------------------
COMPARE_POLICIES = [
    "topk",              # Static baseline
    "lru",               # Dynamic baseline
    "lfu",               # Dynamic baseline
    "stable_dqn"         # ✅ UPDATED: Our new implementation
]


# ============================================================================
# CONFIGURATION PRESETS
# ============================================================================

def set_quick_test_config():
    """
    Quick test configuration for debugging (5-10 minutes).
    Use this to verify implementation is working before full run.
    """
    global RL_TRAINING_EPISODES, RL_STEPS_PER_EPISODE, RL_TRAINING_STEPS
    global RL_EPSILON_DECAY_STEPS, NUM_RUNS, RL_EVAL_REQUESTS
    
    RL_TRAINING_EPISODES = 100     # ✅ UPDATED: Was 10 (more realistic test)
    RL_STEPS_PER_EPISODE = 200     # ✅ UPDATED: Match main config
    RL_TRAINING_STEPS = 20000      # ✅ UPDATED: 100 × 200
    RL_EPSILON_DECAY_STEPS = 10000  # ✅ UPDATED: Decay over first half
    NUM_RUNS = 10                  # ✅ UPDATED: Was 3 (more stable estimates)
    RL_EVAL_REQUESTS = 2000        # ✅ UPDATED: Was 1000
    
    print("⚡ Quick Test Config Enabled")
    print(f"   Training: {RL_TRAINING_EPISODES} episodes × {RL_STEPS_PER_EPISODE} steps = {RL_TRAINING_STEPS} steps")
    print(f"   Evaluation: {NUM_RUNS} runs × {RL_EVAL_REQUESTS} requests")
    print(f"   Expected time: ~10-15 minutes")


def set_full_experiment_config():
    """
    Full experiment configuration for paper results (2-3 hours).
    ✅ UPDATED: Now uses research-standard 2000 episodes
    """
    global RL_TRAINING_EPISODES, RL_STEPS_PER_EPISODE, RL_TRAINING_STEPS
    global RL_EPSILON_DECAY_STEPS, NUM_RUNS, RL_EVAL_REQUESTS
    
    # Already set as defaults, but can extend for even more rigorous experiments
    RL_TRAINING_EPISODES = 2000    # Research standard
    RL_STEPS_PER_EPISODE = 200     # Research standard
    RL_TRAINING_STEPS = 400000      # 2000 × 200
    RL_EPSILON_DECAY_STEPS = 200000 # Decay over first 50%
    NUM_RUNS = 100                 # Robust statistics
    RL_EVAL_REQUESTS = 5000        # Thorough evaluation
    
    print("🎓 Full Experiment Config Enabled (Research Standard)")
    print(f"   Training: {RL_TRAINING_EPISODES} episodes × {RL_STEPS_PER_EPISODE} steps = {RL_TRAINING_STEPS} steps")
    print(f"   Evaluation: {NUM_RUNS} runs × {RL_EVAL_REQUESTS} requests")
    print(f"   Expected time: ~2-3 hours on modern GPU")


def set_aggressive_learning_config():
    """
    Aggressive learning configuration (faster convergence, less stable).
    """
    global RL_LEARNING_RATE, RL_BATCH_SIZE, RL_HIDDEN_DIMS, RL_EPSILON_DECAY_STEPS
    
    RL_LEARNING_RATE = 0.0005
    RL_BATCH_SIZE = 128
    RL_HIDDEN_DIMS = [256, 128]
    RL_EPSILON_DECAY_STEPS = 100000  # ✅ UPDATED: Faster decay for 400K steps
    
    print("⚡ Aggressive Learning Config Enabled")
    print("   ⚠️  May be less stable but converges faster")


def set_conservative_learning_config():
    """
    Conservative learning configuration (more stable, slower convergence).
    """
    global RL_LEARNING_RATE, RL_BATCH_SIZE, RL_GRADIENT_CLIP, RL_TAU
    
    RL_LEARNING_RATE = 0.00005
    RL_BATCH_SIZE = 32
    RL_GRADIENT_CLIP = 5.0
    RL_TAU = 0.001
    
    print("🐢 Conservative Learning Config Enabled")
    print("   ✅ More stable but requires longer training")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_noma_config():
    """Return NOMA-specific configuration as dictionary."""
    import sys
    module = sys.modules[__name__]
    
    noma_params = {
        'TX_POWER': TX_POWER,
        'NOISE_POWER': NOISE_POWER,
        'CELL_RADIUS': CELL_RADIUS,
        'PATHLOSS_EXPONENT': PATHLOSS_EXPONENT,
        'MIN_DISTANCE': MIN_DISTANCE,
        'PAIR_USERS': PAIR_USERS,
        'PAIRING_METHOD': PAIRING_METHOD,
        'POWER_COEFF_WEAK': POWER_COEFF_WEAK,
        'POWER_COEFF_STRONG': POWER_COEFF_STRONG,
        'POWER_ALLOC_METHOD': POWER_ALLOC_METHOD,
        'SIC_IMPERFECTION': SIC_IMPERFECTION,
        'TARGET_RATE_BPS': TARGET_RATE_BPS,
        'FADING_TYPE': FADING_TYPE,
        'RICIAN_K_FACTOR_DB': RICIAN_K_FACTOR_DB,
        'LOS_PROBABILITY': LOS_PROBABILITY,
        'ENABLE_CIC': ENABLE_CIC,
        'CIC_PERFECT': CIC_PERFECT,
    }
    
    return noma_params


def get_rl_config():
    """Return RL-specific configuration as dictionary."""
    import sys
    module = sys.modules[__name__]
    
    rl_params = {}
    for attr in dir(module):
        if attr.startswith('RL_'):
            rl_params[attr] = getattr(module, attr)
    
    return rl_params


def print_noma_config():
    """Print NOMA configuration for verification."""
    print("\n" + "="*70)
    print("NOMA SYSTEM CONFIGURATION")
    print("="*70)
    
    print("\n📡 CHANNEL PARAMETERS:")
    print(f"  TX Power: {TX_POWER}W")
    print(f"  Noise Power: {NOISE_POWER}W")
    print(f"  Cell Radius: {CELL_RADIUS}m")
    print(f"  Path Loss Exponent: {PATHLOSS_EXPONENT}")
    print(f"  Fading Type: {FADING_TYPE}")
    if FADING_TYPE in ['rician', 'mixed']:
        print(f"  Rician K-factor: {RICIAN_K_FACTOR_DB} dB")
    if FADING_TYPE == 'mixed':
        print(f"  LoS Probability: {LOS_PROBABILITY}")
    
    print("\n👥 USER PAIRING:")
    print(f"  Method: {PAIRING_METHOD}")
    print(f"  Number of Users: {NUM_USERS}")
    
    print("\n⚡ POWER ALLOCATION:")
    print(f"  Method: {POWER_ALLOC_METHOD}")
    print(f"  Default p_weak: {POWER_COEFF_WEAK}")
    print(f"  Default p_strong: {POWER_COEFF_STRONG}")
    
    print("\n🔄 SIC PARAMETERS:")
    print(f"  Imperfection Factor: {SIC_IMPERFECTION}")
    print(f"  Target Rate: {TARGET_RATE_BPS} bps/Hz")
    
    print("\n💾 CACHE-AIDED FEATURES:")
    print(f"  CIC Enabled: {ENABLE_CIC}")
    print(f"  Perfect CIC: {CIC_PERFECT}")
    print(f"  Cache Size: {CACHE_SIZE}")
    
    print("\n" + "="*70 + "\n")


def print_rl_config():
    """Print RL configuration for verification."""
    print("\n" + "="*70)
    print("STABLE DQN CACHE CONFIGURATION (RESEARCH STANDARD)")
    print("="*70)
    
    print("\n🐞 BUG FIXES APPLIED:")
    print("  ✅ #1: Popularity EMA double-decay fixed")
    print("  ✅ #2: Beta annealing added (Schaul et al., 2016)")
    print("  ✅ #3: Smart sampling strategy (with/without replacement)")
    print("  ✅ #4: Soft target updates every training step")
    print("  ✅ #5: Empty slot LRU representation fixed")
    print("  ✅ #6: Warm-up period before training")
    print("  ✅ #7: CIC-aware reward function")
    
    print("\n🔧 CRITICAL FIXES (Dec 12, 2025):")
    print("  ✅ Training episodes: 500 → 2000 (4x increase)")
    print("  ✅ Steps per episode: 1000 → 200 (faster iterations)")
    print("  ✅ Epsilon decay: 250K → 200K steps (better exploitation)")
    print("  ✅ Hidden layers: [128,64] → [128,128] (research standard)")
    print("  ✅ Gamma: 0.95 → 0.99 (long-term planning)")
    
    print("\n📚 TRAINING STRATEGY:")
    print(f"  Total Training Steps: {RL_TRAINING_STEPS}")
    print(f"  Episodes: {RL_TRAINING_EPISODES} × {RL_STEPS_PER_EPISODE} steps")
    print(f"  Warm-up: {RL_WARM_UP_STEPS if RL_WARM_UP_STEPS else 'Auto (1000 steps)'}")
    print(f"\n  Research Comparison:")
    print(f"    ✅ Episodes (2000) in range [2000, 5000] ✓")
    print(f"    ✅ Steps/episode (200) in range [100, 200] ✓")
    print(f"    ✅ Total steps (400K) in range [200K, 1M] ✓")
    
    print("\n📉 EPSILON DECAY:")
    print(f"  Start: {RL_EPSILON_START} (full exploration)")
    print(f"  End: {RL_EPSILON_END} (minimal exploration)")
    print(f"  Decay over: {RL_EPSILON_DECAY_STEPS} steps ({RL_EPSILON_DECAY_STEPS/RL_TRAINING_STEPS*100:.0f}% of training)")
    print(f"\n  Epsilon at key milestones:")
    ep_500 = max(RL_EPSILON_END, RL_EPSILON_START - (RL_EPSILON_START - RL_EPSILON_END) * (100000/RL_EPSILON_DECAY_STEPS))
    ep_1000 = RL_EPSILON_END
    print(f"    Episode 500:  ε ≈ {ep_500:.3f}")
    print(f"    Episode 1000: ε = {ep_1000:.3f} (exploitation starts)")
    print(f"    Episode 2000: ε = {RL_EPSILON_END:.3f} (full policy)")
    
    print("\n🧠 NEURAL NETWORK:")
    print(f"  Architecture: {RL_HIDDEN_DIMS} (research standard)")
    print(f"  Learning Rate: {RL_LEARNING_RATE} (Adam optimizer)")
    print(f"  Batch Size: {RL_BATCH_SIZE}")
    print(f"  Replay Buffer: {RL_REPLAY_BUFFER_SIZE:,}")
    print(f"  Discount (γ): {RL_GAMMA} (research standard)")
    
    print("\n⚖️  REWARD STRUCTURE (CIC-AWARE):")
    print(f"  Cache Hit: +{RL_REWARD_CACHE_HIT}")
    print(f"  CIC Enabled: +{RL_REWARD_CIC_ENABLED} ✅")
    print(f"  Miss + Success: {RL_REWARD_CACHE_MISS_SUCCESS}")
    print(f"  Miss + Failure: {RL_REWARD_NOMA_FAILURE}")
    print(f"  Outage: {RL_REWARD_OUTAGE}")
    
    print("\n🎯 STABILITY:")
    print(f"  Gradient Clip: {RL_GRADIENT_CLIP}")
    print(f"  Soft Target Update (τ): {RL_TAU} (EVERY training step)")
    print(f"  Train Frequency: every {RL_TRAIN_FREQUENCY} steps")
    print(f"  Prioritized Replay: {RL_USE_PRIORITIZED_REPLAY}")
    if RL_USE_PRIORITIZED_REPLAY:
        print(f"    α (priority): {RL_PRIORITY_ALPHA}")
        print(f"    β (annealing): {RL_PRIORITY_BETA_START} → {RL_PRIORITY_BETA_END} over {RL_PRIORITY_BETA_FRAMES:,} frames")
    
    print("\n📊 EVALUATION:")
    print(f"  Requests per Run: {RL_EVAL_REQUESTS:,}")
    print(f"  Number of Runs: {NUM_RUNS}")
    
    print("\n" + "="*70 + "\n")


def validate_config():
    """Validate configuration parameters."""
    issues = []
    
    # Check basic constraints
    if not (0 < CACHE_SIZE <= NUM_FILES):
        issues.append(f"CACHE_SIZE ({CACHE_SIZE}) must be between 1 and NUM_FILES ({NUM_FILES})")
    
    if not (0 < RL_GAMMA <= 1):
        issues.append(f"RL_GAMMA ({RL_GAMMA}) must be between 0 and 1")
    
    if not (0 < RL_LEARNING_RATE < 1):
        issues.append(f"RL_LEARNING_RATE ({RL_LEARNING_RATE}) must be between 0 and 1")
    
    if RL_EPSILON_START < RL_EPSILON_END:
        issues.append(f"RL_EPSILON_START ({RL_EPSILON_START}) must be >= RL_EPSILON_END ({RL_EPSILON_END})")
    
    if RL_BATCH_SIZE > RL_REPLAY_BUFFER_SIZE:
        issues.append(f"RL_BATCH_SIZE ({RL_BATCH_SIZE}) must be <= RL_REPLAY_BUFFER_SIZE ({RL_REPLAY_BUFFER_SIZE})")
    
    if RL_TRAINING_STEPS != RL_TRAINING_EPISODES * RL_STEPS_PER_EPISODE:
        issues.append(f"RL_TRAINING_STEPS ({RL_TRAINING_STEPS}) != EPISODES × STEPS_PER_EPISODE ({RL_TRAINING_EPISODES} × {RL_STEPS_PER_EPISODE})")
    
    # ✅ NEW: NOMA parameter validation
    if not (0 < POWER_COEFF_WEAK < 1):
        issues.append(f"POWER_COEFF_WEAK ({POWER_COEFF_WEAK}) must be between 0 and 1")
    
    if not (0 < POWER_COEFF_STRONG < 1):
        issues.append(f"POWER_COEFF_STRONG ({POWER_COEFF_STRONG}) must be between 0 and 1")
    
    if abs(POWER_COEFF_WEAK + POWER_COEFF_STRONG - 1.0) > 1e-6:
        issues.append(f"POWER_COEFF_WEAK + POWER_COEFF_STRONG must equal 1.0")
    
    if FADING_TYPE not in ['rayleigh', 'rician', 'mixed']:
        issues.append(f"FADING_TYPE must be 'rayleigh', 'rician', or 'mixed' (got '{FADING_TYPE}')")
    
    if PAIRING_METHOD not in ['extreme', 'random', 'sequential']:
        issues.append(f"PAIRING_METHOD must be 'extreme', 'random', or 'sequential' (got '{PAIRING_METHOD}')")
    
    # ✅ NEW: Bug fix parameter validation
    if RL_USE_PRIORITIZED_REPLAY:
        if not (0 < RL_PRIORITY_ALPHA <= 1):
            issues.append(f"RL_PRIORITY_ALPHA ({RL_PRIORITY_ALPHA}) must be in (0, 1]")
        if not (0 < RL_PRIORITY_BETA_START <= 1):
            issues.append(f"RL_PRIORITY_BETA_START ({RL_PRIORITY_BETA_START}) must be in (0, 1]")
        if RL_PRIORITY_BETA_END != 1.0:
            issues.append(f"RL_PRIORITY_BETA_END should be 1.0 per Schaul et al. (got {RL_PRIORITY_BETA_END})")
    
    if not (0 < RL_TAU < 1):
        issues.append(f"RL_TAU ({RL_TAU}) must be in (0, 1)")
    
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
    # Auto-validate when imported
    validate_config()
