# src/config.py
"""
Simulation configuration parameters (optimized for Stable DQN Cache).

✅ UPDATED: Optimized parameters for the new stable DQN implementation
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
NUM_RUNS = 10  # ✅ UPDATED: Reduced for faster testing (was 50)

# ------------------------------
# NOMA & channel params
# ------------------------------
TX_POWER = 1.0
NOISE_POWER = 1e-9
CELL_RADIUS = 500.0
PATHLOSS_EXPONENT = 3.5
MIN_DISTANCE = 1.0

PAIR_USERS = True
PAIRING_METHOD = "extreme"

POWER_COEFF_WEAK = 0.8
POWER_COEFF_STRONG = 0.2

SIC_IMPERFECTION = 0.05
TARGET_RATE_BPS = 0.5

POWER_ALLOC_GRID = 101
USE_CLOSED_FORM_ALLOC = False


# ============================================================================
# ✅ STABLE DQN CACHE PARAMETERS (OPTIMIZED)
# ============================================================================

# ------------------------------
# Training Configuration
# ------------------------------
RL_TRAINING_EPISODES = 50        # Number of training episodes
RL_STEPS_PER_EPISODE = 1000      # Requests per episode
RL_TRAINING_STEPS = 50000        # Total training steps (episodes × steps_per_episode)

# ------------------------------
# Epsilon-Greedy Exploration
# ------------------------------
# Strategy: Start with full exploration, decay linearly over first half of training,
#           then exploit learned policy in second half
RL_EPSILON_START = 1.0           # Start with 100% exploration
RL_EPSILON_END = 0.01            # End with 1% exploration (never fully greedy)
RL_EPSILON_DECAY_STEPS = 25000   # Decay over FIRST HALF of training (25k steps)
                                 # Then stays at 0.01 for remaining 25k steps
RL_EVAL_EPSILON = 0.0            # No exploration during evaluation

# Explanation of decay strategy:
# Steps 0-25000:    ε decreases from 1.0 → 0.01 (exploration phase)
# Steps 25001-50000: ε stays at 0.01 (exploitation/refinement phase)
# This prevents premature convergence while allowing policy refinement

# ------------------------------
# Neural Network Architecture
# ------------------------------
RL_USE_NEURAL_NETWORK = True     # Use PyTorch DQN (set False for Q-table fallback)
RL_HIDDEN_DIMS = [128, 64]       # Hidden layer sizes [first_layer, second_layer]
                                 # Larger = more capacity, slower training
                                 # [256, 128] for complex problems
                                 # [64, 32] for faster training

# ------------------------------
# Learning Hyperparameters
# ------------------------------
RL_LEARNING_RATE = 0.0001        # ✅ UPDATED: Lower for stability (was 0.001)
                                 # Lower = more stable, slower convergence
                                 # Higher = faster learning, risk of instability

RL_GAMMA = 0.95                  # Discount factor (0.9-0.99 typical)
                                 # Higher = values long-term rewards more
                                 # Lower = focuses on immediate rewards

RL_BATCH_SIZE = 64               # Batch size for training
                                 # Larger = more stable, slower
                                 # Smaller = faster, more variance
                                 # Typical: 32, 64, 128

RL_REPLAY_BUFFER_SIZE = 50000    # Experience replay buffer size
                                 # Should be >> batch_size
                                 # Larger = more diverse experiences

# ------------------------------
# Training Stability
# ------------------------------
RL_GRADIENT_CLIP = 10.0          # ✅ NEW: Gradient clipping max norm
                                 # Prevents exploding gradients
                                 # Typical: 5.0-10.0

RL_TAU = 0.005                   # ✅ NEW: Soft target network update rate
                                 # Lower = more stable, slower target updates
                                 # Higher = faster convergence, less stable
                                 # Typical: 0.001-0.01

RL_TARGET_UPDATE_FREQ = 1000     # Hard target update frequency (steps)
                                 # Only used if tau=1.0 (hard updates)
                                 # Soft updates (tau<1) are preferred

RL_TRAIN_FREQUENCY = 4           # Train every N steps
                                 # Lower = more training, slower
                                 # Higher = faster, less learning

# ------------------------------
# Reward Function Parameters
# ------------------------------
# ✅ UPDATED: Balanced reward structure for stable learning
RL_REWARD_CACHE_HIT = 10.0       # Reward for cache hit (was 50.0)
RL_REWARD_CACHE_MISS_SUCCESS = -1.0   # Miss but NOMA succeeded
RL_REWARD_NOMA_FAILURE = -5.0    # ✅ NEW: Miss and NOMA failed
RL_REWARD_OUTAGE = -5.0          # Miss and outage occurred
RL_REWARD_POOR_BER = -2.0        # ✅ UPDATED: Additional penalty for high BER
RL_REWARD_GOOD_BER = 1.0         # ✅ UPDATED: Bonus for good BER

# BER thresholds for reward shaping
RL_BER_THRESHOLD_GOOD = 1e-4     # BER below this = good quality
RL_BER_THRESHOLD_POOR = 1e-2     # BER above this = poor quality

# Reward balance explanation:
# Cache hit:       +10  (clear positive signal)
# Miss + success:  -1   (small penalty, content delivered)
# Miss + failure:  -5   (moderate penalty, bad outcome)
# Outage:          -5   (clear negative signal)
# This creates ~10:1 positive:negative ratio for effective learning

# ------------------------------
# Prioritized Experience Replay
# ------------------------------
RL_USE_PRIORITIZED_REPLAY = True # ✅ NEW: Use prioritized replay (recommended)
RL_PRIORITY_ALPHA = 0.6          # Priority exponent (0=uniform, 1=full priority)
RL_PRIORITY_BETA = 0.4           # Importance sampling exponent
                                 # Compensates for bias from prioritization

# ------------------------------
# Evaluation Configuration
# ------------------------------
RL_EVAL_REQUESTS = 5000          # Number of requests per evaluation run
RL_SEPARATE_TRAIN_EVAL = True    # Use separate eval phase (recommended)

# ------------------------------
# Checkpointing & Logging
# ------------------------------
RL_SAVE_CHECKPOINTS = True       # Save model checkpoints during training
RL_CHECKPOINT_FREQ = 10000       # Save every N steps
RL_CHECKPOINT_DIR = "./checkpoints/"  # Directory for checkpoints

RL_TRACK_LEARNING = True         # Track and log training metrics
RL_VERBOSE = True                # Print training progress
RL_PLOT_LEARNING_CURVES = True   # Generate learning curve plots
RL_SAVE_TRAINING_LOGS = True     # Save training logs to CSV

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
    
    RL_TRAINING_EPISODES = 10
    RL_STEPS_PER_EPISODE = 500
    RL_TRAINING_STEPS = 5000
    RL_EPSILON_DECAY_STEPS = 2500
    NUM_RUNS = 3
    RL_EVAL_REQUESTS = 1000
    
    print("⚡ Quick Test Config Enabled")
    print(f"   Training: {RL_TRAINING_EPISODES} episodes × {RL_STEPS_PER_EPISODE} steps = {RL_TRAINING_STEPS} steps")
    print(f"   Evaluation: {NUM_RUNS} runs × {RL_EVAL_REQUESTS} requests")


def set_full_experiment_config():
    """
    Full experiment configuration for paper results (60-90 minutes).
    """
    global RL_TRAINING_EPISODES, RL_STEPS_PER_EPISODE, RL_TRAINING_STEPS
    global RL_EPSILON_DECAY_STEPS, NUM_RUNS, RL_EVAL_REQUESTS
    
    RL_TRAINING_EPISODES = 50
    RL_STEPS_PER_EPISODE = 1000
    RL_TRAINING_STEPS = 50000
    RL_EPSILON_DECAY_STEPS = 25000
    NUM_RUNS = 50
    RL_EVAL_REQUESTS = 5000
    
    print("🎓 Full Experiment Config Enabled")
    print(f"   Training: {RL_TRAINING_EPISODES} episodes × {RL_STEPS_PER_EPISODE} steps = {RL_TRAINING_STEPS} steps")
    print(f"   Evaluation: {NUM_RUNS} runs × {RL_EVAL_REQUESTS} requests")


def set_aggressive_learning_config():
    """
    Aggressive learning configuration (faster convergence, less stable).
    """
    global RL_LEARNING_RATE, RL_BATCH_SIZE, RL_HIDDEN_DIMS, RL_EPSILON_DECAY_STEPS
    
    RL_LEARNING_RATE = 0.0005
    RL_BATCH_SIZE = 128
    RL_HIDDEN_DIMS = [256, 128]
    RL_EPSILON_DECAY_STEPS = 10000
    
    print("⚡ Aggressive Learning Config Enabled")


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


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_rl_config():
    """Return RL-specific configuration as dictionary."""
    import sys
    module = sys.modules[__name__]
    
    rl_params = {}
    for attr in dir(module):
        if attr.startswith('RL_'):
            rl_params[attr] = getattr(module, attr)
    
    return rl_params


def print_rl_config():
    """Print RL configuration for verification."""
    print("\n" + "="*70)
    print("STABLE DQN CACHE CONFIGURATION")
    print("="*70)
    
    config = get_rl_config()
    
    print("\n📚 TRAINING STRATEGY:")
    print(f"  Total Training Steps: {RL_TRAINING_STEPS}")
    print(f"  Episodes: {RL_TRAINING_EPISODES} × {RL_STEPS_PER_EPISODE} steps")
    
    print("\n📉 EPSILON DECAY:")
    print(f"  Start: {RL_EPSILON_START} (full exploration)")
    print(f"  End: {RL_EPSILON_END} (minimal exploration)")
    print(f"  Decay over: {RL_EPSILON_DECAY_STEPS} steps")
    print(f"\n  Phase 1 (Steps 0-{RL_EPSILON_DECAY_STEPS}):")
    print(f"    Exploration: ε decays from {RL_EPSILON_START} to {RL_EPSILON_END}")
    print(f"  Phase 2 (Steps {RL_EPSILON_DECAY_STEPS+1}-{RL_TRAINING_STEPS}):")
    print(f"    Exploitation: ε stays at {RL_EPSILON_END}")
    
    print("\n🧠 NEURAL NETWORK:")
    print(f"  Architecture: {RL_HIDDEN_DIMS}")
    print(f"  Learning Rate: {RL_LEARNING_RATE}")
    print(f"  Batch Size: {RL_BATCH_SIZE}")
    print(f"  Replay Buffer: {RL_REPLAY_BUFFER_SIZE}")
    
    print("\n⚖️  REWARD STRUCTURE:")
    print(f"  Cache Hit: +{RL_REWARD_CACHE_HIT}")
    print(f"  Miss + Success: {RL_REWARD_CACHE_MISS_SUCCESS}")
    print(f"  Miss + Failure: {RL_REWARD_NOMA_FAILURE}")
    print(f"  Outage: {RL_REWARD_OUTAGE}")
    
    print("\n🎯 STABILITY:")
    print(f"  Gradient Clip: {RL_GRADIENT_CLIP}")
    print(f"  Target Update (τ): {RL_TAU}")
    print(f"  Train Frequency: every {RL_TRAIN_FREQUENCY} steps")
    print(f"  Prioritized Replay: {RL_USE_PRIORITIZED_REPLAY}")
    
    print("\n📊 EVALUATION:")
    print(f"  Requests per Run: {RL_EVAL_REQUESTS}")
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
    print_rl_config()
    validate_config()
else:
    # Auto-validate when imported
    validate_config()



    