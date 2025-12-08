"""Simulation configuration parameters (optimized baseline)."""

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
CACHE_POLICY = "noma_aware"
CACHE_DELIVERY_RATE = 10.0  # bits/s/Hz

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
NUM_RUNS = 50

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

# ========== RL-SPECIFIC PARAMETERS ==========

# Training configuration
RL_TRAINING_STEPS = 50000     # Total training steps
RL_TRAINING_EPISODES = 50     # Episodes (1000 steps each)
RL_STEPS_PER_EPISODE = 1000

# EPSILON DECAY STRATEGY (IMPORTANT!)
# The agent will:
# - Steps 0-25000: Explore (ε: 1.0 → 0.01) - Learn through exploration
# - Steps 25001-50000: Exploit (ε: 0.01) - Refine learned policy
# This is INTENTIONAL: decay over first half, then exploit learned policy
RL_EPSILON_START = 1.0
RL_EPSILON_END = 0.01
RL_EPSILON_DECAY_STEPS = 25000  # Decay over first HALF of training
RL_EVAL_EPSILON = 0.0            # No exploration during evaluation

# Evaluation configuration
RL_EVAL_REQUESTS = 5000
RL_SEPARATE_TRAIN_EVAL = True

# Neural network configuration
RL_USE_NEURAL_NETWORK = True
RL_HIDDEN_DIMS = [128, 64]
RL_LEARNING_RATE = 0.001
RL_BATCH_SIZE = 64
RL_REPLAY_BUFFER_SIZE = 50000

# Q-learning parameters
RL_GAMMA = 0.95  # Discount factor
RL_TARGET_UPDATE_FREQ = 100
RL_TRAIN_FREQUENCY = 4

# Reward function parameters
RL_REWARD_CACHE_HIT = 10.0
RL_REWARD_CACHE_MISS_SUCCESS = -1.0
RL_REWARD_OUTAGE = -10.0
RL_REWARD_POOR_BER = -3.0
RL_REWARD_GOOD_BER = +3.0
RL_BER_THRESHOLD_GOOD = 1e-4
RL_BER_THRESHOLD_POOR = 1e-2

# Prioritized experience replay
RL_USE_PRIORITIZED_REPLAY = True
RL_PRIORITY_ALPHA = 0.6
RL_PRIORITY_BETA = 0.4

# Performance tracking
RL_TRACK_LEARNING = True
RL_SAVE_CHECKPOINTS = True
RL_CHECKPOINT_FREQ = 10000  # Save every 10k steps
RL_CHECKPOINT_DIR = "./checkpoints/"

# Comparison experiments
COMPARE_POLICIES = [
    "topk",
    "lru",
    "lfu",
    "improved_dqn_noma"
]

# Logging and visualization
RL_VERBOSE = True
RL_PLOT_LEARNING_CURVES = True
RL_SAVE_TRAINING_LOGS = True


# ========== HELPER FUNCTIONS ==========

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
    print("RL CONFIGURATION")
    print("="*70)
    
    config = get_rl_config()
    
    print("\n📚 EPSILON DECAY STRATEGY:")
    print(f"  Total Training Steps: {RL_TRAINING_STEPS}")
    print(f"  Decay Steps: {RL_EPSILON_DECAY_STEPS}")
    print(f"  Epsilon Range: {RL_EPSILON_START} → {RL_EPSILON_END}")
    print(f"\n  Phase 1 (Steps 0-{RL_EPSILON_DECAY_STEPS}):")
    print(f"    Exploration: ε decays from {RL_EPSILON_START} to {RL_EPSILON_END}")
    print(f"    Purpose: Learn from diverse experiences")
    print(f"\n  Phase 2 (Steps {RL_EPSILON_DECAY_STEPS}-{RL_TRAINING_STEPS}):")
    print(f"    Exploitation: ε stays at {RL_EPSILON_END}")
    print(f"    Purpose: Refine learned policy")
    
    categories = {
        'Training': ['TRAINING_STEPS', 'BATCH_SIZE', 'LEARNING_RATE'],
        'Evaluation': ['EVAL_REQUESTS', 'SEPARATE_TRAIN_EVAL', 'EVAL_EPSILON'],
        'Neural Network': ['USE_NEURAL_NETWORK', 'HIDDEN_DIMS'],
        'Q-Learning': ['GAMMA', 'REPLAY_BUFFER_SIZE'],
    }
    
    for category, params in categories.items():
        print(f"\n{category}:")
        for param in params:
            key = f'RL_{param}'
            if key in config:
                print(f"  {param}: {config[key]}")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    print_rl_config()

    