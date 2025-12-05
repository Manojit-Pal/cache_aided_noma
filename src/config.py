"""Simulation configuration parameters (optimized baseline)."""

# Random seed for reproducibility
RANDOM_SEED = 2025

# ------------------------------
# Content catalog
# ------------------------------
NUM_FILES = 2000        # total unique files in catalog (moderate scale, realistic for edge caching)
ZIPF_ALPHA = 1.0        # Zipf skew parameter. 1.0 = strong skew, realistic traffic (few files are very popular)

# ------------------------------
# Users & requests
# ------------------------------
NUM_USERS = 200         # number of users in the cell
REQUESTS_PER_USER = 50  # number of requests per user (enough to get stable statistics)

# ------------------------------
# Cache
# ------------------------------
CACHE_SIZE = 200        # ~10% of catalog size, realistic for small edge server / base station cache

# Caching policy options: "topk", "lru", "lfu", "random"
CACHE_POLICY = "noma_aware"  # Options: "noma_aware", "joint_opt", "multi_obj", "rl"

# Effective rate when served from local cache (bits/s/Hz) — models very fast local delivery
CACHE_DELIVERY_RATE = 10.0


# ------------------------------
# Novel algorithm parameters
# ------------------------------
NOMA_AWARE_ALPHA_POP = 0.3      # Popularity learning rate
NOMA_AWARE_ALPHA_CHANNEL = 0.2  # Channel learning rate
CACHE_UPDATE_INTERVAL = 100      # How often to re-optimize cache
TIME_SLOTS = 1000               # Simulation time slots for learning

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
NUM_RUNS = 50           # repeat experiments for averaging (good statistical confidence)

# ------------------------------
# NOMA & channel params
# ------------------------------
TX_POWER = 1.0               # total transmit power (linear units, normalized)
NOISE_POWER = 1e-9           # noise power (linear)
CELL_RADIUS = 500.0          # meters (urban small-cell range)
PATHLOSS_EXPONENT = 3.5      # urban/suburban environment
MIN_DISTANCE = 1.0           # minimum distance to avoid singular PL

PAIR_USERS = True            # enable NOMA pairing
PAIRING_METHOD = "extreme"   # best performance: pair weakest with strongest

# Power allocation
POWER_COEFF_WEAK = 0.8       # fraction of power allocated to weak user (higher priority)
POWER_COEFF_STRONG = 0.2     # remainder to strong user

# SIC imperfection
SIC_IMPERFECTION = 0.05      # realistic imperfect SIC (5% residual interference)

# QoS requirement
TARGET_RATE_BPS = 0.5        # target rate in bits/s/Hz per user

# Power allocation method
POWER_ALLOC_GRID = 101       # grid-search resolution for allocation
USE_CLOSED_FORM_ALLOC = False # keep False → grid-search is more general

# ========== NEW: RL-SPECIFIC PARAMETERS ==========

# Training configuration
RL_TRAINING_STEPS = 50000  # ⚠️ INCREASED from 1000
RL_TRAINING_EPISODES = 10  # Number of full training episodes
RL_STEPS_PER_EPISODE = 1000

# Evaluation configuration  
RL_EVAL_REQUESTS = 5000  # Requests for evaluation phase
RL_SEPARATE_TRAIN_EVAL = True  # Use separate training/evaluation

# Neural network configuration
RL_USE_NEURAL_NETWORK = True  # Try to use PyTorch DQN (fallback to Q-table)
RL_HIDDEN_DIMS = [128, 64]  # Hidden layer dimensions
RL_LEARNING_RATE = 0.001
RL_BATCH_SIZE = 64
RL_REPLAY_BUFFER_SIZE = 50000  # ⚠️ INCREASED from 2000

# Exploration configuration
RL_EPSILON_START = 1.0  # Start with full exploration
RL_EPSILON_END = 0.01  # End with 1% exploration
RL_EPSILON_DECAY_STEPS = 25000  # Decay over 5000 steps
RL_EVAL_EPSILON = 0.05  # Low exploration during evaluation

# Q-learning parameters
RL_GAMMA = 0.95  # Discount factor
RL_TARGET_UPDATE_FREQ = 100  # Update target network every 100 steps
RL_TRAIN_FREQUENCY = 4  # Train every 4 steps

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
RL_PRIORITY_ALPHA = 0.6  # Priority exponent
RL_PRIORITY_BETA = 0.4  # Importance sampling weight

# Cache update strategy
CACHE_UPDATE_INTERVAL = 100  # Re-optimize cache every 100 requests
TIME_SLOTS = 1000  # Legacy parameter for backward compatibility

# Performance tracking
RL_TRACK_LEARNING = True  # Track and save learning curves
RL_SAVE_CHECKPOINTS = True  # Save model checkpoints
RL_CHECKPOINT_FREQ = 2000  # Save every 2000 steps
RL_CHECKPOINT_DIR = "./checkpoints/"

# Comparison experiments
COMPARE_POLICIES = [
    "topk",  # Baseline 1
    "lru",   # Baseline 2
    "lfu",   # Baseline 3
    "improved_dqn_noma"  # Your RL policy
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
    
    categories = {
        'Training': ['TRAINING_STEPS', 'TRAINING_EPISODES', 'STEPS_PER_EPISODE'],
        'Evaluation': ['EVAL_REQUESTS', 'SEPARATE_TRAIN_EVAL', 'EVAL_EPSILON'],
        'Neural Network': ['USE_NEURAL_NETWORK', 'HIDDEN_DIMS', 'LEARNING_RATE', 'BATCH_SIZE'],
        'Exploration': ['EPSILON_START', 'EPSILON_END', 'EPSILON_DECAY_STEPS'],
        'Q-Learning': ['GAMMA', 'TARGET_UPDATE_FREQ', 'TRAIN_FREQUENCY'],
        'Rewards': ['REWARD_CACHE_HIT', 'REWARD_CACHE_MISS_SUCCESS', 'REWARD_OUTAGE']
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