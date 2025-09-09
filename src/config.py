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