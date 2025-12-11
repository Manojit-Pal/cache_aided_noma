# src/noma/__init__.py
"""
NOMA Module for Cache-Aided NOMA in 6G Networks

This module provides complete NOMA (Non-Orthogonal Multiple Access) functionality:
- Channel modeling (path loss, fading, CSI)
- User pairing strategies
- Power allocation algorithms
- Successive Interference Cancellation (SIC)
- Cache-aided interference cancellation (CIC)
- System-level simulation

Author: Cache-Aided NOMA Team
Date: December 2025
"""

# Import all submodules
from . import channel_model
from . import noma_base
from . import power_allocation
from . import sic

# Export commonly used functions for easy access
from .channel_model import (
    generate_user_positions,
    compute_channel_gains,
    pathloss,
    rayleigh_gain,
    rician_gain,
    mixed_fading_gain,
    TimeVaryingChannel
)

from .noma_base import (
    simulate_noma_pair,
    simulate_noma_system,
    pair_users,
    pair_users_extreme,
    pair_users_random,
    pair_users_sequential,
    sinr_threshold_from_rate,
    rate_from_sinr
)

from .power_allocation import (
    allocate_power,
    allocate_power_gridsearch,
    allocate_power_closedform,
    allocate_power_cache_aware,
    allocate_power_sumrate_max,
    allocate_power_energy_efficient
)

from .sic import (
    sinr_weak_user,
    sinr_strong_decode_weak,
    sinr_strong_after_sic,
    sinr_weak_user_with_cache,
    sinr_strong_after_perfect_sic,
    compute_residual_interference,
    simulate_sic_process
)

# Define what's available when using "from noma import *"
__all__ = [
    # Submodules
    'channel_model',
    'noma_base',
    'power_allocation',
    'sic',
    
    # Channel model
    'generate_user_positions',
    'compute_channel_gains',
    'pathloss',
    'rayleigh_gain',
    'rician_gain',
    'mixed_fading_gain',
    'TimeVaryingChannel',
    
    # NOMA base
    'simulate_noma_pair',
    'simulate_noma_system',
    'pair_users',
    'pair_users_extreme',
    'pair_users_random',
    'pair_users_sequential',
    'sinr_threshold_from_rate',
    'rate_from_sinr',
    
    # Power allocation
    'allocate_power',
    'allocate_power_gridsearch',
    'allocate_power_closedform',
    'allocate_power_cache_aware',
    'allocate_power_sumrate_max',
    'allocate_power_energy_efficient',
    
    # SIC
    'sinr_weak_user',
    'sinr_strong_decode_weak',
    'sinr_strong_after_sic',
    'sinr_weak_user_with_cache',
    'sinr_strong_after_perfect_sic',
    'compute_residual_interference',
    'simulate_sic_process',
]

__version__ = '1.0.0'
__author__ = 'Cache-Aided NOMA Team'