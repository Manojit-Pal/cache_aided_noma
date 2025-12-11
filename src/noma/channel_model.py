# src/noma/channel_model.py
"""
Wireless Channel Model for Cache-Aided NOMA in 6G Networks

This module implements various wireless channel models including:
- User positioning (uniform spatial distribution)
- Path loss models (distance-dependent attenuation)
- Small-scale fading (Rayleigh and Rician for 6G LoS scenarios)
- Channel State Information (CSI) computation
- Spatial correlation for realistic multi-user scenarios
- Time-varying channel support for mobility

Author: Cache-Aided NOMA Team
Date: December 2025
"""

import numpy as np
from typing import Tuple, Optional


# ============================================================================
# USER POSITIONING
# ============================================================================

def generate_user_positions(num_users: int, cell_radius: float, seed: int = None) -> np.ndarray:
    """
    Generate random user positions uniformly distributed in a circular cell.
    
    Uses the correct uniform distribution method (radius² transformation) to ensure
    users are uniformly distributed across the cell area, not concentrated at center.
    
    Args:
        num_users: Number of users to generate positions for
        cell_radius: Cell radius in meters (e.g., 500m for typical macro cell)
        seed: Random seed for reproducibility (optional)
    
    Returns:
        np.ndarray: Array of shape (num_users, 3) with columns [x, y, distance]
                    - x, y: Cartesian coordinates in meters
                    - distance: Euclidean distance from base station (origin)
    
    Example:
        >>> positions = generate_user_positions(100, 500.0, seed=42)
        >>> print(positions.shape)  # (100, 3)
        >>> print(positions[0])     # [x, y, d] for first user
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Uniform distribution in circle: r ~ sqrt(U[0,1]), θ ~ U[0, 2π]
    r = cell_radius * np.sqrt(np.random.rand(num_users))
    theta = 2 * np.pi * np.random.rand(num_users)
    
    # Convert to Cartesian coordinates
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    d = np.sqrt(x**2 + y**2)  # Distance from origin (base station)
    
    return np.vstack((x, y, d)).T  # Shape: (num_users, 3)


# ============================================================================
# PATH LOSS MODELS
# ============================================================================

def pathloss(distance: float, exponent: float, min_distance: float = 1.0) -> float:
    """
    Compute path loss (large-scale fading) based on distance.
    
    Models signal attenuation due to propagation distance using the
    standard power-law path loss model:
        PL(d) = d^(-α)
    
    where α is the path loss exponent (typically 2-4).
    
    Args:
        distance: Distance between transmitter and receiver in meters
        exponent: Path loss exponent α
                  - Free space: α = 2
                  - Urban: α = 3-4
                  - Indoor: α = 3-5
                  - Your config uses α = 3.5 (typical urban macro cell)
        min_distance: Minimum distance to avoid singularity (default 1.0m)
    
    Returns:
        float: Path loss gain (unitless, < 1)
    
    Example:
        >>> pl = pathloss(100, 3.5)  # User at 100m
        >>> print(f"Path loss: {pl:.6f}")  # ~3.16e-8
    """
    d = max(distance, min_distance)  # Avoid division by zero
    return d ** (-exponent)


def pathloss_db(distance: float, exponent: float, reference_distance: float = 1.0) -> float:
    """
    Compute path loss in dB (alternative formulation).
    
    PL(d) [dB] = PL(d0) + 10·α·log10(d/d0)
    
    Args:
        distance: Distance in meters
        exponent: Path loss exponent
        reference_distance: Reference distance d0 (typically 1m)
    
    Returns:
        float: Path loss in dB (negative value)
    """
    d = max(distance, reference_distance)
    return -10 * exponent * np.log10(d / reference_distance)


# ============================================================================
# SMALL-SCALE FADING (RAYLEIGH & RICIAN FOR 6G)
# ============================================================================

def rayleigh_gain(num_samples: int) -> np.ndarray:
    """
    Generate Rayleigh fading channel power gains.
    
    Rayleigh fading models Non-Line-of-Sight (NLoS) propagation with rich scattering.
    The channel coefficient h follows:
        h ~ CN(0, 1)  (complex Gaussian)
        |h|² ~ Exponential(1)  (power gain)
    
    Used when there's no direct path between transmitter and receiver (blocked by
    buildings, obstacles, etc.).
    
    Args:
        num_samples: Number of independent channel realizations
    
    Returns:
        np.ndarray: Power gains |h|² with shape (num_samples,)
                    Mean = 1, follows exponential distribution
    
    Example:
        >>> gains = rayleigh_gain(1000)
        >>> print(f"Mean: {gains.mean():.2f}")  # Should be ~1.0
    """
    # Directly sample from exponential distribution (more efficient)
    return np.random.exponential(scale=1.0, size=num_samples)


def rician_gain(num_samples: int, K_factor_db: float = 10.0) -> np.ndarray:
    """
    Generate Rician fading channel power gains (for 6G Line-of-Sight scenarios).
    
    Rician fading models channels with a dominant Line-of-Sight (LoS) component
    plus scattered multipath. This is critical for 6G systems operating at mmWave
    and higher frequencies where LoS is common.
    
    The Rician K-factor determines the ratio of LoS to scattered power:
        K = (LoS power) / (scattered power)
    
    Args:
        num_samples: Number of independent channel realizations
        K_factor_db: Rician K-factor in dB (default 10 dB = strong LoS)
                     - K = -∞ dB (0 linear): Pure Rayleigh (no LoS)
                     - K = 0 dB (1 linear): Equal LoS and scattered power
                     - K = 10 dB (10 linear): Strong LoS (typical 6G mmWave)
                     - K = 20 dB (100 linear): Very strong LoS (6G indoor)
    
    Returns:
        np.ndarray: Power gains |h|² with shape (num_samples,)
    
    Physical Interpretation:
        - Higher K → More stable channel (predictable LoS component)
        - Lower K → More random channel (scattering dominates)
    
    Example:
        >>> gains_los = rician_gain(1000, K_factor_db=10)  # Strong LoS
        >>> gains_nlos = rayleigh_gain(1000)  # No LoS
        >>> print(f"LoS variance: {gains_los.var():.2f}")  # Lower variance
        >>> print(f"NLoS variance: {gains_nlos.var():.2f}")  # Higher variance
    """
    # Convert K-factor from dB to linear scale
    K = 10 ** (K_factor_db / 10.0)
    
    # Rician fading: h = sqrt(K/(K+1)) + sqrt(1/(K+1)) * h_rayleigh
    # Where first term is LoS, second term is scattered component
    
    # LoS component (deterministic)
    los_amplitude = np.sqrt(K / (K + 1))
    
    # Scattered component (random)
    scatter_amplitude = np.sqrt(1 / (K + 1))
    
    # Generate complex Gaussian for scattered component
    h_real = np.random.randn(num_samples)
    h_imag = np.random.randn(num_samples)
    h_scattered = (h_real + 1j * h_imag) / np.sqrt(2)  # CN(0,1)
    
    # Combine LoS and scattered (LoS assumed on real axis)
    h_total = los_amplitude + scatter_amplitude * h_scattered
    
    # Return power gain |h|²
    return np.abs(h_total) ** 2


def mixed_fading_gain(num_samples: int, los_probability: float = 0.5, 
                      K_factor_db: float = 10.0) -> np.ndarray:
    """
    Generate mixed fading with probabilistic LoS/NLoS (realistic 6G scenario).
    
    In real 6G deployments, some users have LoS while others don't, depending on
    environment, mobility, and blockage. This function models this heterogeneity.
    
    Args:
        num_samples: Number of channel realizations
        los_probability: Probability of LoS condition (0 to 1)
                        - Urban outdoor: 0.3-0.5
                        - Suburban: 0.6-0.8
                        - Indoor: 0.7-0.9
        K_factor_db: K-factor for LoS channels (dB)
    
    Returns:
        np.ndarray: Power gains, mixture of Rician (LoS) and Rayleigh (NLoS)
    
    Example:
        >>> gains = mixed_fading_gain(1000, los_probability=0.4, K_factor_db=10)
        >>> # ~40% will have Rician fading, ~60% Rayleigh fading
    """
    # Determine which samples have LoS
    los_mask = np.random.rand(num_samples) < los_probability
    num_los = np.sum(los_mask)
    num_nlos = num_samples - num_los
    
    gains = np.zeros(num_samples)
    
    # Generate Rician gains for LoS users
    if num_los > 0:
        gains[los_mask] = rician_gain(num_los, K_factor_db)
    
    # Generate Rayleigh gains for NLoS users
    if num_nlos > 0:
        gains[~los_mask] = rayleigh_gain(num_nlos)
    
    return gains


# ============================================================================
# CHANNEL STATE INFORMATION (CSI) - COMPLETE CHANNEL GAIN
# ============================================================================

def compute_channel_gains(positions: np.ndarray, exponent: float, 
                         min_distance: float = 1.0,
                         fading_type: str = 'rayleigh',
                         K_factor_db: float = 10.0,
                         los_probability: float = 0.5) -> np.ndarray:
    """
    Compute complete channel gains (CSI) = Path Loss × Small-Scale Fading.
    
    This is the key function that combines large-scale (path loss) and small-scale
    (fading) effects to get the complete channel state information.
    
    Channel model:
        g_i = |h_i|² × PL(d_i)
    
    where:
        g_i = total channel power gain for user i
        |h_i|² = small-scale fading power gain
        PL(d_i) = path loss based on distance d_i
    
    Args:
        positions: User positions array of shape (num_users, 3) from generate_user_positions()
        exponent: Path loss exponent (typically 3.5 for urban)
        min_distance: Minimum distance for path loss calculation (meters)
        fading_type: Type of small-scale fading
                    - 'rayleigh': NLoS channels (default)
                    - 'rician': LoS channels (6G mmWave)
                    - 'mixed': Probabilistic LoS/NLoS (realistic 6G)
        K_factor_db: Rician K-factor in dB (only for 'rician' or 'mixed')
        los_probability: LoS probability (only for 'mixed' fading)
    
    Returns:
        np.ndarray: Complete channel gains of shape (num_users,)
    
    Example:
        >>> positions = generate_user_positions(100, 500, seed=42)
        >>> gains = compute_channel_gains(positions, 3.5, fading_type='mixed')
        >>> print(f"Average channel gain: {gains.mean():.2e}")
    """
    num_users = positions.shape[0]
    distances = positions[:, 2]  # Third column is distance
    
    # Compute path loss for all users
    path_losses = np.array([pathloss(d, exponent, min_distance) for d in distances])
    
    # Generate small-scale fading based on type
    if fading_type == 'rayleigh':
        fading_gains = rayleigh_gain(num_users)
    elif fading_type == 'rician':
        fading_gains = rician_gain(num_users, K_factor_db)
    elif fading_type == 'mixed':
        fading_gains = mixed_fading_gain(num_users, los_probability, K_factor_db)
    else:
        raise ValueError(f"Unknown fading type: {fading_type}. Use 'rayleigh', 'rician', or 'mixed'.")
    
    # Complete channel gain = path loss × fading
    channel_gains = path_losses * fading_gains
    
    return channel_gains


# ============================================================================
# SPATIAL CORRELATION (ADVANCED FEATURE)
# ============================================================================

def generate_correlated_fading(num_users: int, correlation_distance: float,
                              positions: np.ndarray) -> np.ndarray:
    """
    Generate spatially correlated Rayleigh fading (advanced feature).
    
    In reality, users close to each other experience correlated fading due to
    similar scattering environments. This is important for accurate multi-user
    NOMA simulations.
    
    Uses exponential correlation model:
        ρ(d) = exp(-d / d_corr)
    
    where d is the distance between users and d_corr is the correlation distance.
    
    Args:
        num_users: Number of users
        correlation_distance: Decorrelation distance in meters (typically 10-50m)
                             - Smaller → faster decorrelation
                             - Larger → channels stay correlated over larger areas
        positions: User positions array of shape (num_users, 3)
    
    Returns:
        np.ndarray: Correlated fading gains of shape (num_users,)
    
    Note: This is computationally expensive for large num_users due to
          covariance matrix computation and Cholesky decomposition.
    """
    # Compute pairwise distances between all users
    xy_positions = positions[:, :2]  # Only x, y coordinates
    distance_matrix = np.zeros((num_users, num_users))
    
    for i in range(num_users):
        for j in range(i+1, num_users):
            d = np.linalg.norm(xy_positions[i] - xy_positions[j])
            distance_matrix[i, j] = d
            distance_matrix[j, i] = d
    
    # Compute correlation matrix using exponential model
    correlation_matrix = np.exp(-distance_matrix / correlation_distance)
    
    # Generate correlated complex Gaussian samples
    # Use Cholesky decomposition: Σ = L L^T
    try:
        L = np.linalg.cholesky(correlation_matrix)
    except np.linalg.LinAlgError:
        # If matrix is not positive definite, add small regularization
        correlation_matrix += 1e-6 * np.eye(num_users)
        L = np.linalg.cholesky(correlation_matrix)
    
    # Generate uncorrelated samples
    uncorrelated_real = np.random.randn(num_users)
    uncorrelated_imag = np.random.randn(num_users)
    
    # Apply correlation
    correlated_real = L @ uncorrelated_real
    correlated_imag = L @ uncorrelated_imag
    
    # Compute power gains
    h_correlated = (correlated_real + 1j * correlated_imag) / np.sqrt(2)
    power_gains = np.abs(h_correlated) ** 2
    
    return power_gains


# ============================================================================
# TIME-VARYING CHANNELS (MOBILITY SUPPORT)
# ============================================================================

class TimeVaryingChannel:
    """
    Time-varying channel model for mobile users.
    
    Implements Jake's model for Doppler spectrum due to user mobility.
    Important for simulating realistic 6G scenarios with mobile users.
    
    Attributes:
        carrier_freq: Carrier frequency in Hz (e.g., 28e9 for 28 GHz mmWave)
        velocity: User velocity in m/s
        sampling_interval: Time between channel samples in seconds
        doppler_freq: Maximum Doppler frequency in Hz
    """
    
    def __init__(self, carrier_freq: float = 28e9, velocity: float = 3.0, 
                 sampling_interval: float = 0.001):
        """
        Initialize time-varying channel.
        
        Args:
            carrier_freq: Carrier frequency in Hz (default 28 GHz for 6G mmWave)
            velocity: User velocity in m/s (default 3 m/s ≈ walking speed)
            sampling_interval: Time between samples in seconds (default 1 ms)
        """
        self.carrier_freq = carrier_freq
        self.velocity = velocity
        self.sampling_interval = sampling_interval
        
        # Speed of light
        c = 3e8  # m/s
        
        # Maximum Doppler frequency: f_d = v * f_c / c
        self.doppler_freq = velocity * carrier_freq / c
        
        print(f"Time-varying channel initialized:")
        print(f"  Carrier: {carrier_freq/1e9:.1f} GHz")
        print(f"  Velocity: {velocity} m/s")
        print(f"  Doppler: {self.doppler_freq:.2f} Hz")
    
    def generate_time_series(self, num_samples: int, fading_type: str = 'rayleigh',
                            K_factor_db: float = 10.0) -> np.ndarray:
        """
        Generate time-correlated fading samples using Jake's model.
        
        Args:
            num_samples: Number of time samples
            fading_type: 'rayleigh' or 'rician'
            K_factor_db: Rician K-factor (only for 'rician')
        
        Returns:
            np.ndarray: Time-correlated fading gains of shape (num_samples,)
        """
        # Simplified Jake's model using low-pass filtered white noise
        # For exact Jake's model, use sum of sinusoids (computationally expensive)
        
        # Normalized Doppler bandwidth
        doppler_normalized = self.doppler_freq * self.sampling_interval
        
        # Correlation coefficient between adjacent samples
        # ρ(τ) ≈ J_0(2π f_d τ) where J_0 is Bessel function of first kind
        from scipy.special import j0
        rho = j0(2 * np.pi * doppler_normalized)
        
        # Generate correlated samples using AR(1) process
        if fading_type == 'rayleigh':
            # Generate complex Gaussian samples
            h_real = np.zeros(num_samples)
            h_imag = np.zeros(num_samples)
            
            # Initialize
            h_real[0] = np.random.randn()
            h_imag[0] = np.random.randn()
            
            # AR(1) process: h[n] = rho * h[n-1] + sqrt(1-rho²) * w[n]
            noise_scale = np.sqrt(1 - rho**2)
            for i in range(1, num_samples):
                h_real[i] = rho * h_real[i-1] + noise_scale * np.random.randn()
                h_imag[i] = rho * h_imag[i-1] + noise_scale * np.random.randn()
            
            h = (h_real + 1j * h_imag) / np.sqrt(2)
            power_gains = np.abs(h) ** 2
        
        elif fading_type == 'rician':
            # For Rician, add LoS component (constant over time)
            K = 10 ** (K_factor_db / 10.0)
            los_amplitude = np.sqrt(K / (K + 1))
            scatter_amplitude = np.sqrt(1 / (K + 1))
            
            # Generate scattered component (time-varying)
            h_real = np.zeros(num_samples)
            h_imag = np.zeros(num_samples)
            h_real[0] = np.random.randn()
            h_imag[0] = np.random.randn()
            
            noise_scale = np.sqrt(1 - rho**2)
            for i in range(1, num_samples):
                h_real[i] = rho * h_real[i-1] + noise_scale * np.random.randn()
                h_imag[i] = rho * h_imag[i-1] + noise_scale * np.random.randn()
            
            h_scattered = (h_real + 1j * h_imag) / np.sqrt(2)
            h_total = los_amplitude + scatter_amplitude * h_scattered
            power_gains = np.abs(h_total) ** 2
        
        else:
            raise ValueError(f"Unknown fading type: {fading_type}")
        
        return power_gains


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def channel_gain_to_snr(channel_gain: float, tx_power: float, noise_power: float) -> float:
    """
    Convert channel gain to SNR (Signal-to-Noise Ratio).
    
    SNR = (P_tx × g) / N_0
    
    Args:
        channel_gain: Channel power gain g
        tx_power: Transmit power in Watts (linear scale)
        noise_power: Noise power in Watts (linear scale)
    
    Returns:
        float: SNR (linear scale, not dB)
    """
    return (tx_power * channel_gain) / noise_power


def snr_to_db(snr_linear: float) -> float:
    """Convert SNR from linear to dB scale."""
    return 10 * np.log10(snr_linear)


def db_to_linear(db_value: float) -> float:
    """Convert dB to linear scale."""
    return 10 ** (db_value / 10.0)


# ============================================================================
# EXAMPLE USAGE AND TESTING
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("TESTING ENHANCED CHANNEL MODEL FOR 6G CACHE-AIDED NOMA")
    print("="*70)
    
    # Test 1: User positioning
    print("\n[Test 1] Generating user positions...")
    positions = generate_user_positions(num_users=10, cell_radius=500.0, seed=42)
    print(f"Generated {len(positions)} user positions")
    print(f"Sample position: x={positions[0,0]:.2f}m, y={positions[0,1]:.2f}m, d={positions[0,2]:.2f}m")
    
    # Test 2: Channel gains with different fading types
    print("\n[Test 2] Computing channel gains...")
    gains_rayleigh = compute_channel_gains(positions, 3.5, fading_type='rayleigh')
    gains_rician = compute_channel_gains(positions, 3.5, fading_type='rician', K_factor_db=10)
    gains_mixed = compute_channel_gains(positions, 3.5, fading_type='mixed', los_probability=0.5)
    
    print(f"Rayleigh fading: mean gain = {gains_rayleigh.mean():.2e}")
    print(f"Rician fading: mean gain = {gains_rician.mean():.2e}")
    print(f"Mixed fading: mean gain = {gains_mixed.mean():.2e}")
    
    # Test 3: Time-varying channel
    print("\n[Test 3] Time-varying channel...")
    channel = TimeVaryingChannel(carrier_freq=28e9, velocity=3.0, sampling_interval=0.001)
    time_series = channel.generate_time_series(num_samples=100, fading_type='rayleigh')
    print(f"Generated {len(time_series)} time-correlated samples")
    print(f"Mean gain: {time_series.mean():.2f}, Variance: {time_series.var():.2f}")
    
    print("\n" + "="*70)
    print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*70)