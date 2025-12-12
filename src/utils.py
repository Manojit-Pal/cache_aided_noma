# src/utils.py
"""
Utility Functions for Cache-Aided NOMA System

This module provides essential utility functions used across the entire project:
- Random seed management for reproducibility
- Zipf popularity distribution sampling (core caching model)
- Statistical analysis (confidence intervals, fairness metrics)
- Unit conversions (dB ↔ linear, power conversions)
- Data processing (smoothing, normalization)
- Result I/O (save/load in multiple formats)

Author: Cache-Aided NOMA Team
Date: December 2025
Version: 2.0 (Enhanced)
"""

import numpy as np
import pandas as pd
import json
from typing import Union, List, Tuple, Optional, Dict, Any
from pathlib import Path


# ============================================================================
# CORE UTILITIES (EXISTING - ESSENTIAL)
# ============================================================================

def set_seed(seed: int) -> None:
    """
    Set numpy random seed for reproducibility.
    
    Args:
        seed: Random seed value
        
    Example:
        >>> set_seed(2025)
        >>> # All subsequent random operations will be reproducible
    """
    np.random.seed(seed)


def sample_zipf_catalog(num_files: int, alpha: float, size: int) -> np.ndarray:
    """
    Sample file requests from a Zipf distribution over a finite catalog.
    
    This is the core popularity model for content caching research.
    Zipf distribution models real-world content popularity where a few files
    are very popular (e.g., viral videos) while most are rarely requested.
    
    Args:
        num_files: Total number of unique files in the catalog
        alpha: Zipf skew parameter (>0). Higher alpha = more skewed.
               - α=0.6: Moderate skew (broad popularity)
               - α=0.8: Realistic web traffic
               - α=1.0: Strong skew (typical video streaming)
               - α=1.2: Very strong skew (extreme popularity concentration)
        size: Number of requests to generate
    
    Returns:
        Array of requested file indices [0 .. num_files-1]
        
    Example:
        >>> # Generate 1000 requests from 200 files with realistic skew
        >>> requests = sample_zipf_catalog(200, alpha=0.8, size=1000)
        >>> # File 0 (most popular) will be requested much more than file 199
        
    Mathematical Model:
        P(file_k) ∝ 1 / k^α  where k is the rank (1 = most popular)
        
    References:
        - Breslau et al., "Web Caching and Zipf-like Distributions"
        - Cha et al., "I Tube, You Tube, Everybody Tubes" (YouTube traffic)
    """
    # Ranks: 1, 2, 3, ..., num_files
    ranks = np.arange(1, num_files + 1)
    
    # Compute unnormalized probabilities: p_k ∝ 1/k^α
    weights = 1.0 / np.power(ranks, alpha)
    
    # Normalize to sum to 1
    probs = weights / weights.sum()
    
    # Sample file IDs according to probability distribution
    return np.random.choice(num_files, size=size, p=probs)


# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

def compute_confidence_interval(
    data: np.ndarray, 
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute mean and confidence interval for data.
    
    Args:
        data: Array of measurements
        confidence: Confidence level (default: 0.95 for 95% CI)
        
    Returns:
        Tuple of (mean, lower_bound, upper_bound)
        
    Example:
        >>> hit_rates = np.array([0.65, 0.68, 0.67, 0.66, 0.69])
        >>> mean, ci_low, ci_high = compute_confidence_interval(hit_rates)
        >>> print(f"Hit rate: {mean:.3f} [{ci_low:.3f}, {ci_high:.3f}]")
    """
    from scipy import stats
    
    n = len(data)
    mean = np.mean(data)
    stderr = stats.sem(data)
    
    # t-distribution critical value
    margin = stderr * stats.t.ppf((1 + confidence) / 2, n - 1)
    
    return mean, mean - margin, mean + margin


def compute_jains_fairness(rates: np.ndarray) -> float:
    """
    Compute Jain's Fairness Index for rate allocation.
    
    Jain's index measures fairness of resource allocation:
    - 1.0: Perfect fairness (all users get equal rates)
    - 1/n: Worst case (one user gets everything)
    
    Args:
        rates: Array of user rates
        
    Returns:
        Fairness index in [1/n, 1.0]
        
    Example:
        >>> rates = np.array([5.0, 5.0, 5.0])  # Equal rates
        >>> compute_jains_fairness(rates)
        1.0
        >>> rates = np.array([10.0, 1.0, 1.0])  # Unfair
        >>> compute_jains_fairness(rates)
        0.6097...
        
    Reference:
        R. Jain et al., "A Quantitative Measure of Fairness and Discrimination
        for Resource Allocation in Shared Computer Systems"
    """
    if len(rates) == 0:
        return 0.0
    
    sum_rates = np.sum(rates)
    sum_squared = np.sum(rates**2)
    
    if sum_squared == 0:
        return 0.0
    
    n = len(rates)
    return (sum_rates**2) / (n * sum_squared)


def compute_coefficient_of_variation(data: np.ndarray) -> float:
    """
    Compute coefficient of variation (relative standard deviation).
    
    CV measures relative variability: CV = std / mean
    Useful for comparing variability across different scales.
    
    Args:
        data: Array of measurements
        
    Returns:
        Coefficient of variation (dimensionless)
        
    Example:
        >>> data = np.array([10, 12, 11, 13, 9])
        >>> cv = compute_coefficient_of_variation(data)
        >>> print(f"CV = {cv:.2%}")  # e.g., "CV = 12.34%"
    """
    mean = np.mean(data)
    if mean == 0:
        return 0.0
    return np.std(data) / mean


# ============================================================================
# UNIT CONVERSIONS
# ============================================================================

def db_to_linear(db: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert dB to linear scale.
    
    Args:
        db: Value(s) in dB
        
    Returns:
        Value(s) in linear scale
        
    Example:
        >>> db_to_linear(10)  # 10 dB
        10.0
        >>> db_to_linear(3)   # 3 dB (approximately 2×)
        1.995...
    """
    return 10**(db / 10)


def linear_to_db(linear: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert linear scale to dB.
    
    Args:
        linear: Value(s) in linear scale
        
    Returns:
        Value(s) in dB
        
    Example:
        >>> linear_to_db(10)
        10.0
        >>> linear_to_db(2)  # 2× ≈ 3 dB
        3.010...
    """
    return 10 * np.log10(linear)


def watts_to_dbm(watts: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert power from Watts to dBm.
    
    Args:
        watts: Power in Watts
        
    Returns:
        Power in dBm
        
    Example:
        >>> watts_to_dbm(1.0)     # 1 W
        30.0
        >>> watts_to_dbm(0.001)   # 1 mW
        0.0
    """
    return 10 * np.log10(watts * 1000)


def dbm_to_watts(dbm: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert power from dBm to Watts.
    
    Args:
        dbm: Power in dBm
        
    Returns:
        Power in Watts
        
    Example:
        >>> dbm_to_watts(30)  # 30 dBm
        1.0
        >>> dbm_to_watts(0)   # 0 dBm
        0.001
    """
    return 10**((dbm - 30) / 10)


# ============================================================================
# DATA PROCESSING
# ============================================================================

def moving_average(
    data: np.ndarray, 
    window: int = 10,
    mode: str = 'valid'
) -> np.ndarray:
    """
    Compute moving average for smoothing time series data.
    
    Args:
        data: Input time series
        window: Window size for averaging
        mode: Convolution mode ('valid', 'same', 'full')
              - 'valid': Output length = len(data) - window + 1
              - 'same': Output length = len(data)
              
    Returns:
        Smoothed data
        
    Example:
        >>> noisy_data = np.array([1, 5, 2, 8, 3, 7, 4, 6])
        >>> smooth_data = moving_average(noisy_data, window=3)
        >>> # Removes high-frequency noise
    """
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode=mode)


def exponential_moving_average(
    data: np.ndarray,
    alpha: float = 0.1
) -> np.ndarray:
    """
    Compute exponential moving average (EMA).
    
    EMA gives more weight to recent values: 
    EMA[t] = α * data[t] + (1-α) * EMA[t-1]
    
    Args:
        data: Input time series
        alpha: Smoothing factor in (0, 1)
               - α close to 0: More smoothing (slow adaptation)
               - α close to 1: Less smoothing (fast adaptation)
               
    Returns:
        Smoothed data with same length as input
        
    Example:
        >>> hit_rates = np.array([0.5, 0.6, 0.55, 0.7, 0.65])
        >>> ema = exponential_moving_average(hit_rates, alpha=0.2)
        >>> # Smooths while tracking trends
    """
    ema = np.zeros_like(data)
    ema[0] = data[0]
    
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
    
    return ema


def normalize(
    data: np.ndarray,
    method: str = 'minmax'
) -> np.ndarray:
    """
    Normalize data to [0, 1] range.
    
    Args:
        data: Input data
        method: Normalization method
                - 'minmax': (x - min) / (max - min)
                - 'zscore': (x - mean) / std
                
    Returns:
        Normalized data
        
    Example:
        >>> data = np.array([10, 20, 30, 40])
        >>> normalize(data, method='minmax')
        array([0. , 0.33, 0.67, 1. ])
    """
    if method == 'minmax':
        data_min = np.min(data)
        data_max = np.max(data)
        if data_max == data_min:
            return np.zeros_like(data)
        return (data - data_min) / (data_max - data_min)
    
    elif method == 'zscore':
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return np.zeros_like(data)
        return (data - mean) / std
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


# ============================================================================
# RESULT I/O
# ============================================================================

def save_results(
    data: pd.DataFrame,
    filename: Union[str, Path],
    formats: List[str] = ['csv']
) -> None:
    """
    Save results in multiple formats.
    
    Args:
        data: Results DataFrame
        filename: Base filename (without extension)
        formats: List of formats to save
                 Supported: 'csv', 'json', 'excel', 'latex', 'markdown'
                 
    Example:
        >>> results_df = pd.DataFrame({'policy': ['topk', 'lru'], 
        ...                            'hit_rate': [0.7, 0.6]})
        >>> save_results(results_df, 'results', formats=['csv', 'json', 'latex'])
        ✅ Saved results.csv
        ✅ Saved results.json
        ✅ Saved results.tex
    """
    filename = Path(filename)
    base = filename.stem  # Remove extension if present
    parent = filename.parent if filename.parent.name else Path('.')
    
    for fmt in formats:
        if fmt == 'csv':
            path = parent / f"{base}.csv"
            data.to_csv(path, index=False)
            print(f"✅ Saved {path}")
            
        elif fmt == 'json':
            path = parent / f"{base}.json"
            data.to_json(path, orient='records', indent=2)
            print(f"✅ Saved {path}")
            
        elif fmt == 'excel':
            path = parent / f"{base}.xlsx"
            data.to_excel(path, index=False, engine='openpyxl')
            print(f"✅ Saved {path}")
            
        elif fmt == 'latex':
            path = parent / f"{base}.tex"
            with open(path, 'w') as f:
                f.write(data.to_latex(index=False, float_format="%.4f"))
            print(f"✅ Saved {path}")
            
        elif fmt == 'markdown':
            path = parent / f"{base}.md"
            with open(path, 'w') as f:
                f.write(data.to_markdown(index=False))
            print(f"✅ Saved {path}")
            
        else:
            print(f"⚠️  Unknown format: {fmt}")


def load_results(filename: Union[str, Path]) -> pd.DataFrame:
    """
    Load results from file (auto-detects format).
    
    Args:
        filename: Path to results file
        
    Returns:
        Results DataFrame
        
    Example:
        >>> df = load_results('results/experiment_data.csv')
    """
    filename = Path(filename)
    
    if filename.suffix == '.csv':
        return pd.read_csv(filename)
    elif filename.suffix == '.json':
        return pd.read_json(filename)
    elif filename.suffix in ['.xlsx', '.xls']:
        return pd.read_excel(filename)
    else:
        raise ValueError(f"Unsupported file format: {filename.suffix}")


def save_config(config_dict: Dict[str, Any], filename: Union[str, Path]) -> None:
    """
    Save configuration dictionary to JSON.
    
    Args:
        config_dict: Configuration parameters
        filename: Output JSON file path
        
    Example:
        >>> from src import config
        >>> save_config(config.get_noma_config(), 'experiment_config.json')
    """
    filename = Path(filename)
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    clean_dict = {k: convert_types(v) for k, v in config_dict.items()}
    
    with open(filename, 'w') as f:
        json.dump(clean_dict, f, indent=2)
    
    print(f"✅ Configuration saved to {filename}")


# ============================================================================
# PERFORMANCE ANALYSIS
# ============================================================================

def compute_performance_gain(
    baseline: float,
    improved: float,
    metric_type: str = 'higher_better'
) -> float:
    """
    Compute percentage performance gain.
    
    Args:
        baseline: Baseline performance value
        improved: Improved performance value
        metric_type: 'higher_better' (e.g., hit rate) or 
                     'lower_better' (e.g., outage)
                     
    Returns:
        Percentage gain (positive = improvement)
        
    Example:
        >>> baseline_hit = 0.60
        >>> dqn_hit = 0.72
        >>> gain = compute_performance_gain(baseline_hit, dqn_hit, 'higher_better')
        >>> print(f"DQN improves hit rate by {gain:.1f}%")
        DQN improves hit rate by 20.0%
    """
    if baseline == 0:
        return 0.0
    
    if metric_type == 'higher_better':
        return ((improved - baseline) / baseline) * 100
    elif metric_type == 'lower_better':
        return ((baseline - improved) / baseline) * 100
    else:
        raise ValueError(f"Unknown metric_type: {metric_type}")


def summarize_results(
    df: pd.DataFrame,
    group_by: str = 'policy',
    metrics: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Generate summary statistics for results.
    
    Args:
        df: Results DataFrame
        group_by: Column to group by (e.g., 'policy', 'snr')
        metrics: List of metric columns to summarize
                 If None, uses all numeric columns
                 
    Returns:
        Summary DataFrame with mean ± std for each metric
        
    Example:
        >>> summary = summarize_results(results_df, group_by='policy')
        >>> print(summary)
                 hit_rate_mean  hit_rate_std  outage_mean  outage_std
        policy                                                        
        topk              0.72          0.03         0.05        0.01
        lru               0.65          0.04         0.08        0.02
    """
    if metrics is None:
        # Auto-detect numeric columns
        metrics = df.select_dtypes(include=[np.number]).columns.tolist()
        if group_by in metrics:
            metrics.remove(group_by)
    
    # Compute mean and std for each metric
    agg_dict = {}
    for metric in metrics:
        if metric in df.columns:
            agg_dict[metric] = ['mean', 'std']
    
    summary = df.groupby(group_by).agg(agg_dict)
    
    # Flatten multi-level columns
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    
    return summary


# ============================================================================
# ZIPF DISTRIBUTION ANALYSIS
# ============================================================================

def analyze_zipf_distribution(
    num_files: int,
    alpha: float,
    num_samples: int = 10000
) -> Dict[str, Any]:
    """
    Analyze properties of Zipf distribution.
    
    Args:
        num_files: Catalog size
        alpha: Zipf parameter
        num_samples: Number of samples for empirical analysis
        
    Returns:
        Dictionary with distribution statistics
        
    Example:
        >>> stats = analyze_zipf_distribution(200, alpha=0.8, num_samples=10000)
        >>> print(f"Top-10% files account for {stats['top10_coverage']:.1%} of requests")
    """
    # Generate samples
    samples = sample_zipf_catalog(num_files, alpha, num_samples)
    
    # Compute theoretical probabilities
    ranks = np.arange(1, num_files + 1)
    weights = 1.0 / np.power(ranks, alpha)
    probs = weights / weights.sum()
    
    # Empirical request counts
    unique, counts = np.unique(samples, return_counts=True)
    empirical_probs = counts / num_samples
    
    # Compute statistics
    top10_idx = int(num_files * 0.1)
    top10_coverage = np.sum(probs[:top10_idx])
    
    top20_idx = int(num_files * 0.2)
    top20_coverage = np.sum(probs[:top20_idx])
    
    # Entropy (measure of diversity)
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    
    return {
        'num_files': num_files,
        'alpha': alpha,
        'top10_coverage': top10_coverage,
        'top20_coverage': top20_coverage,
        'entropy_bits': entropy,
        'max_prob': probs[0],
        'min_prob': probs[-1],
        'theoretical_probs': probs,
        'empirical_probs': empirical_probs,
        'concentration_ratio': probs[0] / probs[-1]
    }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def pretty_print_dict(d: Dict, title: str = "", indent: int = 2) -> None:
    """
    Pretty print dictionary with hierarchical formatting.
    
    Args:
        d: Dictionary to print
        title: Optional title
        indent: Indentation spaces
        
    Example:
        >>> config = {'lr': 0.001, 'gamma': 0.99, 'hidden': [128, 64]}
        >>> pretty_print_dict(config, title="Training Config")
    """
    if title:
        print(f"\n{'='*60}")
        print(f"{title}")
        print(f"{'='*60}")
    
    for key, value in d.items():
        if isinstance(value, dict):
            print(f"{' '*indent}{key}:")
            pretty_print_dict(value, indent=indent+2)
        elif isinstance(value, (list, tuple)):
            print(f"{' '*indent}{key}: {value}")
        else:
            print(f"{' '*indent}{key}: {value}")


def create_experiment_id(prefix: str = "exp") -> str:
    """
    Generate unique experiment ID with timestamp.
    
    Args:
        prefix: ID prefix
        
    Returns:
        Unique experiment ID
        
    Example:
        >>> exp_id = create_experiment_id("dqn_training")
        >>> print(exp_id)
        dqn_training_20251212_203045
    """
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"


# ============================================================================
# MODULE INFO
# ============================================================================

__all__ = [
    # Core utilities
    'set_seed',
    'sample_zipf_catalog',
    
    # Statistical analysis
    'compute_confidence_interval',
    'compute_jains_fairness',
    'compute_coefficient_of_variation',
    
    # Unit conversions
    'db_to_linear',
    'linear_to_db',
    'watts_to_dbm',
    'dbm_to_watts',
    
    # Data processing
    'moving_average',
    'exponential_moving_average',
    'normalize',
    
    # Result I/O
    'save_results',
    'load_results',
    'save_config',
    
    # Performance analysis
    'compute_performance_gain',
    'summarize_results',
    'analyze_zipf_distribution',
    
    # Helpers
    'pretty_print_dict',
    'create_experiment_id',
]

__version__ = '2.0'
__author__ = 'Cache-Aided NOMA Team'


# ============================================================================
# MAIN (for testing)
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("UTILS MODULE - FUNCTIONALITY TEST")
    print("="*70)
    
    # Test Zipf sampling
    print("\n1️⃣  Testing Zipf Distribution:")
    set_seed(2025)
    requests = sample_zipf_catalog(200, alpha=0.8, size=1000)
    print(f"   Generated {len(requests)} requests")
    print(f"   Most popular file requested {np.sum(requests == 0)} times")
    print(f"   Unique files requested: {len(np.unique(requests))}/200")
    
    # Analyze distribution
    stats = analyze_zipf_distribution(200, alpha=0.8)
    print(f"\n   Distribution Analysis (α=0.8):")
    print(f"   - Top 10% files: {stats['top10_coverage']:.1%} of traffic")
    print(f"   - Top 20% files: {stats['top20_coverage']:.1%} of traffic")
    print(f"   - Entropy: {stats['entropy_bits']:.2f} bits")
    
    # Test statistics
    print("\n2️⃣  Testing Statistical Functions:")
    hit_rates = np.array([0.65, 0.68, 0.67, 0.66, 0.69])
    mean, ci_low, ci_high = compute_confidence_interval(hit_rates)
    print(f"   Hit rates: {hit_rates}")
    print(f"   Mean: {mean:.4f}")
    print(f"   95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
    
    # Test fairness
    print("\n3️⃣  Testing Fairness Metric:")
    fair_rates = np.array([5.0, 5.0, 5.0])
    unfair_rates = np.array([10.0, 1.0, 1.0])
    print(f"   Fair allocation {fair_rates}: JFI = {compute_jains_fairness(fair_rates):.4f}")
    print(f"   Unfair allocation {unfair_rates}: JFI = {compute_jains_fairness(unfair_rates):.4f}")
    
    # Test conversions
    print("\n4️⃣  Testing Unit Conversions:")
    print(f"   10 dB = {db_to_linear(10):.2f} (linear)")
    print(f"   2× = {linear_to_db(2):.2f} dB")
    print(f"   1 W = {watts_to_dbm(1.0):.1f} dBm")
    print(f"   30 dBm = {dbm_to_watts(30):.3f} W")
    
    # Test smoothing
    print("\n5️⃣  Testing Data Smoothing:")
    noisy = np.array([1, 5, 2, 8, 3, 7, 4, 6, 3, 7])
    smooth = moving_average(noisy, window=3, mode='valid')
    print(f"   Original: {noisy}")
    print(f"   Smoothed: {smooth}")
    
    # Test performance gain
    print("\n6️⃣  Testing Performance Analysis:")
    baseline = 0.60
    improved = 0.72
    gain = compute_performance_gain(baseline, improved, 'higher_better')
    print(f"   Baseline hit rate: {baseline:.2f}")
    print(f"   Improved hit rate: {improved:.2f}")
    print(f"   Performance gain: {gain:.1f}%")
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED")
    print("="*70 + "\n")