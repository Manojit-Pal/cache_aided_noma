# src/utils.py
import numpy as np

def set_seed(seed: int):
    """Set numpy random seed for reproducibility."""
    np.random.seed(seed)

def sample_zipf_catalog(num_files: int, alpha: float, size: int):
    """
    Sample `size` file requests from a Zipf distribution over a finite catalog.

    Args:
        num_files (int): total number of unique files in the catalog
        alpha (float): Zipf skew parameter (>0). Higher alpha = more skew.
        size (int): number of requests to generate

    Returns:
        np.ndarray: array of requested file indices [0 .. num_files-1]
    """
    # ranks 1..N
    ranks = np.arange(1, num_files + 1)
    # compute unnormalized probabilities ~ 1/r^alpha
    weights = 1.0 / np.power(ranks, alpha)
    # normalize
    probs = weights / weights.sum()
    # sample requests
    return np.random.choice(num_files, size=size, p=probs)
