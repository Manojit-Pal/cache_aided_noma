# src/noma/channel_model.py
import numpy as np

def generate_user_positions(num_users: int, cell_radius: float, seed: int = None):
    """
    Generate random user positions uniformly in a circle of radius cell_radius.
    Returns an array of shape (num_users, 2) with x,y coordinates and distances from origin.
    """
    if seed is not None:
        np.random.seed(seed)
    r = cell_radius * np.sqrt(np.random.rand(num_users))
    theta = 2 * np.pi * np.random.rand(num_users)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    d = np.sqrt(x**2 + y**2)
    return np.vstack((x, y, d)).T  # columns: x, y, d

def pathloss(distance: float, exponent: float, min_distance: float = 1.0):
    d = max(distance, min_distance)
    return d ** (-exponent)

def rayleigh_gain(num_samples: int):
    """
    Complex Gaussian h ~ CN(0,1), power gain |h|^2 ~ exponential(1)
    Return power gains (real positive).
    """
    # sample complex Gaussian (not necessary, directly sample exponential)
    return np.random.exponential(scale=1.0, size=num_samples)
