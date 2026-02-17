import numpy as np
from scipy.ndimage import gaussian_filter

def best_corner_local_gaussian(pred, sigma):
    """
    pred: 2D probability map (numpy array).
    sigma: Gaussian std in pixels for local neighborhood.
    Returns: (y, x) sub-pixel coordinate of best corner.
    """
    v_map = gaussian_filter(pred, sigma=sigma, mode='constant', cval=0.0)
    max_idx = np.unravel_index(np.argmax(v_map), v_map.shape)
    y, x = max_idx
    return (float(y), float(x))
