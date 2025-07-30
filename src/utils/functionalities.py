from scipy.stats import wasserstein_distance
import numpy as np
import ot
from sklearn.preprocessing import MinMaxScaler

def sliced_wasserstein_distance(X, Y, num_directions=500, seed=None):
    """
    Compute approximate sliced Wasserstein distance between two datasets.
    
    Parameters:
        X, Y: numpy arrays of shape (n_samples, n_features)
        num_directions: number of random projections
        seed: optional random seed

    Returns:
        Approximate sliced Wasserstein distance
    """
    assert X.shape[1] == Y.shape[1], "X and Y must have same number of features"

    rng = np.random.default_rng(seed)
    d = X.shape[1]
    swd = 0.0

    for _ in range(num_directions):
        # Sample a random unit direction
        theta = rng.normal(size=(d,))
        theta /= np.linalg.norm(theta)

        # Project both datasets
        proj_X = X @ theta
        proj_Y = Y @ theta

        # Compute 1D Wasserstein distance
        swd += wasserstein_distance(proj_X, proj_Y)

    return swd / num_directions 
    
def minmax_scale(arr):
    min_vals = arr.min(axis=0)
    max_vals = arr.max(axis=0)
    return (arr - min_vals) / (max_vals - min_vals + 1e-8)  # avoid divide-by-zero

def ot_align(X1, X2, reg=0.1):
    
    # Uniform weights
    a, b = ot.unif(X1.shape[0]), ot.unif(X2.shape[0])
    
    # Cost matrix (Euclidean distance)
    M = ot.dist(X1, X2)
    
    # Compute transport plan using Sinkhorn
    G = ot.sinkhorn(a, b, M, reg=reg)
    sinkhorn_dist = ot.sinkhorn2(a, b, M, reg=reg)
    
    # Project Xt into Xs space via barycentric mapping
    X2_proj = G.T @ X1
    return X2_proj, sinkhorn_dist