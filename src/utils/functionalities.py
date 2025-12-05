from scipy.stats import wasserstein_distance
import numpy as np
import ot
from scipy.spatial.distance import cdist
import tqdm as tqdm
import random
import torch
from sklearn.cluster import MiniBatchKMeans


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

def compute_kmeans_distribution(data, n_clusters=100, batch_size=10000, random_state=42):
    """
    Performs K-Means clustering using MiniBatchKMeans for large datasets,
    computes the distribution of points over clusters, and returns centroids.

    Args:
        data (np.ndarray): Input array of shape (n, 6).
        n_clusters (int): Number of clusters to form.
        batch_size (int): Mini-batch size for faster training.
        random_state (int): Random seed for reproducibility.

    Returns:
        a (np.ndarray): Vector of shape (n_clusters,) with proportion of data in each cluster.
        centroids (np.ndarray): Array of shape (n_clusters, 6) with cluster centroids.
        labels (np.ndarray): Cluster assignments for each sample.
    """
    # Initialize MiniBatchKMeans (faster for millions of samples)
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=batch_size,
        random_state=random_state,
        n_init="auto"
    )

    # Fit and predict
    labels = kmeans.fit_predict(data)

    # Compute distribution (counts and proportions)
    counts = np.bincount(labels, minlength=n_clusters)
    a = counts / counts.sum()  # normalize to sum to 1

    # Extract centroids
    centroids = kmeans.cluster_centers_

    return a, centroids, labels, kmeans

def rbf_kernel(X, Y, sigma=1.0):
    """Compute the RBF (Gaussian) kernel between two sets of vectors."""
    dists = cdist(X, Y, 'sqeuclidean')
    return np.exp(-dists / (2 * sigma ** 2))

def compute_mmd(X, Y, sigma=1.0):
    """Biased estimator of MMD², guaranteed to be ≥ 0."""
    K_XX = rbf_kernel(X, X, sigma)
    K_YY = rbf_kernel(Y, Y, sigma)
    K_XY = rbf_kernel(X, Y, sigma)

    mmd2 = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
    return mmd2

def mmd_test(X, Y, sigma=1.0, num_permutations=1000, seed=None):
    """Two-sample test using MMD with permutation test to get p-value."""
    rng = np.random.default_rng(seed)
    n, m = len(X), len(Y)
    Z = np.vstack([X, Y])
    observed_mmd = compute_mmd(X, Y, sigma=sigma)

    permuted_mmds = []
    for _ in (range(num_permutations)):
        idx = rng.permutation(n + m)
        X_perm = Z[idx[:n]]
        Y_perm = Z[idx[n:]]
        permuted_mmds.append(compute_mmd(X_perm, Y_perm, sigma=sigma))

    p_value = np.mean([mmd >= observed_mmd for mmd in permuted_mmds])
    return observed_mmd, permuted_mmds, p_value

def median_pairwise_distance(X, Y):
    Z = np.vstack([X, Y])
    dists = cdist(Z, Z, 'euclidean')
    return np.median(dists[np.triu_indices_from(dists, k=1)])

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False