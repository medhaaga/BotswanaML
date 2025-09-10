import numpy as np

def signed_log(x: np.ndarray) -> np.ndarray:
    """Signed log transform for zero-centered features."""
    return np.sign(x) * np.log1p(np.abs(x))

def compute_combined_quantiles(datasets,
                               pos_idx, center_idx,
                               n_sample_per_target=200_000,
                               low_q=0.05, high_q=0.95,
                               random_state=0):
    """
    Compute robust quantiles across source + all target domains.
    """
    rng = np.random.default_rng(random_state)

    def subsample(X, n):
        if X.shape[0] <= n:
            return X
        idx = rng.choice(X.shape[0], size=n, replace=False)
        return X[idx]

    sampled_datasets = [subsample(X, n_sample_per_target) for X in datasets]
    X_comb = np.vstack(sampled_datasets)

    # Transform before quantile calculation
    Xq = X_comb.copy()
    if pos_idx:
        Xq[:, pos_idx] = np.log1p(Xq[:, pos_idx])
    if center_idx:
        Xq[:, center_idx] = signed_log(Xq[:, center_idx])

    lows = np.quantile(Xq, low_q, axis=0)
    highs = np.quantile(Xq, high_q, axis=0)
    highs = np.where(highs == lows, lows + 1e-6, highs)

    return lows, highs

def transform_and_scale(X, pos_idx, center_idx, lows, highs,
                        clip_to_quantile=True):
    X = X.astype(float).copy()
    if pos_idx:
        X[:, pos_idx] = np.log1p(X[:, pos_idx])
    if center_idx:
        X[:, center_idx] = signed_log(X[:, center_idx])

    denom = highs - lows
    denom[denom == 0] = 1.0
    X_scaled = -1.0 + 2.0 * (X - lows) / denom

    if clip_to_quantile:
        np.clip(X_scaled, -1.0, 1.0, out=X_scaled)

    return X_scaled
