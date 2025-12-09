import numpy as np
import torch

def signed_log(x: np.ndarray) -> np.ndarray:
    """Signed log transform for zero-centered features."""
    return np.sign(x) * np.log1p(np.abs(x))

def compute_combined_quantiles(datasets,
                               n_sample_per_target=200_000,
                               pos_idx=None, center_idx=None,
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

    dim = datasets[0].shape[1]

    if pos_idx and all(0 <= x < dim for x in pos_idx):
        X_comb[:, pos_idx] = np.log1p(X_comb[:, pos_idx])
    if center_idx and all(0 <= x < dim for x in center_idx):
        X_comb[:, center_idx] = signed_log(X_comb[:, center_idx])

    Xq = X_comb.copy()

    lows = np.quantile(Xq, low_q, axis=0)
    highs = np.quantile(Xq, high_q, axis=0)
    highs = np.where(highs == lows, lows + 1e-6, highs)

    return lows, highs

class TransformAndScale:
    def __init__(self, pos_idx, center_idx, lows, highs, clip_to_quantile=False):

        self.pos_idx = pos_idx
        self.center_idx = center_idx
        self.clip_to_quantile = clip_to_quantile

        if self.clip_to_quantile and self.remove_outliers:
            raise ValueError("Only one of clip_to_quantile or remove_outliers can be True.")

        self.lows = torch.tensor(lows, dtype=torch.float32)
        self.highs = torch.tensor(highs, dtype=torch.float32)

        self.denom = self.highs - self.lows
        self.denom[self.denom == 0] = 1.0

    def signed_log(self, x):
        return np.sign(x) * np.log1p(np.abs(x))

    def __call__(self, X):
        lows = self.lows.to(X.device)
        highs = self.highs.to(X.device)
        denom = self.denom.to(X.device)

        X = X.clone().float()

        # add batch dim if sample is 1D
        if X.ndim == 1:
            X = X.unsqueeze(0)

        if self.pos_idx:
            X[:, self.pos_idx] = torch.log1p(X[:, self.pos_idx])
        if self.center_idx:
            X[:, self.center_idx] = self.signed_log(X[:, self.center_idx])

        X_scaled = -1.0 + 2.0 * (X - lows) / denom

        if self.clip_to_quantile:
            X_scaled = torch.clamp(X_scaled, -1.0, 1.0)

        # remove batch dim if input was 1D
        if X_scaled.shape[0] == 1:
            X_scaled = X_scaled.squeeze(0)

        return X_scaled