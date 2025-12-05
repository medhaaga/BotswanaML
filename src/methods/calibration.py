import numpy as np
import ot
import torch

def calibration(cluster_dist: np.array, 
                cluster_centroids: np.array, 
                cluster_labels: np.array, 
                model: torch.nn.Module, 
                device: torch.device ='cpu', 
                label_dist=None, reg=0.1, tol=1e-10):
    
    """
    Calibrates model-predicted label distributions using optimal transport.

    This function takes the predicted label scores assigned to cluster centroids by a
    neural network model and adjusts them using an entropic optimal transport 
    formulation. The adjustment aligns the model-predicted distribution of labels 
    with a target label distribution while considering the empirical distribution 
    across clusters.

    The optimal transport coupling matrix redistributes mass from clusters to labels,
    producing a calibrated label distribution and updated per-cluster predicted scores.

    Parameters
    ----------
    cluster_dist : np.array
        A 1-D array representing the empirical probability distribution of clusters. 
        Must sum to 1, or will be renormalized.
    
    cluster_centroids : np.array
        Array of cluster centroid feature vectors, used as input to the model for 
        prediction.

    cluster_labels : np.array
        Integer label per sample indicating its cluster assignment; used to map 
        calibrated cluster scores back to sample-level predictions.

    model : torch.nn.Module
        Trained model whose forward pass returns prediction logits or scores for each 
        label, where the second output element ([1]) is assumed to be label logits.

    device : torch.device, optional (default='cpu')
        Device on which to run the model forward pass.

    label_dist : np.array, optional
        Target label distribution used for calibration. If provided, it is normalized.

    reg : float, optional (default=0.1)
        Entropic regularization parameter for the Sinkhorn optimal transport solver.

    tol : float, optional (default=1e-10)
        Small tolerance value used to prevent numerical issues when normalizing and 
        computing logarithms.

    Returns
    -------
    P : np.array
        Optimal transport coupling matrix of shape (num_clusters, num_labels).

    cluster_scores : np.array
        Model-predicted probability scores per cluster before calibration.

    unadjusted_label_dist : np.array
        Label distribution induced directly from model scores weighted by cluster_dist.

    adjusted_label_dist : np.array
        Calibrated label distribution obtained by summing the optimal transport matrix.

    adjusted_scores : np.array
        Per-cluster adjusted scores derived from the optimal transport matrix.

    predictions : np.array
        Final predicted label per sample, based on calibrated scores mapped via 
        cluster_labels.

    Notes
    -----
    The cost matrix is constructed as negative log-probabilities from model outputs and 
    normalized before being passed to the Sinkhorn solver.
    The function assumes the model returns a tuple where the second element corresponds 
    to raw label logits prior to sigmoid activation.

    """

    model = model.to(device)
    model.eval()
    cluster_scores = model(torch.tensor(cluster_centroids, dtype=torch.float32).to(device))[1]
    cluster_scores = torch.sigmoid(cluster_scores).detach().cpu().numpy()

    # Ensure distributions are normalized
    cluster_dist = np.clip(cluster_dist, tol, None)
    cluster_dist /= cluster_dist.sum()
    label_dist = np.clip(label_dist, tol, None)
    label_dist /= label_dist.sum()

    cost_matrix = - np.log(cluster_scores + tol)
    cost_matrix /= np.max(np.abs(cost_matrix))
    P = ot.sinkhorn(cluster_dist, label_dist, cost_matrix, reg, numItermax=500, stopThr=1e-9, verbose=False)

    unadjusted_label_dist = np.sum(cluster_scores * cluster_dist[:, np.newaxis], axis=0)
    adjusted_label_dist = np.sum(P, axis=0)

    adjusted_scores = np.divide(P, cluster_dist[:, np.newaxis], where=cluster_dist[:, np.newaxis]>0)
    predictions = np.argmax(adjusted_scores, axis=1)[cluster_labels]

    return {
        "P": P,
        "cluster_scores": cluster_scores,
        "unadjusted_label_dist": unadjusted_label_dist,
        "adjusted_label_dist": adjusted_label_dist,
        "adjusted_scores": adjusted_scores,
        "predictions": predictions
    }
