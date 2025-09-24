from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np

def evaluate_label_distribution(feat_model, clf_model, data, n_classes, label_encoder, 
                                device="cpu", batch_size=1024, num_workers=0, pin_memory=True, verbose=False):
    """
    Apply trained models to target data, print predicted label distribution,
    and return both softmax scores and class probabilities.

    Parameters
    ----------
    feat_model : torch.nn.Module
        Feature extractor model.
    clf_model : torch.nn.Module
        Label classifier model.
    data : DataLoader or Dataset
        Either a PyTorch DataLoader or Dataset.
    n_classes : int
        Number of classes.
    label_encoder : sklearn.preprocessing.LabelEncoder
        Encoder to convert class indices to original labels.
    device : str
        'cpu' or 'cuda'.
    batch_size : int
        Batch size if `data` is a Dataset.
    num_workers : int
        Number of workers if `data` is a Dataset.
    pin_memory : bool
        Use pinned memory if `data` is a Dataset and device is CUDA.
    """

    # Wrap dataset into a DataLoader if needed
    if isinstance(data, torch.utils.data.Dataset):
        loader = torch.utils.data.DataLoader(
            data, batch_size=batch_size, shuffle=False, 
            num_workers=num_workers, pin_memory=pin_memory
        )
    else:
        loader = data

    feat_model = feat_model.to(device)
    clf_model = clf_model.to(device)
    
    feat_model.eval()
    clf_model.eval()

    all_preds, all_softmax = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            # Handle (X, y) or just X
            X_batch = batch[0] if isinstance(batch, (list, tuple)) else batch
            X_batch = X_batch.to(device).float()

            features = feat_model(X_batch)
            logits = clf_model(features)

            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds)
            softmax_scores = F.softmax(logits, dim=1)
            all_softmax.append(softmax_scores)

    # Concatenate all batch predictions and softmax scores
    all_preds = torch.cat(all_preds, dim=0).cpu().numpy()
    all_softmax = torch.cat(all_softmax, dim=0).cpu().numpy()

    # Compute label distribution
    counts = np.bincount(all_preds, minlength=n_classes)
    probs = np.round(100 * counts / counts.sum(), 2)

    if verbose:
        for i, p in enumerate(probs):
            print(f"Class {label_encoder.inverse_transform([i])[0]}: {p:.3f}")

    return all_softmax, probs
