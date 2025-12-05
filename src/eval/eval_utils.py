from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset

def evaluate_label_distribution(model, data, n_classes, label_encoder, 
                                device="cpu", batch_size=1024, num_workers=0, pin_memory=True, verbose=True):
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
    if isinstance(data, DataLoader):
        loader = data
    elif isinstance(data, Dataset):
        loader = DataLoader(
            data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    elif isinstance(data, torch.Tensor):
        dataset = TensorDataset(data)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    elif isinstance(data, np.ndarray):
        tensor = torch.as_tensor(data, dtype=torch.float32)
        dataset = TensorDataset(tensor)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    else:
        raise TypeError(
            f"Unsupported data type: {type(data)}. Expected Dataset, Tensor, ndarray, or DataLoader."
        )

    model.eval()

    all_preds, all_softmax = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            # Handle (X, y) or just X
            X_batch = batch[0] if isinstance(batch, (list, tuple)) else batch
            X_batch = X_batch.to(device).float()

            model_output = model(X_batch)
            outputs = model_output[1] if isinstance(model_output, tuple) else model_output
            predictions = torch.argmax(outputs, dim=1)
            all_preds.append(predictions.cpu().numpy())
            all_softmax.append(F.softmax(outputs, dim=1).cpu().numpy())

    # Concatenate all batch predictions and softmax scores
    all_preds = np.concatenate(all_preds)
    all_softmax = np.concatenate(all_softmax)

    # Compute label distribution
    counts = np.bincount(all_preds, minlength=n_classes)
    probs = np.round(100 * counts / counts.sum(), 2)

    if verbose:
        for i, p in enumerate(probs):
            print(f"Class {label_encoder.inverse_transform([i])[0]}: {p:.3f}")

    return all_preds, all_softmax, probs

def evaluate_multilabel_distribution(
    model, 
    data, 
    label_encoder, 
    threshold=0.5,
    device="cpu", 
    batch_size=1024, 
    num_workers=0, 
    pin_memory=True, 
    verbose=True
):
    """
    Evaluate model predictions for MULTI-LABEL classification.
    Counts a class as predicted if sigmoid(logit) >= threshold.
    """

    # Wrap data into DataLoader if needed
    if isinstance(data, DataLoader):
        loader = data
    elif isinstance(data, Dataset):
        loader = DataLoader(
            data, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory
        )
    elif isinstance(data, torch.Tensor):
        loader = DataLoader(
            TensorDataset(data), batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory
        )
    elif isinstance(data, np.ndarray):
        t = torch.as_tensor(data, dtype=torch.float32)
        loader = DataLoader(
            TensorDataset(t), batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory
        )
    else:
        raise TypeError("Unsupported data type")

    model.eval()

    all_probs = []
    all_pred_binary = []

    with torch.no_grad():
        for batch in loader:
            X_batch = batch[0] if isinstance(batch, (list, tuple)) else batch
            X_batch = X_batch.to(device).float()

            # model outputs: feats, logits, probs (if following previous design)
            outputs = model(X_batch)
            logits = outputs[1] if isinstance(outputs, tuple) else outputs

            probs = torch.sigmoid(logits)              # (B, n_classes)
            preds = (probs >= threshold).int()         # binary predictions

            all_probs.append(probs.cpu().numpy())
            all_pred_binary.append(preds.cpu().numpy())

    # Concatenate results
    all_probs = np.concatenate(all_probs, axis=0)           # (N, n_classes)
    all_pred_binary = np.concatenate(all_pred_binary, 0)    # (N, n_classes)

    # Count frequency of each class being predicted
    predicted_counts = all_pred_binary.sum(axis=0)          # (n_classes,)
    predicted_percent = np.round(100 * predicted_counts / len(all_pred_binary), 2)

    if verbose:
        print("\nPredicted class % occurrence:")
        for i, pct in enumerate(predicted_percent):
            label = label_encoder.inverse_transform([i])[0]
            print(f"  {label}: {pct:.2f}%")

    return all_pred_binary, all_probs, predicted_percent
