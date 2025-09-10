import torch
import numpy as np

def evaluate_label_distribution(feat_model, clf_model, X_target, n_classes, label_encoder, device="cpu", domain_name="Target"):
    """
    Apply trained models to target data and print predicted label distribution.
    """
    feat_model.eval()
    clf_model.eval()

    X_tensor = torch.from_numpy(X_target).float().to(device)
    with torch.no_grad():
        features = feat_model(X_tensor)
        logits = clf_model(features)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    # Compute label distribution
    counts = np.bincount(preds, minlength=n_classes)
    probs = np.round(100 * counts / counts.sum(), 2)
    
    print(f"\nLabel distribution for {domain_name}:")
    for i, p in enumerate(probs):
        print(f"Class {label_encoder.inverse_transform([i])[0]}: {p:.3f}")

    return probs
