import torch
import torch.nn as nn
import torch.nn.functional as F

def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute CORAL loss between source and target features.
    source: (n_s, d)
    target: (n_t, d)
    returns scalar tensor.
    """
    # Basic shape checks
    if source.dim() != 2 or target.dim() != 2:
        raise ValueError("source and target must be 2D tensors (batch_size x feat_dim).")
    n_s, d = source.shape
    n_t, d_t = target.shape
    if d != d_t:
        raise ValueError("source and target must have same feature dimension.")

    # If batch too small, return zero loss
    if n_s <= 1 or n_t <= 1:
        return torch.tensor(0.0, device=source.device, dtype=source.dtype)

    # Center the features
    src_mean = torch.mean(source, dim=0, keepdim=True)  # (1, d)
    tgt_mean = torch.mean(target, dim=0, keepdim=True)
    src_centered = source - src_mean     # (n_s, d)
    tgt_centered = target - tgt_mean     # (n_t, d)

    # Compute covariances (d x d)
    # using unbiased estimator with (n-1) denominator
    Cs = (src_centered.t() @ src_centered) / (n_s - 1)
    Ct = (tgt_centered.t() @ tgt_centered) / (n_t - 1)

    # Frobenius norm between covariances
    diff = Cs - Ct
    # squared Frobenius norm
    loss = torch.sum(diff * diff)
    # normalize by 4*d^2 as in the original paper
    loss = loss / (4.0 * (d ** 2))

    return loss

# Example: deep coral combined loss inside a training step
class SimpleFeatureNet(nn.Module):
    """Feature extractor + classifier for multi-label classification."""
    def __init__(self, input_dim, feat_dim, num_classes):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feat_dim),
            nn.ReLU(),

        )
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x, return_features=True):
        feats = self.backbone(x)           # (B, feat_dim)
        logits = self.classifier(feats)    # (B, num_classes)
        
        if return_features:
            return feats, logits
        else:
            return logits
    
    
# Usage in a train step (sketch):
def coral_train_step(model, source_x, source_y, target_x, optimizer, lambda_coral=1.0):
    """
    source_x: (n_s, input_dim) tensor
    source_y: (n_s,) long tensor
    target_x: (n_t, input_dim) tensor (unlabelled)
    """
    model.train()
    optimizer.zero_grad()

    # Forward source and target through same network
    src_feats, src_logits = model(source_x)   # src_feats: (n_s, d)
    tgt_feats, _ = model(target_x)            # tgt_feats: (n_t, d) -- classifier logits for target not used

    # Supervised classification loss on source
    cls_loss = F.cross_entropy(src_logits, source_y)

    # CORAL loss between source and target features
    loss_coral = coral_loss(src_feats, tgt_feats)

    # Total loss
    loss = cls_loss + lambda_coral * loss_coral

    # Backprop
    loss.backward()
    optimizer.step()

    return {
        "total_loss": loss.item(),
        "cls_loss": cls_loss.item(),
        "coral_loss": loss_coral.item()
    }
