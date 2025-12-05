import torch
import torch.nn as nn

# Gradient reversal
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None

def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)

# Feature extractor
class FeatureExtractor(nn.Module):
    def __init__(self, in_dim=9, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
        )

    def forward(self, x):
        return self.net(x)

# Label classifier
class LabelClassifier(nn.Module):
    def __init__(self, in_dim=64, hidden_dim=32, n_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, feat):
        return self.net(feat)

# Domain classifier
class DomainClassifier(nn.Module):
    def __init__(self, in_dim=64, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # binary: source vs. target
        )

    def forward(self, h, lambd=1.0):
        h_rev = grad_reverse(h, lambd)
        return self.net(h_rev)
    
# DANN Model Container

class DANNModel(nn.Module):
    """A container for the three DANN networks to make them compatible with train_run."""
    def __init__(self, feature_extractor, label_classifier, domain_classifier):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.label_classifier = label_classifier
        self.domain_classifier = domain_classifier

    def forward(self, x):
        """The forward pass for inference/validation uses only the feature extractor and label classifier."""
        features = self.feature_extractor(x)
        logits = self.label_classifier(features)
        return features, logits