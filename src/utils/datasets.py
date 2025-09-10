from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch



class NumpyDataset(Dataset):
    """Wrap a NumPy array (X) and optional labels (y) into a PyTorch Dataset."""
    def __init__(self, X, y=None):
        self.X = torch.from_numpy(X).float()
        self.y = None if y is None else torch.from_numpy(y).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx], 0  # dummy label
        return self.X[idx], self.y[idx]

def create_dataloader(features, labels, batch_size=32, shuffle=True):
    dataset = TensorDataset(torch.tensor(features, dtype=torch.float32),
                            torch.tensor(labels, dtype=torch.long))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
