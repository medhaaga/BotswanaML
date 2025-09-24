from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch



class NumpyDataset(Dataset):
    """Wrap a NumPy array (X) and optional labels (y) into a PyTorch Dataset."""
    def __init__(self, X, y=None, transform=None):
        self.X = torch.from_numpy(X).float()
        self.y = None if y is None else torch.from_numpy(y).long()
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        X_item = self.X[idx]

        if self.transform:
            X_item = self.transform(X_item)

        if self.y is None:
            y_item = torch.tensor(0, dtype=torch.long)  # dummy scalar label
        else:
            y_item = self.y[idx]

        return X_item, y_item 

def create_dataloader(features, labels, batch_size=32, shuffle=True):
    dataset = TensorDataset(torch.tensor(features, dtype=torch.float32),
                            torch.tensor(labels, dtype=torch.long))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
