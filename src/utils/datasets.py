from torch.utils.data import DataLoader, TensorDataset
import torch

def create_dataloader(features, labels, batch_size=32, shuffle=True):
    dataset = TensorDataset(torch.tensor(features, dtype=torch.float32),
                            torch.tensor(labels, dtype=torch.long))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
