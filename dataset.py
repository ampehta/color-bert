import os

from torch.utils.data import DataLoader, Dataset


class ColorDataset(Dataset):
    def __init__(self, data_path, label_path=os.path.join("data", "labels.txt")):
        with open(data_path) as f:
            self.data = f.read().splitlines()
        with open(label_path) as f:
            self.labels = f.read().splitlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
