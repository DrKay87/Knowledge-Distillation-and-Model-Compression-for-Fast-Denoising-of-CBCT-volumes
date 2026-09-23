"""Dataset utilities matching the original notebook data convention."""
import pickle
import torch
from torch.utils.data import Dataset

def load_data(pickle_file):
    with open(pickle_file, "rb") as f:
        return pickle.load(f)

class CBCTDataset(Dataset):
    def __init__(self, noisy_patches, target_patches):
        if len(noisy_patches) != len(target_patches):
            raise ValueError("Noisy and target arrays must have equal length.")
        self.noisy = noisy_patches
        self.target = target_patches

    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, idx):
        noisy = torch.tensor(self.noisy[idx], dtype=torch.float32)
        target = torch.tensor(self.target[idx], dtype=torch.float32)
        return noisy, target
