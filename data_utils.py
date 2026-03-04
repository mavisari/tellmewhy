from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class MRIDataset(Dataset):
    def __init__(self, data, targets, split, transform=None):
        self.data = data
        self.labels = list(targets)
        self.split = split
        self.transform = transform
        
    def __len__(self):
        return len(self.data) 
    
    def __getitem__(self, idx):
        X = self.data[idx]
        y = self.labels[idx]
        
        if self.transform is not None:
            X = self.transform(X)
        return X, y