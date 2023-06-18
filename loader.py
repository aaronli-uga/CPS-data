'''
Author: Qi7
Date: 2023-06-17 21:23:27
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2023-06-17 21:24:23
Description: 
'''
from torch.utils.data import Dataset
import torch
import numpy as np

class RegularLoader(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        features = self.X[idx]
        target = self.y[idx]
        return features, target