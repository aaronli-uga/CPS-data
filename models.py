'''
Author: Qi7
Date: 2023-06-17 21:05:41
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2023-06-17 23:37:56
Description: 
'''
import torch
import torch.nn as nn


class ANN(nn.Module):
    def __init__(self, n_input, n_classes):
        super(ANN, self).__init__()
        self.num_classes = n_classes
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(n_input, 64)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, n_classes)
        
        # self.fc = nn.Linear(n_input, 32)
        # self.fc1 = nn.Linear(32, n_classes)
        
    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        output = self.fc3(x)
        # x = self.fc(x)
        # x = self.relu(x)
        # output = self.fc1(x)
        if self.num_classes == 1:
            output = self.sigmoid(output)
        return output