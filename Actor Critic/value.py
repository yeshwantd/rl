import torch
from torch import nn

class Value(nn.Module):
    def __init__(self, state_dim):
        super(Value, self).__init__()
        self.state_dim = state_dim
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU() 

    def forward(self, state):
        out = self.relu(self.fc1(state))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out
