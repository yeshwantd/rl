import torch
from torch import nn

class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, state):
        # state has dimensions B X state_dim (which is 8 for lunar lander)
        out = self.relu(self.fc1(state))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        # out = self.softmax(out)
        return out