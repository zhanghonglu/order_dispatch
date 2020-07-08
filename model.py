import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_uints=64, fc2_uints=64):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.norm = torch.nn.BatchNorm1d(num_features=state_size+action_size)
        self.fc1 = nn.Linear(state_size+action_size, fc1_uints)
        self.fc2 = nn.Linear(fc1_uints, fc2_uints)
        self.fc3 = nn.Linear(fc2_uints, 1)

    def forward(self, state_action):
        x = self.norm(state_action)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

