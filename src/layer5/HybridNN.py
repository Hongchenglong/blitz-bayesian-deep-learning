import torch.nn as nn
import torch.nn.functional as F
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator

@variational_estimator
class HybridNN(nn.Module):
    def __init__(self):
        super().__init__()
        # self.fc1 = nn.Linear(28 * 28, 512)
        # self.fc2 = nn.Linear(512, 256)
        # self.fc3 = nn.Linear(256, 128)
        # self.fc4 = nn.Linear(128, 64)
        # self.fc5 = BayesianLinear(64, 10)
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = BayesianLinear(128, 10)

    def forward(self, x):
        # out = F.relu(self.fc1(x))
        # out = F.relu(self.fc2(out))
        # out = F.relu(self.fc3(out))
        # out = F.relu(self.fc4(out))
        # out = self.fc5(out)
        # return out
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
