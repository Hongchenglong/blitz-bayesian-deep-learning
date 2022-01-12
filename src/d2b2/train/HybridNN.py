import torch.nn as nn
import torch.nn.functional as F
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator

@variational_estimator
class HybridNN(nn.Module):
    def __init__(self, dim1=256, dim2=128, width=64):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, dim1)
        self.fc2 = nn.Linear(dim1, dim2)
        self.fc3 = BayesianLinear(dim2, width)
        self.fc4 = BayesianLinear(width, 10)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        return out
