import torch.nn as nn
import torch.nn.functional as F


class MLPClassifier(nn.Module):
    def __init__(self, input_feature, hidden_size):
        super(MLPClassifier, self).__init__()
        self.layer1 = nn.Linear(input_feature, hidden_size)
        self.layer2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return x.squeeze()
