import torch
import torch.nn as nn
import torchvision.models as models


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = models.alexnet()
        self.fc = nn.Linear(1000, 2)
    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return torch.sigmoid(x)