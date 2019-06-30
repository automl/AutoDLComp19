import torch
from torch import nn


class Dummy(nn.Module):
    def __init__(self):
        super(Dummy, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=12, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=16, kernel_size=3, stride=2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(in_features=2704, out_features=1)

    def forward(self, x):
        x = self.conv1(x)
        print(x.shape)
        x = self.relu(x)
        print(x.shape)
        x = self.conv2(x)
        print(x.shape)
        x = self.relu(x)
        print(x.shape)
        x = self.conv3(x)
        print(x.shape)
        x = self.relu(x)
        print(x.shape)
        x = x.view(-1,2704)
        print(x.shape)
        x = self.fc(x)
        print(x.shape)

        return x