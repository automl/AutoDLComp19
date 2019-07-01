import torch
import torch.nn as nn
from tf_model_zoo.compression_layers import Conv2dP, LinearP


class Dummy(nn.Module):
    def __init__(self):
        super(Dummy, self).__init__()

        self.conv1 = Conv2dP(in_channels=3, out_channels=8, kernel_size=3, stride=2)
        self.conv2 = Conv2dP(in_channels=8, out_channels=12, kernel_size=3, stride=2)
        self.conv3 = Conv2dP(in_channels=12, out_channels=16, kernel_size=3, stride=2)
        self.relu = nn.ReLU()
        self.fc = LinearP(in_features=2704, out_features=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = x.view(-1,2704)
        x = self.fc(x)

        return x