import torch.nn as nn


"""
3D ResNet code from: https://github.com/kenshohara/3D-ResNets-PyTorch
"""


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class res3a(nn.Module):
    def __init__(self):
        super(res3a, self).__init__()

        # res3a
        # input: res3a_2n
        self.res3a_2n = conv3x3x3(96, 128, stride=1)
        self.res3a_bn = nn.BatchNorm3d(128)
        self.res3a_relu = nn.ReLU(inplace=False)  # res3b_1
        self.res3b_1 = conv3x3x3(128, 128, stride=1)
        self.res3b_1_bn = nn.BatchNorm3d(128)
        self.res3b_relu = nn.ReLU(inplace=False)  # res3b_2

        self.res3b_bn = nn.BatchNorm3d(128)
        self.res3b_relu = nn.ReLU(inplace=False)

    def forward(self, input):
        residual = self.res3a_2n(input)

        x = self.res3a_bn(residual)  # result of res3a_2n
        x = self.res3a_relu(x)  # res3b_1

        x = self.res3b_1(x)
        x = self.res3b_1_bn(x)
        x = self.res3b_relu(x)  # res3b_2

        x += residual

        x = self.res3b_bn(x)
        x = self.res3b_relu(x)

        return x  # res3b_relu


class res4(nn.Module):
    def __init__(self):
        super(res4, self).__init__()

        self.res4a_1 = conv3x3x3(128, 256, stride=2)
        self.res4a_1_bn = nn.BatchNorm3d(256)
        self.res4a_1_relu = nn.ReLU(inplace=False)

        self.res4a_2 = conv3x3x3(256, 256)

        self.res4a_down = nn.Sequential(
            nn.Conv3d(128, 256 * 1, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm3d(256 * 1)
        )

        self.res4a_bn = nn.BatchNorm3d(256)
        self.res4a_relu = nn.ReLU(inplace=False)

        self.res4b_1 = conv3x3x3(256, 256, stride=1)

        self.res4b_1_bn = nn.BatchNorm3d(256)
        self.res4b_1_relu = nn.ReLU(inplace=False)

        self.res4b_2 = conv3x3x3(256, 256, stride=1)

        self.res4b_bn = nn.BatchNorm3d(256)
        self.res4b_relu = nn.ReLU(inplace=False)

    def forward(self, input):
        residual = self.res4a_down(input)

        x = self.res4a_1(input)  # take res3b_relu directly, res4a_1
        x = self.res4a_1_bn(x)
        x = self.res4a_1_relu(x)  # res4a_2

        x = self.res4a_2(x)

        x += residual  # res4b

        residual2 = x

        x = self.res4a_bn(x)
        x = self.res4a_relu(x)

        x = self.res4b_1(x)

        x = self.res4b_1_bn(x)
        x = self.res4b_1_relu(x)

        x = self.res4b_2(x)

        x += residual2

        x = self.res4b_bn(x)
        x = self.res4b_relu(x)

        return x  # res4b_relu


class res5(nn.Module):
    def __init__(self):
        super(res5, self).__init__()

        self.res5a_1 = conv3x3x3(256, 512, stride=2)
        self.res5a_1_bn = nn.BatchNorm3d(512)
        self.res5a_1_relu = nn.ReLU(inplace=False)

        self.res5a_2 = conv3x3x3(512, 512)

        self.res5a_down = nn.Sequential(
            nn.Conv3d(256, 512 * 1, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm3d(512 * 1)
        )

        self.res5a_bn = nn.BatchNorm3d(512)
        self.res5a_relu = nn.ReLU(inplace=False)

        self.res5b_1 = conv3x3x3(512, 512, stride=1)

        self.res5b_1_bn = nn.BatchNorm3d(512)
        self.res5b_1_relu = nn.ReLU(inplace=False)

        self.res5b_2 = conv3x3x3(512, 512, stride=1)

        self.res5b_bn = nn.BatchNorm3d(512)
        self.res5b_relu = nn.ReLU(inplace=False)

    def forward(self, input):
        residual = self.res5a_down(input)

        x = self.res5a_1(input)  # take res4b_relu directly, res5a_1
        x = self.res5a_1_bn(x)
        x = self.res5a_1_relu(x)  # res4a_2

        x = self.res5a_2(x)

        x += residual  # res5a

        residual2 = x

        x = self.res5a_bn(x)
        x = self.res5a_relu(x)

        x = self.res5b_1(x)

        x = self.res5b_1_bn(x)
        x = self.res5b_1_relu(x)

        x = self.res5b_2(x)

        x += residual2  # res5b

        x = self.res5b_bn(x)
        x = self.res5b_relu(x)

        return x  # res5b_relu


class resnet3d(nn.Module):
    def __init__(self):
        super(resnet3d, self).__init__()

        # res3a
        # input: res3a_2n
        self.res3 = res3a()
        self.res4 = res4()
        self.res5 = res5()

    def forward(self, input):
        # input: (bs, 96, ns, 28 , 28)
        x = self.res3(input)  # (bs, 128, 16, 28, 28)
        x = self.res4(x)  # (bs, 256, 8, 14, 14)
        x = self.res5(x)  # (bs, 512, 4, 7, 7)

        return x
