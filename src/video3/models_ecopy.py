import sys

import torch.nn.functional as F
from tf_model_zoo.ECOfull_py.bninception import bninception_pretrained
from tf_model_zoo.ECOfull_py.efficientnet import efficientnet
from tf_model_zoo.ECOfull_py.resnet_3d import resnet3d
from torch import nn
from transforms import *


class ECOfull(nn.Module):
    def __init__(
        self,
        num_classes,
        num_segments,
        modality,
        dropout=0.8,
        partial_bn=True,
        freeze_eco=False,
        freeze_interval=[2, 63, -1, -1]
    ):
        super(ECOfull, self).__init__()
        self.num_segments = num_segments
        self.channel = 3
        self.reshape = True
        self.dropout = dropout
        self.modality = modality

        self.input_size = 224
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]

        #self.bninception_pretrained = bninception_pretrained(
        #    num_classes=1000, eco_version="full"
        #)
        self.bninception_pretrained = efficientnet(
            num_classes=1000, endpoint=28)
        )
        self.resnet3d = resnet3d()
        self.fc = nn.Linear(1536, num_classes)

        self._enable_pbn = partial_bn
        self._enable_freeze_eco = freeze_eco
        self._freeze_interval = freeze_interval
        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def forward(self, input):
        """
        input: (bs, c*ns, h, w)
        """

        sample_len = 3
        bs, c_ns, h, w = input.shape
        input = input.view((-1, sample_len) + input.size()[-2:])  # (bs*ns, c, h, w)

        # base model: BNINception pretrained model
        x, x_b = self.bninception_pretrained(input)

        # reshape (2D to 3D)
        x = x.view(bs, 96, self.num_segments, 28, 28)  # (bs, 96, ns, 28, 28)

        # 3D resnet
        x = self.resnet3d(x)  # (bs, 512, 4, 7, 7)

        # global average pooling (modified version to fit for arbitrary the number of segments
        bs, _, fc, fh, hw = x.shape
        x = F.avg_pool3d(x, kernel_size=(fc, fh, hw), stride=(1, 1, 1))

        # fully connected
        x = x.view(-1, 512)

        x_b = x_b.view(-1, self.num_segments, 1024)
        x_b = torch.mean(x_b, dim=1)  # avg over frames
        x = torch.cat((x, x_b), 1)  # concatenate output of 3D net and 2D net
        x = self.fc(x)

        return x

    def partialBN(self, enable):
        self._enable_pbn = enable

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(ECOfull, self).train(mode)
        count = 0
        if self._enable_freeze_eco:
            print(
                "Freezing all layers in ECO except the first one and last layers for regression."
            )
            for m in self.bninception_pretrained.modules():
                # print(m)
                if (
                    not isinstance(m, nn.ReLU) and not isinstance(m, nn.MaxPool2d) and
                    not isinstance(m, nn.AvgPool2d) and
                    not isinstance(m, nn.AvgPool3d) and not isinstance(m, nn.Dropout)
                ):
                    count += 1
                    # print("000"*30)
                    # print(count)
                    assert len(
                        self._freeze_interval
                    ) == 4, '--freeze_interval must have 4 int numbers, {} numbers: {} are given'.format(
                        len(self._freeze_interval), self._freeze_interval
                    )
                    if (
                        count >= self._freeze_interval[0] and
                        count <= self._freeze_interval[1]
                    ) or (
                        count >= self._freeze_interval[2] and
                        count <= self._freeze_interval[3]
                    ):
                        m.eval()
                        print("Freezing - {} : {} ".format(count, m))
                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
                    elif count != 1:
                        print("No Freezing - {} : {} ".format(count, m))

        count = 0
        if self._enable_pbn:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.bninception_pretrained.modules():
                if (isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d)):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()

                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
        else:
            print("No BN layer Freezing.")

    def get_optim_policies(self):
        first_3d_conv_weight = []
        first_3d_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []

        conv_2d_cnt = 0
        conv_3d_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            # (conv1d or conv2d) 1st layer's params will be append to list: first_conv_weight & first_conv_bias, total num 1 respectively(1 conv2d)
            # (conv1d or conv2d or Linear) from 2nd layers' params will be append to list: normal_weight & normal_bias, total num 69 respectively(68 Conv2d + 1 Linear)
            if isinstance(m, torch.nn.Conv2d):
                ps = list(m.parameters())
                conv_2d_cnt += 1
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])

            elif isinstance(m, torch.nn.Conv3d):
                ps = list(m.parameters())
                conv_3d_cnt += 1
                if conv_3d_cnt == 1:
                    first_3d_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_3d_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])

            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
            # (BatchNorm1d or BatchNorm2d) params will be append to list: bn, total num 2 (enabled pbn, so only: 1st BN layer's weight + 1st BN layer's bias)
            elif isinstance(m, torch.nn.BatchNorm1d):
                bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm3d):
                bn_cnt += 1
                # 4
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError(
                        "New atomic module type: {}. Need to give it a learning policy".
                        format(type(m))
                    )
        return [
            {
                'params': first_3d_conv_weight,
                'lr_mult': 5 if self.modality == 'Flow' else 1,
                'decay_mult': 1,
                'name': "first_3d_conv_weight"
            },
            {
                'params': first_3d_conv_bias,
                'lr_mult': 10 if self.modality == 'Flow' else 2,
                'decay_mult': 0,
                'name': "first_3d_conv_bias"
            },
            {
                'params': normal_weight,
                'lr_mult': 1,
                'decay_mult': 1,
                'name': "normal_weight"
            },
            {
                'params': normal_bias,
                'lr_mult': 2,
                'decay_mult': 0,
                'name': "normal_bias"
            },
            {
                'params': bn,
                'lr_mult': 1,
                'decay_mult': 0,
                'name': "BN scale/shift"
            },
        ]

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self):
        return torchvision.transforms.Compose(
            [
                GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                GroupRandomHorizontalFlip(is_flow=False)
            ]
        )


class ECO(nn.Module):
    def __init__(self, num_classes, num_segments, dropout=0.8):
        super(ECO, self).__init__()
        self.num_segments = num_segments
        self.channel = 3
        self.reshape = True
        self.dropout = dropout

        self.input_size = 224

        # self.bninception = bninception()
        self.bninception_pretrained = bninception_pretrained(
            num_classes=1000, eco_version="lite"
        )
        self.resnet3d = resnet3d()

        self.fc = nn.Linear(512, num_classes)

    def forward(self, input):
        """
        input: (bs, c*ns, h, w)
        """

        sample_len = 3
        bs, c_ns, h, w = input.shape
        input = input.view((-1, sample_len) + input.size()[-2:])  # (bs*ns, c, h, w)

        # base model: BNINception pretrained model
        x = self.bninception_pretrained(input)

        # reshape (2D to 3D)
        x = x.view(bs, 96, self.num_segments, 28, 28)  # (bs, 96, ns, 28, 28)

        # 3D resnet
        x = self.resnet3d(x)  # (bs, 512, 4, 7, 7)

        # global average pooling (modified version to fit for arbitrary the number of segments
        bs, _, fc, fh, hw = x.shape
        x = F.avg_pool3d(x, kernel_size=(fc, fh, hw), stride=(1, 1, 1))

        # fully connected
        x = x.view(-1, 512)
        x = self.fc(x)

        return x

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self):
        return torchvision.transforms.Compose(
            [
                GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                GroupRandomHorizontalFlip(is_flow=False)
            ]
        )
