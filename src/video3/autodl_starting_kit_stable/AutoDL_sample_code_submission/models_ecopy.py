import sys

import torch.nn.functional as F
from tf_model_zoo.ECOfull_py.bninception import bninception_pretrained
from tf_model_zoo.ECOfull_py.efficientnet import EfficientNet
from tf_model_zoo.ECOfull_py.resnet_3d import resnet3d, resnet3d_eff
from torch import nn
from transforms import *


class ECOfull_efficient(nn.Module):
    def __init__(
        self,
        num_classes,
        num_segments,
        modality,
        dropout=0.8,
        partial_bn=True,
        freeze_eco=False,
        freeze_interval=[2, 63, -1, -1],
        input_size=224
    ):
        super(ECOfull_efficient, self).__init__()
        self.num_segments = num_segments
        self.channel = 3
        self.reshape = True
        
        self.dropout = dropout
        self.alphadrop = nn.AlphaDropout(p=self.dropout)
        self.modality = modality

        self.input_size = input_size
        print("Input size: ", self.input_size)

        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]
        #self.scale = 1
        #self.efficientnet = EfficientNet(num_classes=num_classes,
        #    width_coef=self.scale, depth_coef=self.scale, 
        #    scale=self.scale,dropout_ratio=0.2,
        #    pl=0.2, endpoint=112, arch='full')
        self.base = EfficientNet(
            num_classes=1000, width_coef=0.7,depth_coef=0.7,scale=0.7,dropout_ratio=0.02,pl=0.02,endpoint=56,arch="full")
        self.resnet3d_eff = resnet3d_eff()
        self.batch_norm_2do = nn.BatchNorm1d(896, eps=1e-3, momentum=0.01)
        #self.batch_norm_2do = nn.BatchNorm1d((512+896), eps=1e-3, momentum=0.01)
        # changed, because we dont get bias out of efficientnet
        #self.fc = nn.Linear(1792, num_classes)  # kernels * segments = 40*16 = 640
        #self.fc = nn.Linear((512+896), num_classes)  # kernels * segments = 40*16 = 640
        
        reduction = 8
        self.transfer2D = nn.Sequential(
            nn.Linear(896, 2048//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.8),
            nn.Linear(2048//reduction, 896, bias=False),
            nn.Sigmoid())
        
        self.fc = nn.Linear(896, num_classes)  # kernels * segments = 40*16 = 640
        self._enable_pbn = partial_bn
        self._enable_freeze_eco = freeze_eco
        self._freeze_interval = freeze_interval
        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def forward(self, input, num_segments=0):
        """
        input: (bs, c*ns, h, w)
        """
        if num_segments != 0: self.num_segments = num_segments
        sample_len = 3
        bs, c_ns, h, w = input.shape

        input = input.view((-1, sample_len) + input.size()[-2:])  # (bs*ns, c, h, w)

        # base model: BNINception pretrained model
        x, x_b = self.base(input)
        # print('bs : ', x.shape, x_b.shape)
        
        #print('shape after effnet: ', x.shape)
        # reshape (2D to 3D)
        '''
        x = x.view(bs, 68, self.num_segments, 16, 16)  # (bs, 40, ns, 28, 28)
        #print('shape after segments effnet: ', x.shape)
        # 3D resnet
        x = self.resnet3d_eff(x)  # (bs, 512, 4, 7, 7)

        # global average pooling (modified version to fit for arbitrary the number of segments
        bs, _, fc, fh, hw = x.shape
        x = F.avg_pool3d(x, kernel_size=(fc, fh, hw), stride=(1, 1, 1))

        # fully connected
        x = x.view(-1, 512)

        x_b = x_b.view(-1, self.num_segments, 896)
        x_b = torch.mean(x_b, dim=1)  # avg over frames
        
        x = torch.cat((x, x_b), 1)  # concatenate output of 3D net and 2D net
        x = self.batch_norm_2do(x)
        x = self.alphadrop(x)
        x = self.fc(x)
        '''
        #######################
        #'''
        x_b = x_b.view(-1, self.num_segments, 896)
        x_b = torch.mean(x_b, dim=1)  # avg over frames
        x_bt = self.transfer2D(x_b)
        x_b = self.batch_norm_2do(x_bt*x_b)
        x_b = self.alphadrop(x_b)
        x = self.fc(x_b)
        #'''

        return x

    def partialBN(self, enable):
        self._enable_pbn = enable

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(ECOfull_efficient, self).train(mode)
        count = 0
        if self._enable_freeze_eco:
            print(
                "Freezing all layers in ECO except the first one and last layers for regression."
            )
            for m in self.efficientnet.modules():
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
            for m in self.efficientnet.modules():
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
        if self.input_size == 128:
            return torchvision.transforms.Compose(
                [
                    GroupScale(scale=0.6),
                    GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                    GroupRandomHorizontalFlip(is_flow=False),
                    GroupRandomGrayscale(p=0.001),
                ]
            )
        else:
            return torchvision.transforms.Compose(
                [
                    GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                    GroupRandomHorizontalFlip(is_flow=False)
                ]
            )


class ECOfull(nn.Module):
    def __init__(
        self,
        num_classes,
        num_segments,
        modality,
        dropout=0.8,
        partial_bn=True,
        freeze_eco=False,
        freeze_interval=[2, 63, -1, -1],
        input_size=224
    ):
        super(ECOfull, self).__init__()
        self.num_segments = num_segments
        self.channel = 3
        self.reshape = True
        self.dropout = dropout
        self.modality = modality

        self.input_size = input_size
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]

        self.bninception_pretrained = bninception_pretrained(
            num_classes=1000, eco_version="full"
        )
        self.resnet3d = resnet3d()
        self.fc = nn.Linear(1536, num_classes)

        self._enable_pbn = partial_bn
        self._enable_freeze_eco = freeze_eco
        self._freeze_interval = freeze_interval
        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def forward(self, input, num_segments=0):
        """
        input: (bs, c*ns, h, w)
        """
        if num_segments != 0: self.num_segments = num_segments
        sample_len = 3
        bs, c_ns, h, w = input.shape
        input = input.view((-1, sample_len) + input.size()[-2:])  # (bs*ns, c, h, w)
        # base model: BNINception pretrained model
        x, x_b = self.bninception_pretrained(input)
        #print('bs : ', x.shape, x_b.shape)
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
        if self.input_size == 128:
            return torchvision.transforms.Compose(
                [
                    GroupScale(scale=0.6),
                    GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                    GroupRandomHorizontalFlip(is_flow=False),
                    GroupRandomGrayscale(p=0.001),
                ]
            )
        else:
            return torchvision.transforms.Compose(
                [
                    GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                    GroupRandomHorizontalFlip(is_flow=False)
                ]
            )


class ECO_bninception(nn.Module):
    def __init__(
        self,
        num_classes,
        num_segments,
        modality,
        dropout=0.8,
        partial_bn=True,
        freeze_eco=False,
        freeze_interval=[2, 63, -1, -1], input_size=224
    ):
        super(ECO_bninception, self).__init__()
        self.num_segments = num_segments
        self.channel = 3
        self.reshape = True
        self.dropout = dropout
        self.modality = modality
        self.num_classes = num_classes
        
        self.input_size = input_size #224
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]
        
        self.dropout = 0.1
        #print(self.dropout)
        self.alphadrop = nn.AlphaDropout(p=self.dropout)
        
        reduction = 8
        self.transfer2D = nn.Sequential(
            nn.Linear(1024, 2048//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.8),
            nn.Linear(2048//reduction, 1024, bias=False),
            nn.Sigmoid())
        
        self.batch_norm_2do = nn.BatchNorm1d(1024, eps=1e-3, momentum=0.01)

        self.bninception_pretrained = bninception_pretrained(
            num_classes=1000, eco_version="bninception"
        )

        self.last_linear = nn.Linear (1024, num_classes)
        self._enable_pbn = partial_bn
        self._enable_freeze_eco = freeze_eco
        self._freeze_interval = freeze_interval
        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def forward(self, input, num_segments=0):
        """
        input: (bs, c*ns, h, w)
        """
        if num_segments != 0: self.num_segments = num_segments
        sample_len = 3
        bs, c_ns, h, w = input.shape
        input = input.view((-1, sample_len) + input.size()[-2:])  # (bs*ns, c, h, w)
        # base model: BNINception pretrained model
        x = self.bninception_pretrained(input)
        
        x_t = self.transfer2D(x)
        x = self.batch_norm_2do(x_t*x)
        x = self.alphadrop(x)
        #print("x.shape: ", x.shape)
        
        x = x.view(-1, self.num_segments, 1024)  # (bs, 96, ns, 28, 28)
        x = torch.mean(x, dim=1)  # avg over frames
        #print("x.shape2: ", x.shape)

        x = self.last_linear(x)
        #print("x.shape3: ", x.shape)



        #print('bs3 : ', x.shape)

        return x

    def partialBN(self, enable):
        self._enable_pbn = enable

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(ECO_bninception, self).train(mode)
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
    def __init__(self, num_classes, num_segments, dropout=0.8, input_size=224):
        super(ECO, self).__init__()
        self.num_segments = num_segments
        self.channel = 3
        self.reshape = True
        self.dropout = dropout

        self.input_size = input_size

        # self.bninception = bninception()
        self.efficientnet = bninception_pretrained(
            num_classes=1000, eco_version="lite"
        )
        self.resnet3d = resnet3d()

        self.fc = nn.Linear(512, num_classes)

    def forward(self, input, num_segments=0):
        """
        input: (bs, c*ns, h, w)
        """
        if num_segments != 0: self.num_segments = num_segments
        sample_len = 3
        bs, c_ns, h, w = input.shape
        input = input.view((-1, sample_len) + input.size()[-2:])  # (bs*ns, c, h, w)

        # base model: BNINception pretrained model
        x = self.efficientnet(input)

        # reshape (2D to 3D)
        x = x.view(bs, 40, self.num_segments, 28, 28)  # (bs, 96, ns, 28, 28)

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
        if self.input_size == 128:
            return torchvision.transforms.Compose(
                [
                    GroupScale(scale=0.6),
                    GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                    GroupRandomHorizontalFlip(is_flow=False),
                    GroupRandomGrayscale(p=0.001),
                ]
            )
        else:
            return torchvision.transforms.Compose(
                [
                    GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                    GroupRandomHorizontalFlip(is_flow=False)
                ]
            )
