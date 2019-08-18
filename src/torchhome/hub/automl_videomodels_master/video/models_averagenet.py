import logging

import torch
import torch.nn.functional as F
import torchvision
from torch import nn

from .tf_model_zoo.ECOfull_py.efficientnet import EfficientNet
from .transforms import (
    GroupMultiScaleCrop, GroupRandomGrayscale, GroupRandomHorizontalFlip, GroupScale
)

logger = logging.getLogger(__file__)


class Averagenet_feature(nn.Module):
    def __init__(
        self,
        num_classes,
        num_segments,
        modality,
        base=None,
        # BNinception: 1024; Effnet: 1280,
        n_output_base=1280,
        dropout=0.8,
        input_size=224,
        partial_bn=False,
        freeze=False,
        freeze_interval=[2, 63, -1, -1],
        scaleing=0.7,
    ):
        super(Averagenet_feature, self).__init__()
        self.num_segments = num_segments
        self.channel = 3
        self.reshape = True
        self.scaling = scaleing

        self.dropout = dropout
        self.alphadrop = nn.AlphaDropout(p=self.dropout)
        self.modality = modality

        self.input_size = input_size
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]
        if base is None:
            self.base = EfficientNet(
                num_classes=num_classes,
                width_coef=self.scaling,
                depth_coef=self.scaling,
                scale=self.scaling,
                dropout_ratio=0.2,
                pl=0.2,
                endpoint=112,
                arch='full'
            )
            # bninception_pretrained(
            #         num_classes=1000, eco_version="full")
            # efficientnet_b0(
            #     num_classes=num_classes, arch='full')
        else:
            self.base = base
        self.n_output_base = int(n_output_base * self.scaling)
        self.fc = nn.Linear(self.n_output_base, num_classes)
        self._enable_pbn = partial_bn
        self._enable_freeze = freeze
        self._freeze_interval = freeze_interval
        if partial_bn:
            self.partialBN(True)

    def forward(self, input, num_segments=0):
        """
        input: (bs, c*ns, h, w)
        """
        if num_segments != 0:
            self.num_segments = num_segments
        sample_len = 3
        bs, c_ns, h, w = input.shape
        # logger.debug('ip : ', input.shape)
        input = input.view((-1, sample_len) + input.size()[-2:])  # (bs*ns, c, h, w)
        # logger.debug('ip_view : ', input.shape)
        # base model: BNINception pretrained model
        x = self.base(input)
        # logger.debug('bs : ', x.shape)

        # logger.debug('shape after effnet: ', x.shape)
        # reshape (2D to 3D)
        x = x.view(bs, self.num_segments, self.n_output_base)  # (bs, 40, ns, 28, 28)
        # logger.debug('shape after segments effnet: ', x.shape)
        # global average pooling (modified version to fit for arbitrary the number of segments
        # x = F.adaptive_avg_pool1d(x, self.n_output_base)
        bs, seg, _ = x.shape
        x = F.avg_pool2d(x, kernel_size=(seg, 1), stride=(1, 1))
        # logger.debug('shape after pool: ', x.shape)
        x = x.view(-1, self.n_output_base)
        # logger.debug('shape after view: ', x.shape)
        x = self.alphadrop(x)
        x = self.fc(x)

        return x

    def partialBN(self, enable):
        self._enable_pbn = enable

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(Averagenet_feature, self).train(mode)
        count = 0
        if False:  # TODO: why always true??self._enable_freeze:
            logger.debug(
                "Freezing all layers in ECO except the first one and last layers for regression."
            )
            for m in self.base.modules():
                # logger.debug(m)
                if (
                    not isinstance(m, nn.ReLU) and not isinstance(m, nn.MaxPool2d) and
                    not isinstance(m, nn.AvgPool2d) and
                    not isinstance(m, nn.AvgPool3d) and not isinstance(m, nn.Dropout)
                ):
                    count += 1
                    # logger.debug("000"*30)
                    # logger.debug(count)
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
                        logger.debug("Freezing - {} : {} ".format(count, m))
                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
                    elif count != 1:
                        logger.debug("No Freezing - {} : {} ".format(count, m))

        count = 0
        if self._enable_pbn:
            logger.debug("Freezing BatchNorm2D except the first one.")
            for m in self.base.modules():
                if (isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d)):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()

                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
        else:
            logger.debug("No BN layer Freezing.")

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
        if self.modality == 'RGB':
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
                        GroupRandomHorizontalFlip(is_flow=False),
                    ]
                )
        elif self.modality == 'Flow':
            return torchvision.transforms.Compose(
                [
                    GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                    GroupRandomHorizontalFlip(is_flow=True)
                ]
            )
        elif self.modality == 'RGBDiff':
            return torchvision.transforms.Compose(
                [
                    GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                    GroupRandomHorizontalFlip(is_flow=False)
                ]
            )


class Averagenet(nn.Module):
    def __init__(
        self,
        num_classes,
        num_segments,
        modality,
        base=None,
        # BNinception: 1024; Effnet: 1280,
        n_output_base=1280,
        dropout=0.8,
        input_size=224,
        partial_bn=False,
        freeze=False,
        freeze_interval=[2, 63, -1, -1],
        scaleing=0.7,
    ):
        super(Averagenet, self).__init__()
        self.num_segments = num_segments
        self.num_classes = num_classes
        self.channel = 3
        self.reshape = True
        self.scaling = scaleing
        self._dropout = dropout
        self.alphadrop = nn.AlphaDropout(p=self.dropout)
        self.modality = modality

        self.input_size = input_size
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]
        if base is None:
            self.base = EfficientNet(
                num_classes=num_classes,
                width_coef=.7,
                depth_coef=.7,
                scale=.7,
                dropout_ratio=0.2,
                pl=0.2,
                endpoint=112,
                arch='full'
            )
            # bninception_pretrained(
            #         num_classes=1000, eco_version="full")
            # efficientnet_b0(
            #     num_classes=num_classes, arch='full')
        else:
            self.base = base
        self.n_output_base = int(n_output_base * self.scaling)
        self.fc = nn.Linear(self.n_output_base, num_classes)
        self._enable_pbn = partial_bn
        self._enable_freeze = freeze
        self._freeze_interval = freeze_interval
        if partial_bn:
            self.partialBN(True)

    @property
    def dropout(self):
        return self._dropout

    @dropout.setter
    def dropout(self, val):
        self.alphadrop = nn.AlphaDropout(p=val)
        self._dropout = val

    def forward(self, input, num_segments=0):
        """
        input: (bs, c*ns, h, w)
        """
        if num_segments != 0:
            self.num_segments = num_segments
        sample_len = 3
        bs, c_ns, h, w = input.shape
        input = input.view((-1, sample_len) + input.size()[-2:])  # (bs*ns, c, h, w)
        # base model: BNINception pretrained model
        x = self.base(input)
        self.alphadrop(x)
        x = self.fc(x)
        x = x.view(-1, self.num_segments, self.num_classes)  # (bs, 96, ns, 28, 28)
        x = torch.mean(x, dim=1)  # avg over frames

        return x

    def partialBN(self, enable):
        self._enable_pbn = enable

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(Averagenet, self).train(mode)
        count = 0
        if False:  # TODO: why always true??self._enable_freeze:
            logger.debug(
                "Freezing all layers in ECO except the first one and last layers for regression."
            )
            for m in self.base.modules():
                # logger.debug(m)
                if (
                    not isinstance(m, nn.ReLU) and not isinstance(m, nn.MaxPool2d) and
                    not isinstance(m, nn.AvgPool2d) and
                    not isinstance(m, nn.AvgPool3d) and not isinstance(m, nn.Dropout)
                ):
                    count += 1
                    # logger.debug("000"*30)
                    # logger.debug(count)
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
                        logger.debug("Freezing - {} : {} ".format(count, m))
                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
                    elif count != 1:
                        logger.debug("No Freezing - {} : {} ".format(count, m))

        count = 0
        if self._enable_pbn:
            logger.debug("Freezing BatchNorm2D except the first one.")
            for m in self.base.modules():
                if (isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d)):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()

                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
        else:
            logger.debug("No BN layer Freezing.")

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
        if self.modality == 'RGB':
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
                        GroupRandomHorizontalFlip(is_flow=False),
                    ]
                )
        elif self.modality == 'Flow':
            return torchvision.transforms.Compose(
                [
                    GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                    GroupRandomHorizontalFlip(is_flow=True)
                ]
            )
        elif self.modality == 'RGBDiff':
            return torchvision.transforms.Compose(
                [
                    GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                    GroupRandomHorizontalFlip(is_flow=False)
                ]
            )


class StagedAveragenet(Averagenet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # NOTE(Philipp): This is copied from above but I highly object to
        # a hard set dropout rate I cannot change from outside even with init!
        self.base = EfficientNet(
            num_classes=self.num_classes,
            width_coef=.7,
            depth_coef=.7,
            scale=.7,
            dropout_ratio=0.2,
            pl=0.2,
            endpoint=None,
            arch='full',
            eager=True
        )
        self.base.arch = ''
        self.endpoints = [
            112, 56, 28, 14, 13, 12, 7, 0
        ]  # Check effnet class if you re curious
        self.base.endpoint = self.endpoints.pop(0)
        self._old_fc = None
        self.fc = None

    def forward(self, input, num_segments=0):
        """
        input: (bs, c*ns, h, w)
        """
        if num_segments != 0:
            self.num_segments = num_segments
        sample_len = 3
        bs, c_ns, h, w = input.shape
        input = input.view((-1, sample_len) + input.size()[-2:])

        x = self.base(input)

        self.alphadrop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x) if self.fc is not None else self.extent_fc(x)
        x = x.view(-1, self.num_segments, self.num_classes)
        x = torch.mean(x, dim=1)

        return x

    def unlock_next(self):
        if len(self.endpoints) > 0:
            self.base.endpoint = self.endpoints.pop(0)
        self.fc = None

    def extent_fc(self, input):
        in_shape = input.view(input.size(0), -1).shape[1:][0]
        self.fc = nn.Linear(in_shape, self.num_classes)
        self.fc.to(input.device)
        # if self._old_fc is not None:
        #     for p1, p2 in zip(self.fc.parameters(), self._old_fc.parameters()):
        #         p1.requires_grad = False
        #         p1.copy_(p2)
        #         p1.requires_grad = True
        self._old_fc = self.fc
        return self.fc(input)
