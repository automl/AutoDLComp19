from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

__all__ = ['Efficientnet', 'efficientnet']

#pretrained_settings = {
#    'bninception':
#        {
#            'imagenet':
#                {
#                    # Was ported using python2 (may trigger warning)
#                    'url':
#                        'http://data.lip6.fr/cadene/pretrainedmodels/#bn_inception-52deb4733.pth',
#                    # 'url': 'http://yjxiong.me/others/bn_inception-9f5701afb96c8044.pth',
#                    'input_space':
#                        'BGR',
#                    'input_size': [3, 224, 224],
#                    'input_range': [0, 255],
#                    'mean': [104, 117, 128],
#                    'std': [1, 1, 1],
#                    'num_classes':
#                        1000
#                }
#        }
#}

# depthwise is unnescessary to import, read
# When groups == in_channels and out_channels == K * in_channels,
# where K is a positive integer, this operation is also termed in
# literature as depthwise convolution.
# https://pytorch.org/docs/stable/nn.html#torch.nn.Conv2d


class Swish(nn.Module):
    def __init__(self, train_beta=False):
        super(Swish, self).__init__()
        if train_beta:
            self.weight = Parameter(torch.Tensor([1.]))
        else:
            self.weight = 1.0

    def forward(self, input):
        return input * torch.sigmoid(self.weight * input)


def swish(x):
    return x * torch.sigmoid(x)


class SqueezeExcite(nn.Module):
    def __init__(self, inplanes, se_ratio, prob=0.):
        super(SqueezeExcite, self).__init__()
        hidden_dim = int(inplanes / se_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # in Paper Linear in Tf implementation conv2d but 1x1xchannels
        # is like a fully connected layer and dropout can be applied
        self.conv_reduce = nn.Linear(inplanes, hidden_dim, bias=True)
        self.conv_expand = nn.Linear(hidden_dim, inplanes, bias=True)
        #self.conv_reduce = nn.Conv2d(inplanes, hidden_dim,
        #                     kernel_size=(1, 1), stride=(1, 1), # )
        #                     bias=True)
        #self.conv_expand = nn.Conv2d(hidden_dim, inplanes,
        #                     kernel_size=(1, 1), stride=(1, 1), # )
        #                     bias=True)
        self.sigmoid = nn.Sigmoid()
        self.prob = 1 - prob

    def forward(self, x):
        out = self.avg_pool(x).view(x.size(0), -1)
        out = self.conv_reduce(out)
        # Dropout experiments
        #out = F.alpha_dropout(out, 0.00, self.training)
        out = swish(out)
        out = self.conv_expand(out)
        # Dropout experiments
        #out = F.alpha_dropout(out, 0.0*self.prob, self.training)
        out = self.sigmoid(out)
        out = out.unsqueeze(2).unsqueeze(3)
        out = x * out.expand_as(x)
        return out


class InvertedResidual(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, stride, expand, se_ratio, prob=1.0):
        super(InvertedResidual, self).__init__()
        if expand == 1:
            self.conv2 = nn.Conv2d(
                inplanes * expand,
                inplanes * expand,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=inplanes * expand,
                bias=True
            )
            self.bn2 = nn.BatchNorm2d(inplanes * expand, momentum=0.1, eps=1e-5)
            self.se = SqueezeExcite(inplanes * expand, se_ratio)
            self.conv3 = nn.Conv2d(inplanes * expand, planes, kernel_size=1, bias=True)
            self.bn3 = nn.BatchNorm2d(planes, momentum=0.1, eps=1e-5)
        else:
            self.conv1 = nn.Conv2d(inplanes, inplanes * expand, kernel_size=1, bias=True)
            self.bn1 = nn.BatchNorm2d(inplanes * expand, momentum=0.1, eps=1e-5)
            self.conv2 = nn.Conv2d(
                inplanes * expand,
                inplanes * expand,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=inplanes * expand,
                bias=True
            )
            self.bn2 = nn.BatchNorm2d(inplanes * expand, momentum=0.1, eps=1e-5)
            self.se = SqueezeExcite(inplanes * expand, se_ratio, prob)
            self.conv3 = nn.Conv2d(inplanes * expand, planes, kernel_size=1, bias=True)
            self.bn3 = nn.BatchNorm2d(planes, momentum=0.1, eps=1e-5)

        self.correct_dim = (stride == 1) and (inplanes == planes)
        self.prob = torch.Tensor([prob])

    def forward(self, x):
        if self.training:
            if not torch.bernoulli(self.prob):
                return x

        if hasattr(self, 'conv1'):
            out = self.conv1(x)
            out = self.bn1(out)
            out = swish(out)
        else:
            out = x
        out = self.conv2(out)  # depth wise conv
        out = self.bn2(out)
        out = swish(out)
        out = self.se(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.correct_dim:
            out += x

        return out


class MBConv(nn.Module):
    def __init__(
        self,
        inplanes,
        planes,
        repeat,
        kernel_size,
        stride,
        expand,
        se_ratio,
        sum_layer,
        count_layer=None,
        pl=0.5
    ):
        super(MBConv, self).__init__()
        layer = []

        layer.append(
            InvertedResidual(inplanes, planes, kernel_size, stride, expand, se_ratio)
        )

        for l in range(1, repeat):
            # https://arxiv.org/pdf/1603.09382.pdf
            prob = 1.0 - (count_layer + l) / sum_layer * (1 - pl)
            layer.append(
                InvertedResidual(
                    planes, planes, kernel_size, 1, expand, se_ratio, prob=prob
                )
            )

        self.layer = nn.Sequential(*layer)

    def forward(self, x):
        out = self.layer(x)
        return out


class Upsample(nn.Module):
    def __init__(self, scale):
        super(Upsample, self).__init__()
        self.scale = scale

    def forward(self, x):
        return F.interpolate(
            x, scale_factor=self.scale, mode='bilinear', align_corners=False
        )


class Flatten(nn.Module):
    def __init(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class EfficientNet(nn.Module):
    """ Efficientnet implementation for single usage (arch='fullefficientnet')
    or for eco usage if arch='lite' and
    endpoint in dimension [112, 56, 28, 14, 7]
    returns endpoint of dim dimension
    if arch = 'full' returns also op of last pooling
    """

    def __init__(
        self,
        num_classes=1000,
        width_coef=1.,
        depth_coef=1.,
        scale=1.,
        dropout_ratio=0.2,
        pl=0.5,
        endpoint=None,
        arch='fullEfficientnet',
        eager=False
    ):

        super(EfficientNet, self).__init__()
        self.endpoint = endpoint
        self.arch = arch
        self.eager = eager
        self.last_endpoint_x = None

        # Efficientnet parameters
        channels = [32, 16, 24, 40, 80, 112, 192, 320, 1280]
        expands = [1, 6, 6, 6, 6, 6, 6]
        repeats = [1, 2, 2, 3, 3, 4, 1]
        strides = [1, 2, 2, 2, 1, 2, 1]
        kernel_sizes = [3, 3, 5, 3, 5, 5, 3]
        depth = depth_coef
        width = width_coef

        channels = [
            round(x * width) for x in channels
        ]  # [int(x*width) for x in channels]
        repeats = [round(x * depth) for x in repeats]  # [int(x*width) for x in repeats]

        # Tracker of depth for stochastic depth
        sum_layer = sum(repeats)

        # self.upsample = Upsample(scale)
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(
                channels[0],
                momentum=0.1,
                eps=1e-05,
                affine=True,
                track_running_stats=True
            )
        )
        se_ratio = 4
        self.stage2 = MBConv(
            channels[0],
            channels[1],
            repeats[0],
            kernel_size=kernel_sizes[0],
            stride=strides[0],
            expand=expands[0],
            se_ratio=se_ratio,
            sum_layer=sum_layer,
            count_layer=sum(repeats[:0]),
            pl=pl
        )
        if (endpoint is None or 'full' in arch or endpoint in [56, 28, 14, 7]):
            se_ratio = 24
            self.stage3 = MBConv(
                channels[1],
                channels[2],
                repeats[1],
                kernel_size=kernel_sizes[1],
                stride=strides[1],
                expand=expands[1],
                se_ratio=se_ratio,
                sum_layer=sum_layer,
                count_layer=sum(repeats[:1]),
                pl=pl
            )
        if (endpoint is None or 'full' in arch or endpoint in [28, 14, 7]):
            self.stage4 = MBConv(
                channels[2],
                channels[3],
                repeats[2],
                kernel_size=kernel_sizes[2],
                stride=strides[2],
                expand=expands[2],
                se_ratio=se_ratio,
                sum_layer=sum_layer,
                count_layer=sum(repeats[:2]),
                pl=pl
            )
        if (endpoint is None or 'full' in arch or endpoint in [14, 7]):
            self.stage5 = MBConv(
                channels[3],
                channels[4],
                repeats[3],
                kernel_size=kernel_sizes[3],
                stride=strides[3],
                expand=expands[3],
                se_ratio=se_ratio,
                sum_layer=sum_layer,
                count_layer=sum(repeats[:3]),
                pl=pl
            )
        if (endpoint is None or 'full' in arch or endpoint in [7]):
            self.stage6 = MBConv(
                channels[4],
                channels[5],
                repeats[4],
                kernel_size=kernel_sizes[4],
                stride=strides[4],
                expand=expands[4],
                se_ratio=se_ratio,
                sum_layer=sum_layer,
                count_layer=sum(repeats[:4]),
                pl=pl
            )
            self.stage7 = MBConv(
                channels[5],
                channels[6],
                repeats[5],
                kernel_size=kernel_sizes[5],
                stride=strides[5],
                expand=expands[5],
                se_ratio=se_ratio,
                sum_layer=sum_layer,
                count_layer=sum(repeats[:5]),
                pl=pl
            )

            self.stage8 = MBConv(
                channels[6],
                channels[7],
                repeats[6],
                kernel_size=kernel_sizes[6],
                stride=strides[6],
                expand=expands[6],
                se_ratio=se_ratio,
                sum_layer=sum_layer,
                count_layer=sum(repeats[:6]),
                pl=pl
            )
        if endpoint is None or 'full' in arch:
            self.stage9 = nn.Sequential(
                nn.Conv2d(channels[7], channels[8], kernel_size=1, bias=True),
                nn.BatchNorm2d(channels[8], momentum=0.1, eps=1e-5), Swish(),
                nn.AdaptiveAvgPool2d((1, 1)), Flatten()
            )
        if arch == 'fullEfficientnet':
            self.head = nn.Sequential(
                nn.Dropout(p=dropout_ratio), nn.Linear(channels[8], num_classes)
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def features(self, x):
        # x = self.upsample(x)  # 2, 3, 224, 224
        end_dict = {}
        x = swish(self.stage1(x))  # 2, 32, 112, 112
        x = swish(self.stage2(x))  # 2, 16, 112, 112
        if self.endpoint in [112] or self.arch == 'full':
            end_dict[112] = x
        x = swish(self.stage3(x))  # 2, 24, 56, 56
        if self.endpoint in [56] or self.arch == 'full':
            end_dict[56] = x
        x = swish(self.stage4(x))  # 2, 40, 28, 28
        if self.endpoint in [28] or self.arch == 'full':
            end_dict[28] = x
        x = swish(self.stage5(x))  # 2, 80, 14, 14
        if self.endpoint in [14] or self.arch == 'full':
            end_dict[14] = x
        x = swish(self.stage6(x))  # 2, 112, 7, 7
        x = swish(self.stage7(x))  # 2, 192, 7, 7
        x = swish(self.stage8(x))  # 2, 320, 7, 7
        if self.endpoint in [7] or self.arch == 'full':
            end_dict[7] = x
        x = swish(self.stage9(x))
        # x = swish(self.head(x))
        # print(x.shape)
        if self.endpoint is not None or 'full' not in self.arch:
            self.last_endpoint_x = end_dict[self.endpoint]
            return x
        return x

    def eager_features(self, x):
        x = swish(self.stage1(x))  # 2, 32, 112, 112
        x = swish(self.stage2(x))  # 2, 16, 112, 112
        if self.endpoint in [112] or self.arch == 'full':
            return x
        x = swish(self.stage3(x))  # 2, 24, 56, 56
        if self.endpoint in [56] or self.arch == 'full':
            return x
        x = swish(self.stage4(x))  # 2, 40, 28, 28
        if self.endpoint in [28] or self.arch == 'full':
            return x
        x = swish(self.stage5(x))  # 2, 80, 14, 14
        if self.endpoint in [14] or self.arch == 'full':
            return x
        x = swish(self.stage6(x))  # 2, 112, 7, 7
        if self.endpoint in [13] or self.arch == 'full':
            return x
        x = swish(self.stage7(x))  # 2, 192, 7, 7
        if self.endpoint in [12] or self.arch == 'full':
            return x
        x = swish(self.stage8(x))  # 2, 320, 7, 7
        if self.endpoint in [7] or self.arch == 'full':
            return x
        x = swish(self.stage9(x))
        return x

    def forward(self, input):
        if self.endpoint is not None and not self.eager:
            x = self.features(input)
            return x
        elif self.endpoint is not None and self.eager:
            x = self.eager_features(input)
            return x
        x = swish(self.stage1(input))  # 2, 32, 112, 112
        x = swish(self.stage2(x))  # 2, 16, 112, 112
        x = swish(self.stage3(x))  # 2, 24, 56, 56
        x = swish(self.stage4(x))  # 2, 40, 28, 28
        x = swish(self.stage5(x))  # 2, 80, 14, 14
        x = swish(self.stage6(x))  # 2, 112, 7, 7
        x = swish(self.stage7(x))  # 2, 192, 7, 7
        x = swish(self.stage8(x))  # 2, 320, 7, 7
        x = self.stage9(x)
        logit = self.head(x)  # 2, classes
        return logit


def efficientnet_b0(num_classes=1000, endpoint=None, arch='fullEfficientnet'):
    return EfficientNet(
        num_classes=num_classes,
        width_coef=1.0,
        depth_coef=1.0,
        scale=1.0,
        dropout_ratio=0.2,
        pl=0.2,
        endpoint=endpoint,
        arch=arch
    )


def efficientnet_b1(num_classes=1000, endpoint=None):
    return EfficientNet(
        num_classes=num_classes,
        width_coef=1.0,
        depth_coef=1.1,
        scale=240 / 224,
        dropout_ratio=0.2,
        pl=0.2,
        endpoint=endpoint
    )


def efficientnet_b2(num_classes=1000, endpoint=None):
    return EfficientNet(
        num_classes=num_classes,
        width_coef=1.1,
        depth_coef=1.2,
        scale=260 / 224.,
        dropout_ratio=0.3,
        pl=0.3,
        endpoint=endpoint
    )


def efficientnet_b3(num_classes=1000, endpoint=None):
    return EfficientNet(
        num_classes=num_classes,
        width_coef=1.2,
        depth_coef=1.4,
        scale=300 / 224,
        dropout_ratio=0.3,
        pl=0.3,
        endpoint=endpoint
    )


def efficientnet_b4(num_classes=1000, endpoint=None):
    return EfficientNet(
        num_classes=num_classes,
        width_coef=1.4,
        depth_coef=1.8,
        scale=380 / 224,
        dropout_ratio=0.4,
        pl=0.4,
        endpoint=endpoint
    )


def efficientnet_b5(num_classes=1000, endpoint=None):
    return EfficientNet(
        num_classes=num_classes,
        width_coef=1.6,
        depth_coef=2.2,
        scale=456 / 224,
        dropout_ratio=0.4,
        pl=0.4,
        endpoint=endpoint
    )


def efficientnet_b6(num_classes=1000, endpoint=None):
    return EfficientNet(
        num_classes=num_classes,
        width_coef=1.8,
        depth_coef=2.6,
        scale=528 / 224,
        dropout_ratio=0.5,
        pl=0.4,
        endpoint=endpoint
    )


def efficientnet_b7(num_classes=1000, endpoint=None):
    return EfficientNet(
        num_classes=num_classes,
        width_coef=2.0,
        depth_coef=3.1,
        scale=600 / 224,
        dropout_ratio=0.5,
        pl=0.5,
        endpoint=endpoint
    )


def test():
    x = torch.FloatTensor(64, 3, 224, 224).cuda()
    model = efficientnet_b0(num_classes=1000, endpoint=28, arch='full').cuda()
    from torchsummary import summary
    logit = model(x)
    print(model)
    print(summary(model, (3, 224, 224)))


if __name__ == '__main__':
    test()


def efficientnet_pretrained(num_classes=1000, pretrained='imagenet', eco_version="full"):
    r"""Efficientnet model architecture from
    <https://arxiv.org/pdf/1905.11946.pdf>`_
    paper and repo <https://github.com/katsura-jp/efficientnet-pytorch>.
    """

    if eco_version == "full":
        model = BNInception_full_pre(num_classes=num_classes)

    elif eco_version == "lite":
        model = BNInception_pre(num_classes=num_classes)

    if pretrained is not None:
        settings = pretrained_settings['bninception'][pretrained]
        assert num_classes == settings['num_classes'], \
            "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)
        model.load_state_dict(model_zoo.load_url(settings['url']))
        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']

    return model


def efficientnet():
    r"""Efficientnet model architecture from
    <https://arxiv.org/pdf/1905.11946.pdf>`_
    paper and repo <https://github.com/katsura-jp/efficientnet-pytorch>.
    """
    model = Efficientnet()
    return model
