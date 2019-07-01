import torch
import math
import torch.nn.init as init
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
from torch.nn.parameter import Parameter
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix


def init_model_pruning(model, prune_ratio, prune_start, prune_end, prune_min_nonzero):
    for module in model.modules():
        if isinstance(module, Compression):
            module.init_pruning_parameters(prune_ratio, prune_start, prune_end, prune_min_nonzero)


def prune_model(model, epoch):
    for module in model.modules():
        if isinstance(module, Compression):
            module.prune(epoch)


def quantize_model(model, bits):
    for module in model.modules():
        if isinstance(module, Compression):
            module.quantize(bits)


def reset_parameters(weight, bias):
    init.kaiming_uniform_(weight, a=math.sqrt(5))
    if bias is not None:
        fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in)
    init.uniform_(bias, -bound, bound)



class Compression(nn.Module):
    def __init__(self, weight_shape, prune_start=None, prune_end=None, prune_ratio=None, prune_min_nonzero=None):
        super().__init__()
        self.init_pruning_parameters(prune_start, prune_end, prune_ratio, prune_min_nonzero)
        self.weight = Parameter(torch.Tensor(*weight_shape))
        self.mask = Parameter(torch.ones(*weight_shape), requires_grad=False).cuda(non_blocking=True)

    def init_pruning_parameters(self, prune_start, prune_end, prune_ratio, prune_min_nonzero):
        self.prune_start = prune_start              # start epoch of pruning
        self.prune_end = prune_end                  # end epoch of pruning
        self.prune_ratio = prune_ratio              # ratio of nonzero elements to keep
        self.prune_min_nonzero = prune_min_nonzero  # minimum absolute number of nonzero elements to keep (overrides prune_ratio)

    def prune(self, epoch):
        print('prune')
        if None in [self.prune_start, self.prune_end, self.prune_ratio, self.prune_min_nonzero]:
            raise ValueError('pruning values not initialized')

        ratio_max = max(0, 1-self.prune_min_nonzero/self.weight.numel())
        ratio_max = min(ratio_max, self.prune_ratio)

        if epoch < self.prune_start:
            self.mask.data = torch.ones(self.mask.shape).cuda(non_blocking=True)
            return
        elif epoch >= self.prune_end:
            perc = ratio_max
        else:
            x = (epoch-self.prune_start) / (self.prune_end - self.prune_start)
            perc = ratio_max * (1+(x-1)**3)

        perc = perc*100

        self.weight.data = self.weight.data.cuda() * self.mask.data.cuda()

        weight_dev = self.weight.device
        mask_dev = self.mask.device
        weight_np = self.weight.data.cpu().numpy()
        mask_np = self.mask.data.cpu().numpy()

        perc_value = np.percentile(abs(weight_np), perc)
        mask_new_np = np.where(abs(weight_np) < perc_value, 0, mask_np)

        self.weight.data = torch.from_numpy(weight_np * mask_new_np).to(weight_dev)
        self.mask.data = torch.from_numpy(mask_new_np).to(mask_dev)

    def quantize(self, bits):
        weight_dev = self.weight.device
        weight_np = self.weight.data.cpu().numpy()
        shape = weight_np.shape
        csr = csr_matrix(weight_np.flatten())
        space = np.linspace(min(csr.data), max(csr.data), num=2**bits)
        kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1,1), n_init=1, precompute_distances=True, algorithm="full")
        kmeans.fit(csr.data.reshape(-1,1))
        new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
        self.weight.data = torch.from_numpy(new_weight.reshape(shape)).to(weight_dev)

    def get_compression_ratio(self):
        print(torch.sum(self.mask))
        return (torch.numel(self.mask) - torch.sum(self.mask)) / torch.numel(self.mask)


class _ConvNdP(Compression):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        super().__init__(weight_shape=[out_channels, in_channels // groups, *kernel_size])

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        reset_parameters(self.weight, self.bias)



class Conv1dP(_ConvNdP):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):

        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                         padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        return f.conv1d(input=x, weight=self.weight * self.mask, bias=self.bias, stride=self.stride,
                        padding=self.padding, dilation=self.dilation, groups=self.groups)



class Conv2dP(_ConvNdP):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        elif len(kernel_size) == 2:
            kernel_size = (kernel_size[0], kernel_size[1])
        else:
            raise ValueError('wrong kernel size')

        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                         padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        return f.conv2d(input=x, weight=self.weight * self.mask, bias=self.bias, stride=self.stride,
                        padding=self.padding, dilation=self.dilation, groups=self.groups)



class Conv3dP(_ConvNdP):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):

        if isinstance(kernel_size, int):
            kernel_size = (in_channels, out_channels, kernel_size, kernel_size)
        elif len(kernel_size) == 3:
            kernel_size = (in_channels, out_channels, kernel_size[0], kernel_size[1], kernel_size[2])
        else:
            raise ValueError('wrong kernel size')

        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                         padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        return f.conv3d(input=x, weight=self.weight * self.mask, bias=self.bias, stride=self.stride,
                        padding=self.padding, dilation=self.dilation, groups=self.groups)



class LinearP(Compression):
    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features

        super().__init__(weight_shape=[out_features, in_features])

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        reset_parameters(self.weight, self.bias)


    def forward(self, x):
        return f.linear(input=x, weight=self.weight * self.mask, bias=self.bias)


