import numpy as np
import torch
import torch.nn as nn
# from utils import LOGGER

RESIZE_FACTOR = 1.3


class WrapperNet(nn.Module):
    def __init__(self, out_size: int, fast_augment: bool):
        super().__init__()
        if not isinstance(out_size, int):
            raise ValueError
        self.fast_augment = fast_augment
        self.out_size = np.array((out_size, out_size), dtype=np.int)
        self.re_size = np.ceil(self.out_size * RESIZE_FACTOR).astype(np.int)

        self.augmentation = {
            'train': (
                nn.Sequential(
                    SwapAxes(),
                    FormatChannels(3),
                    Interpolate(self.re_size.tolist()),
                    RandomCrop(self.out_size.tolist()),
                    Stack(),
                    Normalize()
                ) if self.fast_augment
                else nn.Sequential(
                    SwapAxes(),
                    FormatChannels(3),
                    RandomCrop(self.out_size.tolist()),
                    Stack(),
                    Normalize()
                )
            ),
            'eval': (
                nn.Sequential(
                    SwapAxes(),
                    FormatChannels(3),
                    Interpolate(self.out_size.tolist()),
                    Stack(),
                    Normalize()
                ) if self.fast_augment
                else nn.Sequential(
                    SwapAxes(),
                    FormatChannels(3),
                    Stack(),
                    Normalize()
                )
            )
        }

    def forward(self, x):
        mode = 'train' if self.training else 'eval'
        with torch.no_grad():
            x = self.augmentation[mode](x)
        return x


class FormatChannels(nn.Module):
    '''
    Adapt number of channels. If there are more than desired, use only the first n channels.
    If there are less, copy existing channels
    '''
    def __init__(self, channels_des):
        super().__init__()
        self.channels_des = channels_des

    def forward(self, x):
        channels = x.shape[2]
        if channels < self.channels_des:
            shape = x.shape
            x = x.expand(*shape[:2], int(np.ceil(self.channels_des / channels)), *shape[3:])
        x = x[:, :, 0:self.channels_des, :, :]

        return x


class SwapAxes(nn.Module):
    '''
    Swap axes for interpolation
    '''
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        x = x.permute(0, 1, 4, 2, 3)
        return x


class Interpolate(nn.Module):
    '''
    Resize image to desired size
    '''
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.interp = nn.functional.interpolate

    def forward(self, x):
        # first squeeze first two dimensions to make it suitable for interpolation
        shape = x.shape
        x = x.view(-1, *shape[2:])
        # do interpolation
        x = self.interp(x, size=self.size, mode='nearest')
        # and unsqueeeze dimensions again
        x = x.view(*shape[0:3], *self.size)
        return x


class RandomCrop(nn.Module):
    '''
    Randomly crop a selection of the image
    '''
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        # input shape
        row_in = x.shape[-2]
        col_in = x.shape[-1]
        # output shape
        row_out = self.size[0]
        col_out = self.size[1]
        # random start index for crop
        row_start = int(np.random.random() * (row_in - row_out))
        col_start = int(np.random.random() * (col_in - col_out))
        # resulting end index for crop
        row_end = int(row_start + row_out)
        col_end = int(col_start + col_out)
        # row_index = torch.tensor([row_start, row_end])
        # col_index = torch.tensor([col_start, col_end])
        x = x[:, :, :, row_start:row_end, col_start:col_end]
        return x


class Stack(nn.Module):
    '''
    Concatenate subsequent images of one video by stacking the channels
    '''
    def __init__(self):
        super().__init__()

    def forward(self, x):
        shape = x.shape
        shape_new = [x.shape[0], x.shape[1] * x.shape[2], *shape[3:]]
        x = x.view(*shape_new)
        return x


class Normalize(nn.Module):
    '''
    Normalize from the 0-255 range to the 0-1 range.
    '''
    def __init__(self):
        super().__init__()

    def forward(self, x):
        min_val = torch.min(x).cpu().numpy()

        if min_val < 0:
            x = x + min_val

        max_val = torch.max(x).cpu().numpy()

        if max_val <= 255 and max_val > 1:
            x = x / 255
        elif max_val > 255:
            x = x / max_val
            print('Weird max_val: ' + str(max_val))

        return x
