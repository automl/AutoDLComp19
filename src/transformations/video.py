from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from PIL import ImageOps
from scipy.stats import norm
from torchvision import transforms
from utils import LOGGER, AugmentNet, MonkeyNet

RESIZE_FACTOR = 1.3


# ########################################################
# Transformations and augmentation prepending stacks
# ########################################################
def baseline_transforms(model, dataset):
    LOGGER.info('Using ###   BaselineAugmentNet   ### for transformationstack')

    # Classical transformations performed per sample
    transf_dict = {
        'train':
            {
                'samples':
                    transforms.Compose(
                        [
                            CPU.DynamicSelectSegmentsUniform(model),
                            CPU.RandomCropPadOrg(model.input_size)
                        ]
                    ),
                'labels':
                    CPU.FormatLabel(dataset.is_multilabel)
            },
        'test':
            {
                'samples':
                    transforms.Compose(
                        [
                            CPU.DynamicSelectSegmentsUniform(model),
                            CPU.RandomCropPadOrg(model.input_size)
                        ]
                    ),
                'labels':
                    CPU.FormatLabel(dataset.is_multilabel)
            }
    }

    # Prepend an augmentation network performing transformations on the gpu
    # on a whole batch
    aug_net = AugmentNet(
        {
            'train': [
                GPU.SwapAxes(),
                GPU.FormatChannels(3),
                GPU.Stack(),
                GPU.Normalize()
            ],
            'test': [GPU.SwapAxes(),
                     GPU.FormatChannels(3),
                     GPU.Stack(),
                     GPU.Normalize()]
        }
    )
    # To expose the original model's attributes use the MonkeyNet(nn.Sequential)
    model = MonkeyNet(OrderedDict([
        ('aug_net', aug_net),
        ('main_net', model),
    ]))
    return model, transf_dict


def normal_segment_dist(model, dataset):
    LOGGER.info('Using ###   normal_segment_dist   ### for transformationstack')
    transf_dict = {
        'train':
            {
                'samples':
                    transforms.Compose(
                        [
                            CPU.DynamicSelectSegmentsNormal(model),
                            CPU.RandomCropPad(model.input_size),
                            CPU.RandomHFlip(0.4)
                        ]
                    ),
                'labels':
                    CPU.FormatLabel(dataset.is_multilabel)
            },
        'test':
            {
                'samples':
                    transforms.Compose(
                        [
                            CPU.DynamicSelectSegmentsNormal(model),
                            CPU.RandomCropPad(model.input_size)
                        ]
                    ),
                'labels':
                    CPU.FormatLabel(dataset.is_multilabel)
            }
    }
    aug_net = AugmentNet(
        {
            'train': [
                GPU.SwapAxes(),
                GPU.FormatChannels(3),
                GPU.Stack(),
                GPU.Normalize()
            ],
            'test': [GPU.SwapAxes(),
                     GPU.FormatChannels(3),
                     GPU.Stack(),
                     GPU.Normalize()]
        }
    )
    model = MonkeyNet(OrderedDict([
        ('aug_net', aug_net),
        ('main_net', model),
    ]))
    return model, transf_dict


def resize_normal_seg_selection(model, dataset, crop_size):
    LOGGER.info('Using ###   resize_normal_seg_selection   ### for transformationstack')
    transf_dict = {
        'train':
            {
                'samples':
                    transforms.Compose(
                        [
                            CPU.DynamicSelectSegmentsNormal(model),
                            CPU.CropResizeImage(model.input_size, crop_size),
                            CPU.PILToNumpy(),
                        ]
                    ),
                'labels':
                    CPU.FormatLabel(dataset.is_multilabel)
            },
        'test':
            {
                'samples':
                    transforms.Compose(
                        [
                            CPU.DynamicSelectSegmentsNormal(model),
                            CPU.CropResizeImage(model.input_size),
                            CPU.PILToNumpy(),
                        ]
                    ),
                'labels':
                    CPU.FormatLabel(dataset.is_multilabel)
            }
    }
    aug_net = AugmentNet(
        {
            'train': [GPU.Normalize(),
                      GPU.FormatChannels(3),
                      GPU.Stack()],
            'test': [GPU.Normalize(), GPU.FormatChannels(3),
                     GPU.Stack()]
        }
    )
    model = MonkeyNet(OrderedDict([
        ('aug_net', aug_net),
        ('main_net', model),
    ]))
    return model, transf_dict


# ########################################################
# TRANSFORMS
# ########################################################
class GPU():
    class SwapAxes(nn.Module):
        '''
        Swap axes for interpolation
        '''
        def __init__(self):
            super().__init__()
            pass

        def forward(self, x: torch.Tensor):
            x = x.permute(0, 1, 4, 2, 3)
            return x

    class FormatChannels(nn.Module):
        '''
        Adapt number of channels. If there are more than desired, use only the first n channels.
        If there are less, copy existing channels
        '''
        def __init__(self, channels_des):
            super().__init__()
            self.channels_des = channels_des

        def forward(self, x: torch.Tensor):
            channels = x.shape[2]
            if channels < self.channels_des:
                shape = x.shape
                x = x.expand(
                    *shape[:2], int(np.ceil(self.channels_des / channels)), *shape[3:]
                )
            x = x[:, :, 0:self.channels_des, :, :].contiguous()
            return x

    class Stack(nn.Module):
        '''
        Concatenate subsequent images of one video by stacking the channels
        '''
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor):
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

        def forward(self, x: torch.Tensor):
            min_val = torch.min(x).cpu().numpy()
            if min_val < -0.01:
                x = x + min_val
            max_val = torch.max(x).cpu().numpy()
            if max_val <= 255 and max_val > 1:
                x = x / 255
            elif max_val > 255:
                x = x / max_val
                LOGGER.warning('Weird max_val: ' + str(max_val))
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


class CPU():
    class FormatLabel(nn.Module):
        def __init__(self, multilabel):
            super().__init__()
            self.is_multilabel = multilabel

        def __call__(self, x: torch.Tensor):
            x = x if self.is_multilabel else np.argmax(x)
            return x

    class SwapAxes(nn.Module):
        '''
        Swap axes for interpolation
        '''
        def __init__(self):
            super().__init__()
            pass

        def forward(self, x):
            x = np.moveaxis(x, -1, 1)
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
            channels = x.shape[1]
            if channels < self.channels_des:
                shape = x.shape
                x = x.expand(
                    *shape[:1], int(np.ceil(self.channels_des / channels)), *shape[2:]
                )
            x = x[:, 0:self.channels_des, :, :]

            return x

    class CombineSegments(nn.Module):
        '''
        Concatenate subsequent images of one video by stacking the channels
        '''
        def __init__(self):
            super().__init__()

        def forward(self, x):
            x = x.reshape(-1, *x.shape[2:])
            return x

    class Normalize(nn.Module):
        '''
        Normalize from the 0-255 range to the 0-1 range.
        '''
        def __init__(self):
            super().__init__()

        def forward(self, x):
            if x.min() >= 0 and x.max() > 1 and x.max() < 255:
                x = x / 255
            else:
                x = (x + x.min()) / x.max()
            return x

    class SelectSegments(nn.Module):
        def __init__(self, num_segments, random=True):
            super().__init__()
            self.num_segments = num_segments

        def __call__(self, x: torch.Tensor):
            choices = []
            if x.shape[0] <= self.num_segments:
                choices = np.linspace(0, x.shape[0] - 1, self.num_segments, dtype=int)
            else:
                choices = np.linspace(0, x.shape[0] - 1, x.shape[0], dtype=int)
                choices = np.array_split(choices, self.num_segments)
                choices = np.array([np.random.choice(c, 1) for c in choices]).squeeze()
            x = x[choices.tolist()]
            return x

    class DynamicSelectSegmentsUniform(nn.Module):
        def __init__(self, model, random=True):
            super().__init__()
            self.master_model = model

        def __call__(self, x: torch.Tensor):
            choices = []
            num_segments = self.master_model.num_segments
            if x.shape[0] <= num_segments:
                choices = np.linspace(0, x.shape[0] - 1, num_segments, dtype=int)
            else:
                choices = np.linspace(0, x.shape[0] - 1, x.shape[0], dtype=int)
                choices = np.array_split(choices, num_segments)
                choices = [np.random.choice(c, 1)[0].tolist() for c in choices]
            x = x[choices]
            return x

    class DynamicSelectSegmentsNormal(nn.Module):
        def __init__(self, model, random=True):
            super().__init__()
            self.master_model = model

        def __call__(self, x):
            num_segments = self.master_model.num_segments
            frame_count = x.shape[0]

            idxs = np.linspace(0, frame_count - 1, frame_count, dtype=int)
            if frame_count <= num_segments * 2:
                idxs = np.repeat(idxs, frame_count * num_segments / len(idxs))
                frame_count *= num_segments
            seg_sizes = norm.pdf(np.linspace(-1, 1, num_segments))
            seg_sizes = 1 - seg_sizes if frame_count > num_segments else seg_sizes
            seg_sizes = (seg_sizes / seg_sizes.sum()) * frame_count
            seg_sizes = seg_sizes.astype(int)

            choices = []
            last_idx = 0
            for i, seg_size in enumerate(seg_sizes):
                next_idx = last_idx + seg_size if i < len(seg_sizes) - 1 else None
                choices.append(np.random.choice(idxs[last_idx:next_idx], 1)[0].tolist())
                last_idx = next_idx
            x = x[choices]
            return x

    class CropResizeImage(nn.Module):
        '''
        Crop to fit the target aspect ratio and then resize to the
        target size. Additionally randomly crop out a part of size (in percent)
        random_crop from the resulting image and resize it to target size.

        RandomCrop:
        The croping region is the same for all frames given
        '''
        def __init__(self, target_im_size, random_crop=1.):
            super().__init__()
            self.target_im_size = target_im_size
            self.convert_to_pil = transforms.ToPILImage()
            self.random_crop = random_crop

        def __call__(self, x):
            '''
            Expects SxHxWxC
            '''
            new_sizes = 2 * (self.target_im_size, )
            center = [0.5, 0.5]
            xmin, xmax = (x.min(), x.max())
            if xmin < 0.:
                x += xmin
            if xmax > 1.:
                x /= xmax if xmax > 255 else 255
            x = [self.convert_to_pil((e * 255).astype(np.uint8)) for e in x[:]]
            x = [ImageOps.fit(e, new_sizes, centering=center) for e in x]
            if self.random_crop > 0. and self.random_crop < 1.:
                crop_size = int(self.target_im_size * self.random_crop)
                offset = new_sizes[0] - crop_size - 1
                offset = np.random.randint(0, offset, 4)
                offset[2:] = offset[:2] + crop_size
                x = [e.crop(offset).resize(new_sizes) for e in x]
            return x

    class PILToNumpy(nn.Module):
        def __call__(self, x):
            '''
            Returns SxCxHxW unnormalize (ie. [0-255])
            '''
            x = np.array([np.array(e) for e in x])
            if len(x.shape) < 4:
                # Only have a single frame, ie. it's an image
                x = np.expand_dims(x, -1)
            x = np.moveaxis(x, -1, 1)
            return x

    class RandomCropPadOrg(object):
        def __init__(self, size_des):
            self.size_des = size_des

        def __call__(self, pics):
            row = pics.shape[1]
            row_des = self.size_des
            col = pics.shape[2]
            col_des = self.size_des

            row_rand = -1
            col_rand = -1

            if row <= row_des:  # pad rows
                row_pad_start = int(np.floor((row_des - row) / 2))
                row_pad_end = row + int(np.floor((row_des - row) / 2))
                row_start = 0
                row_end = row
            else:  # crop rows
                row_rand = int(np.random.random() * int(np.floor((row - row_des))))
                row_pad_start = 0
                row_pad_end = row_des
                row_start = row_rand
                row_end = row_des + row_rand
            if col <= col_des:  # pad columns
                col_pad_start = int(np.floor((col_des - col) / 2))
                col_pad_end = col + int(np.floor((col_des - col) / 2))
                col_start = 0
                col_end = col
            else:  # crop columns
                col_rand = int(np.random.random() * int(np.floor((col - col_des))))
                col_pad_start = 0
                col_pad_end = col_des
                col_start = col_rand
                col_end = col_des + col_rand

            pics_pad = np.zeros(
                (pics.shape[0], row_des, col_des, pics.shape[3]), dtype=pics.dtype
            )
            pics_pad[:, row_pad_start:row_pad_end, col_pad_start:col_pad_end, :
                    ] = pics[:, row_start:row_end, col_start:col_end, :]
            return pics_pad

    class RandomCropPad(nn.Module):
        def __init__(self, target_im_size):
            super().__init__()
            self.target_im_size = (
                (target_im_size,
                 target_im_size) if isinstance(target_im_size, int) else target_im_size
            )

        def __call__(self, x):
            # expects FxWxHxC Format
            im_size = np.array(x.shape[1:-1])
            wiggle_room = im_size - self.target_im_size

            wi = int(
                np.random.choice(
                    np.arange(wiggle_room[0], 0, -np.sign(wiggle_room[0]), dtype=np.int),
                    1
                )
            ) if wiggle_room[0] != 0 else 0
            hi = int(
                np.random.choice(
                    np.arange(wiggle_room[1], 0, -np.sign(wiggle_room[1]), dtype=np.int),
                    1
                )
            ) if wiggle_room[1] != 0 else 0

            wis, wie = (wi, wi + self.target_im_size[0])
            nwis, nwie = (0, self.target_im_size[0])
            if wi < 0:
                wis, wie = (0, im_size[0])
                nwis, nwie = (-im_size[0] + wi, wi)

            his, hie = (hi, hi + self.target_im_size[1])
            nhis, nhie = (0, self.target_im_size[1])
            if hi < 0:
                his, hie = (0, im_size[1])
                nhis, nhie = (-im_size[1] + hi, hi)

            new_x = np.zeros(
                (x.shape[0], *self.target_im_size, x.shape[-1]), dtype=x.dtype
            )
            new_x[:, nwis:nwie, nhis:nhie, :] = x[:, wis:wie, his:hie, :]
            return new_x

    # AUGMENTATIONS
    class RandomHFlip(nn.Module):
        def __init__(self, p):
            self.p = p

        def __call__(self, x):
            x = np.array(x if np.random.uniform() > self.p else np.flip(x, axis=2))
            return x
