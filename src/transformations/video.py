from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm
from torchvision import transforms
from utils import LOGGER, AugmentNet, MonkeyNet
import pretrainedmodels.utils as putils
from PIL import Image

RESIZE_FACTOR = 1.3


# ########################################################
# Transformations and augmentation prepending stacks
# ########################################################

def gpu_resize(model, dataset, transf_args):
    LOGGER.info('Using ###  GPU Resize   ### for transformations')
    use_gpu_resize = (transf_args['use_gpu_resize'] == 'True')
    resize_factor = float(transf_args['resize_factor'])
    flip_factor = float(transf_args['flip_factor'])

    cpu_resize = (
        np.any(dataset.min_shape[1:] != dataset.max_shape[1:]) or not use_gpu_resize
    )
    print('CPU resize: ' + str(cpu_resize))

    out_size = model.input_size
    out_size = np.array((out_size, out_size), dtype=np.int)
    re_size = np.ceil(out_size * resize_factor).astype(np.int)

    transf_dict = {
        'train':
            {
                'samples':
                    transforms.Compose(
                        [
                            CPUDynamicSelectSegmentsUniform(model),
                            CPUFormatAxes(3),
                            CPUCrop(1.3),
                            CPURandomHFlip(0.4),
                            CPUCadeneTransform(model)
                        ]
                    ),
                'labels':
                    transforms.
                    Lambda(lambda x: x if dataset.is_multilabel else np.argmax(x))
            },
        'test':
            {
                'samples':
                    transforms.Compose(
                        [
                            CPUDynamicSelectSegmentsUniform(model),
                            CPUFormatAxes(3),
                            CPUCadeneTransform(model)
                        ]
                    ),
                'labels':
                    transforms.
                    Lambda(lambda x: x if dataset.is_multilabel else np.argmax(x))
            }
    }
    aug_net = AugmentNet(
        {
            'train':
                [
                ],
            'test':
                [
                ]
        }
    )
    model = MonkeyNet(OrderedDict([
        ('aug_net', aug_net),
        ('main_net', model),
    ]))

    return model, transf_dict


# ########################################################
# Helpers
# ########################################################
class CPUFormatAxes(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

    def forward(self, x):
        x = x[0]

        if x.shape[2] == 1:
            x = np.tile(x, (1, 1, self.channels))
        elif x.shape[2] > self.channels:
            x = x[:,:,0:self.channels]
        return x


class CPUCrop(nn.Module):
    def __init__(self, crop_factor):
        super().__init__()
        self.crop_factor = crop_factor

    def forward(self, x):
        crop_factor = (self.crop_factor-1)*np.random.random() + 1
        img_width = x.shape[0] / crop_factor
        img_height = x.shape[1] / crop_factor
        off_width = (x.shape[0]-img_width) * np.random.random()
        off_height = (x.shape[1] - img_height) * np.random.random()

        x = x[int(off_width):int(off_width+img_width), int(off_height):int(off_height+img_height),:]
        return x


class CPUCadeneTransform(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.convert_to_pil = transforms.ToPILImage(mode = 'RGB')
        self.transf = putils.TransformImage(model)

    def forward(self, x):
        x = self.convert_to_pil(x)
        return self.transf(x)


class CPUFormatLabel(nn.Module):
    def __init__(self, multilabel):
        super().__init__()
        self.is_multilabel = multilabel

    def __call__(self, x: torch.Tensor):
        x = x if self.is_multilabel else np.argmax(x, axis=0)
        return x

    def forward(self, x):
        return self(x)


class CPUDynamicSelectSegmentsUniform(nn.Module):
    def __init__(self, model, random=True):
        super().__init__()
        self.master_model = model

    def __call__(self, x: torch.Tensor):
        choices = []
        num_segments = 1#self.master_model.num_segments
        if x.shape[0] <= num_segments:
            choices = np.linspace(0, x.shape[0] - 1, num_segments, dtype=int)
        else:
            choices = np.linspace(0, x.shape[0] - 1, x.shape[0], dtype=int)
            choices = np.array_split(choices, num_segments)
            choices = [np.random.choice(c, 1)[0].tolist() for c in choices]
        x = x[choices]
        return x


class CPUDynamicSelectSegmentsNormal(nn.Module):
    def __init__(self, model, random=True):
        super().__init__()
        self.master_model = model

    def __call__(self, x: torch.Tensor):
        num_segments = 1#self.master_model.num_segments
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


class CPUSelectSegments(nn.Module):
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


class CPURandomHFlip(nn.Module):
    def __init__(self, p):
        self.p = p

    def __call__(self, x):
        x = np.array(x if np.random.uniform() > self.p else np.flip(x, axis=2))
        return x
