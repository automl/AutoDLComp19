import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from utils import LOGGER

from torchhome.hub.autodlcomp_models_master.video.transforms import SelectSamples, RandomCropPad


# Set the device which torch should use
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ########################################################
# Perform only necessary transformations on the cpu
# ########################################################
def default_transformations_selector(dataset, model, resize):
    transf_dict = {
        'train': {
            'samples': transforms.Compose(
                [
                    SelectSamples(model.num_segments),
                    *((
                        (RandomCropPad(model.input_size), CPUFormatImage(False))
                        if resize
                        else
                        (CPUFormatImage(True), CPUResizeImage(model.input_size))
                    )),
                ]
            ),
            'labels': transforms.Lambda(
                lambda x: x if dataset.is_multilabel else np.argmax(x)
            )
        },
        'test': {
            'samples': transforms.Compose(
                [
                    SelectSamples(model.num_segments),
                    *((
                        (RandomCropPad(model.input_size), CPUFormatImage(False))
                        if resize
                        else
                        (CPUFormatImage(True), CPUResizeImage(model.input_size))
                    )),
                ]
            ),
            'labels': transforms.Lambda(
                lambda x: x if dataset.is_multilabel else np.argmax(x)
            )
        }
    }
    return transf_dict


# ########################################################
# Perform transformations on the cpu
# ########################################################
def cpu_transforms(dataset, model, resize):
    LOGGER.info('Using cpu transformations')
    base_transform = transforms.Compose([
        CPUSelectSegments(model.num_segments),
        CPUFormatImage(resize),
        (
            CPUResizeImage(model.input_size)
            if resize
            else CPURandomCropPad(model.input_size)
        ),
        CPUNormImage(),
        CPUStackSegments()
    ])
    label_transform = transforms.Compose([
        CPUFormatLabel(dataset.is_multilabel)
    ])

    transf_dict = {
        'train': {
            'samples': transforms.Compose([
                base_transform,
            ]),
            'labels': transforms.Compose([
                label_transform,
            ]),
        },
        'test': {
            'samples': transforms.Compose([
                base_transform,
            ]),
            'labels': transforms.Compose([
                label_transform,
            ]),
        }
    }
    return transf_dict


# ########################################################
# Perform transformations on the gpu and leave it there
# This is slow compared to cpu transformations
# ########################################################
def gpu_transforms(dataset, model, resize):
    LOGGER.info('Using gpu transformations')
    base_transform = transforms.Compose([
        GPUMoveToGPU(),
        GPUSelectSegments(model.num_segments),
        GPUFormatImage(),
        (
            GPUResizeImage(model.input_size, 'bicubic')
            if resize
            else GPURandomCropPad(model.input_size)
        ),
        GPUNormImage(),
        GPUStackSegments()
    ])
    label_transform = transforms.Compose([
        GPUFormatLabel(dataset.is_multilabel)
    ])

    transf_dict = {
        'train': {
            'samples': transforms.Compose([
                base_transform,
            ]),
            'labels': transforms.Compose([
                label_transform,
            ]),
            # 'labels': transforms.Compose([
            #     label_transform,
            # ]),
        },
        'test': {
            'samples': transforms.Compose([
                base_transform,
            ]),
            'labels': transforms.Compose([
                label_transform,
            ]),
        }
    }
    return transf_dict


# ########################################################
# Helpers
# ########################################################
class CPUFormatLabel(nn.Module):
    def __init__(self, multilabel):
        self.is_multilabel = multilabel

    def __call__(self, x: torch.Tensor):
        x = x if self.is_multilabel else np.argmax(x, axis=0)
        return x

    def forward(self, x):
        return self(x)


class CPUSelectSegments(nn.Module):
    def __init__(self, num_segments, random=True):
        self.num_segments = num_segments
        self.random = random

    def __call__(self, x: torch.Tensor):
        seg_idx = list(range(x.shape[0]))
        self.num_segments = (
            x.shape[0]
            if x.shape[0] <= self.num_segments
            else self.num_segments
        )
        choices = np.random.choice(
            seg_idx,
            self.num_segments,
            replace=False,
        )
        choices.sort()
        x = x[choices.tolist(), :, :, :]
        return x


class CPUFormatImage(nn.Module):
    def __init__(self, to_pil_image):
        self.to_pil_image = to_pil_image
        self.convert_to_pil = transforms.ToPILImage()

    def __call__(self, x):
        if self.to_pil_image:
            x = [self.convert_to_pil(e) for e in x[:]]
        else:
            x = np.moveaxis(x, 3, 1)
        return x


class CPUNormImage(nn.Module):
    def __call__(self, x):
        x = x / 255 if x.max() > 1. else x
        if x.shape[1] < 3:
            x = np.repeat(x, 3, 1)
        return x


class CPUResizeImage(nn.Module):
    def __init__(self, target_im_size):
        self.target_im_size = (
            (target_im_size, target_im_size)
            if isinstance(target_im_size, int)
            else target_im_size
        )

    def __call__(self, x):
        x = [e.resize(self.target_im_size) for e in x]
        x = np.array([np.array(e) for e in x])
        if len(x.shape) < 4:
            x = np.expand_dims(x, 1)
        return x


class CPURandomCropPad(nn.Module):
    def __init__(self, target_im_size):
        self.target_im_size = (
            (target_im_size, target_im_size)
            if isinstance(target_im_size, int)
            else target_im_size
        )

    def __call__(self, x):
        im_size = np.array(x.shape[-2:])
        wiggle_room = im_size - self.target_im_size

        wi = int(np.random.choice(np.arange(
            wiggle_room[0],
            0,
            -np.sign(wiggle_room[0])
        ), 1))
        hi = int(np.random.choice(np.arange(
            wiggle_room[1],
            0,
            -np.sign(wiggle_room[1])
        ), 1))

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

        new_x = np.zeros((*x.shape[:-2], *self.target_im_size))
        new_x[:, :, nwis:nwie, nhis:nhie] = x[:, :, wis:wie, his:hie]

        return new_x


class CPUStackSegments(nn.Module):
    def __call__(self, x):
        x = x.reshape((-1, *x.shape[2:]))
        return x


# ########################################################
# GPU
# ########################################################
class GPUFormatLabel(nn.Module):
    def __init__(self, multilabel):
        self.is_multilabel = multilabel

    def __call__(self, x: torch.Tensor):
        x = (
            torch.tensor(x, dtype=torch.long)
            if self.is_multilabel
            else torch.tensor(
                np.argmax(x, axis=0),
                dtype=torch.long
            )
        )
        return x


class GPUSelectSegments(nn.Module):
    def __init__(self, num_segments, random=True):
        self.num_segments = num_segments
        self.random = random

    def __call__(self, x: torch.Tensor):
        seg_idx = list(range(x.shape[0]))
        self.num_segments = (
            x.shape[0]
            if x.shape[0] <= self.num_segments
            else self.num_segments
        )
        choices = np.random.choice(
            seg_idx,
            self.num_segments,
            replace=False,
        )
        choices.sort()
        x = x[choices.tolist(), :, :, :]
        return x.contiguous()


class GPUMoveToGPU(nn.Module):
    def __call__(self, x):
        x = torch.Tensor(x).to(DEVICE)
        return x


class GPUFormatImage(nn.Module):
    def __call__(self, x):
        # Input is assumed to be SCHW
        x = x.transpose_(1, 3).transpose_(2, 3)
        if x.shape[1] == 4:
            # Images are cmyk so we need to convert to RGB
            # R = 1-C * 1-K
            # G = 1-M * 1-K
            # B = 1-Y * 1-K
            # and drop the black channel
            x[:, 0, :, :] = (1 - x[:, 0, :, :]) * (1 - x[:, 3, :, :])
            x[:, 1, :, :] = (1 - x[:, 1, :, :]) * (1 - x[:, 3, :, :])
            x[:, 2, :, :] = (1 - x[:, 2, :, :]) * (1 - x[:, 3, :, :])
            x = x[:, 0:2, :, :]
        if x.shape[1] == 1:
            # We got a grayscale image
            x = x.expand(-1, 3, -1, -1)
        return x.contiguous()


class GPUNormImage(nn.Module):
    def __call__(self, x: torch.Tensor):
        return x / 255 if x.max() > 1. else x


class GPUResizeImage(nn.Module):
    def __init__(self, target_im_size, mode):
        self.target_im_size = (
            (target_im_size, target_im_size)
            if isinstance(target_im_size, int)
            else target_im_size
        )
        self.mode = mode

    def __call__(self, x: torch.Tensor):
        squeeze = False
        if len(x.shape) < 4:
            x = x.unsqueeze(0)
            squeeze = True
        x = nn.functional.interpolate(
            x,
            self.target_im_size,
            mode=self.mode,
            align_corners=False
        )
        return x.squeeze() if squeeze else x


class GPURandomCropPad(nn.Module):
    def __init__(self, target_im_size):
        self.target_im_size = (
            (target_im_size, target_im_size)
            if isinstance(target_im_size, int)
            else target_im_size
        )

    def __call__(self, x: torch.Tensor):
        im_size = np.array(x.shape[-2:])
        wiggle_room = im_size - self.target_im_size
        new_x = torch.zeros((*x.shape[:-2], *self.target_im_size))

        wi = int(np.random.choice(np.arange(
            wiggle_room[0],
            0,
            -np.sign(wiggle_room[0])
        ), 1))
        hi = int(np.random.choice(np.arange(
            wiggle_room[1],
            0,
            -np.sign(wiggle_room[1])
        ), 1))

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

        new_x[:, :, nwis:nwie, nhis:nhie] = x[:, :, wis:wie, his:hie]

        return new_x.contiguous()


class GPUStackSegments(nn.Module):
    def __call__(self, x: torch.Tensor):
        x = x.view(-1, *x.shape[2:])
        return x.contiguous()


class GPUToNumpy(nn.Module):
    def __call__(self, x: torch.Tensor):
        x = x.cpu().numpy()
        return x
