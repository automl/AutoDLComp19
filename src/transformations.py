import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from wrapper_net import WrapperNet

from utils import LOGGER

from torchhome.hub.autodlcomp_models_master.video.transforms import SelectSamples, RandomCropPad


# Set the device which torch should use
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ########################################################
# Perform only necessary transformations on the cpu
# ########################################################
def aug_net(autodl_model, dataset):
    model = autodl_model.model
    LOGGER.info('Using ###   aug_net   ### for transformations')

    # Inject the WrapperNet at the top of the modules list
    need_to_resize = np.any(dataset.min_shape[1:] != dataset.max_shape[1:])
    aug_net = WrapperNet(model.input_size, not need_to_resize)
    # Monkeypatch the new network to expose the original model's attributes
    # autodl_model.model = MonkeyNet(
    #     aug_net,
    #     model
    # )
    autodl_model.model = MonkeyNet(
        aug_net,
        model
    )
    transf_dict = {
        'train': {
            'samples': transforms.Compose(
                [
                    CPUSelectSegmentsDynamic(autodl_model.model),
                    *(
                        (CPUResizeImage(aug_net.re_size.tolist()), )
                        if need_to_resize else ()
                    )
                ]
            ),
            'labels': transforms.Lambda(
                lambda x: x if dataset.is_multilabel else np.argmax(x)
            )
        },
        'test': {
            'samples': transforms.Compose(
                [
                    CPUSelectSegmentsDynamic(autodl_model.model),
                    *(
                        (CPUResizeImage(aug_net.out_size), )
                        if need_to_resize else ()
                    )
                ]
            ),
            'labels': transforms.Lambda(
                lambda x: x if dataset.is_multilabel else np.argmax(x)
            )
        }
    }
    return transf_dict


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
# Helpers
# ########################################################
class MonkeyNet(nn.Sequential):
    '''
    The idea of the monkeynet is to expose all attributes of the networks
    it's given and is therefore a special kind of Sequential network
    '''
    def __init__(self, *nets):
        super().__init__(*nets)
        super().__setattr__('__finished_init__', True)

    def __getattr__(self, attr):
        try:
            super(nn.Sequential, self).__getattribute__(attr)
        except AttributeError:
            super(nn.Sequential, self).__getattribute__('__finished_init__')
        for m in self.children():
            try:
                return getattr(m, attr)
            except AttributeError:
                continue
        raise AttributeError('The monkey is sorry because it could not find ''{0}'''.format(attr))

    def __setattr__(self, attr, val):
        try:
            super(nn.Sequential, self).__getattribute__(attr)
            super(nn.Sequential, self).__setattr__(attr, val)
            return
        except AttributeError:
            try:
                super(nn.Sequential, self).__getattribute__('__finished_init__')
            except AttributeError:
                super(nn.Sequential, self).__setattr__(attr, val)
                return
        for m in self.children():
            try:
                getattr(m, attr)
                setattr(m, attr, val)
                return
            except AttributeError:
                continue
        raise AttributeError('The monkey is sorry because it could not set ''{0}'''.format(attr))


class CPUFormatLabel(nn.Module):
    def __init__(self, multilabel):
        super().__init__()
        self.is_multilabel = multilabel

    def __call__(self, x: torch.Tensor):
        x = x if self.is_multilabel else np.argmax(x, axis=0)
        return x

    def forward(self, x):
        return self(x)


class CPUSelectSegmentsDynamic(nn.Module):
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
            choices = np.array([np.random.choice(c, 1) for c in choices]).squeeze()
        x = x[choices.tolist()]
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


class CPUFormatImage(nn.Module):
    def __init__(self, to_pil_image):
        super().__init__()
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
        super().__init__()
        self.target_im_size = (
            (target_im_size, target_im_size)
            if isinstance(target_im_size, int)
            else target_im_size
        )
        self.convert_to_pil = transforms.ToPILImage()

    def __call__(self, x):
        x = [self.convert_to_pil(e) for e in x[:]]
        x = [e.resize(self.target_im_size) for e in x]
        x = np.array([np.array(e) for e in x])
        if len(x.shape) < 4:
            x = np.expand_dims(x, -1)
        return x


class CPURandomCropPad(nn.Module):
    def __init__(self, target_im_size):
        super().__init__()
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
