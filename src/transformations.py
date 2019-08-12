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
                    SelectSamples(model.num_segments),
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
                    SelectSamples(model.num_segments),
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
def monkey_getter(obj_a, obj_b, attr):
    try:
        return obj_a.__dict[attr]
    except KeyError:
        try:
            return obj_b.__dict[attr]
        except KeyError:
            msg = "'{0}' object has no attribute '{1}'"
            raise AttributeError(msg.format(type(obj_a).__name__, attr))


def monkey_setter(obj_a, obj_b, attr, val):
    try:
        obj_a.__dict[attr] = val
    except KeyError:
        try:
            obj_b.__dict[attr] = val
        except KeyError:
            msg = "'{0}' object has no attribute '{1}'"
            raise AttributeError(msg.format(type(obj_a).__name__, attr))


class MonkeyNet(nn.Sequential):
    '''
    The idea of the monkeynet is to expose all attributes of the networks
    it's given and is therefore a special kind of Sequential network
    '''
    def __init__(self, *args):
        self._nets = args
        super().__init__(*args)

    def __getattr__(self, attr):
        try:
            return self.__dict__[attr]
        except KeyError:
            pass
        try:
            return getattr(super(nn.Sequential, self), attr)
        except AttributeError:
            pass
        for m in self._nets:
            try:
                return getattr(m, attr)
            except AttributeError:
                continue
        raise AttributeError('The monkey is sorry because it could not find ''{0}'''.format(attr))

    def __setattr__(self, attr, val):
        if attr == '_nets':
            self.__dict__.update({attr: val})
            return
        if hasattr(self, attr):
            self.__dict__[attr] = val
            return
        if hasattr(super(nn.Sequential, self), attr):
            setattr(super(nn.Sequential, self), attr, val)
            return
        for m in self._nets:
            if hasattr(m, attr):
                setattr(m, attr, val)
                return
        self.__dict__.update({attr: val})
    # def __setattr__(self, attr, val):
    #     try:
    #         super().__setattr__(attr, val)
    #     except AttributeError:
    #         pass
    #     for m in self._nets:
    #         try:
    #             m.__dict__[attr] = val
    #         except KeyError:
    #             continue
    #     raise AttributeError('The monkey is sorry but could not find ''{0}'''.format(attr))


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
