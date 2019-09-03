import torch
import math
import numbers
import random

import numpy as np
import torchvision
import torchvision.transforms.functional as F
from PIL import Image, ImageOps, ImageChops


# #########################################################
# Operators
# #########################################################
class GroupRandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_group):

        w, h = img_group[0].size
        th, tw = self.size

        out_images = list()

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        for img in img_group:
            assert (img.size[0] == w and img.size[1] == h)
            if w == tw and h == th:
                out_images.append(img)
            else:
                out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

        return out_images


class GroupCenterCrop(object):
    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupRandomGrayscale(object):
    def __init__(self, output_channels=3, p=0.1):
        self.p = p
        self.worker = torchvision.transforms.Grayscale(
                            num_output_channels=output_channels)

    def __call__(self, img_group):
        if self.p >= random.random():
            return [self.worker(img) for img in img_group]
        else: 
            return img_group


class GroupResize(object):
    """ Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BICUBIC
    """
    def __init__(self, size):
        self.worker = torchvision.transforms.Resize(size)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupRandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __init__(self, is_flow=False):
        self.is_flow = is_flow

    def __call__(self, img_group, is_flow=False):
        v = random.random()
        if v < 0.5:
            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            if self.is_flow:
                for i in range(0, len(ret), 2):
                    ret[i] = ImageOps.invert(
                        ret[i]
                    )  # invert flow pixel values when flipping
            return ret
        else:
            return img_group


class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        rep_mean = self.mean * (tensor.size()[0] // len(self.mean))
        rep_std = self.std * (tensor.size()[0] // len(self.std))

        # TODO: make efficient
        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.sub_(m).div_(s)

        return tensor


class GroupScale(object):
    """ Rescales the input PIL.Image to the given 'scale'
    keeps acpect ratio.
    size: scale of the output image
    interpolation: Default: PIL.Image.BICUBIC
    """
            
    def __init__(self, scale=1, interpolation=Image.BICUBIC):
        self.scale = scale
        self.interpolation = interpolation
        self.w, self.h, self.ow, self.oh = (None, )*4
        self.worker = None

    def __call__(self, img_group):
        w, h = img_group[0].size
        if w != self.w or h != self.h:
            self.w = w
            self.h = h
            if w < h:
                self.ow = int(self.scale * w)
                self.oh = int(self.ow * h / w)
                self.worker = torchvision.transforms.Resize(
                    (self.ow, self.oh), self.interpolation)
            else:
                self.oh = int(self.scale * h)
                self.ow = int(self.oh * w / h)
                self.worker = torchvision.transforms.Resize(
                    (self.ow, self.oh), self.interpolation)

        return [self.worker(img) for img in img_group]


class GroupOverSample(object):
    def __init__(self, crop_size, scale_size=None, flip=True):
        self.crop_size = crop_size if not isinstance(crop_size,
                                                     int) else (crop_size, crop_size)

        if scale_size is not None:
            self.scale_worker = GroupScale(scale_size)
        else:
            self.scale_worker = None
        self.flip = flip

    def __call__(self, img_group):

        if self.scale_worker is not None:
            img_group = self.scale_worker(img_group)

        image_w, image_h = img_group[0].size
        crop_w, crop_h = self.crop_size

        offsets = GroupMultiScaleCrop.fill_fix_offset(
            False, image_w, image_h, crop_w, crop_h
        )
        oversample_group = list()
        for o_w, o_h in offsets:
            normal_group = list()
            flip_group = list()
            for i, img in enumerate(img_group):
                crop = img.crop((o_w, o_h, o_w + crop_w, o_h + crop_h))
                normal_group.append(crop)
                flip_crop = crop.copy().transpose(Image.FLIP_LEFT_RIGHT)

                if img.mode == 'L' and i % 2 == 0:
                    flip_group.append(ImageOps.invert(flip_crop))
                else:
                    flip_group.append(flip_crop)

            oversample_group.extend(normal_group)
            if self.flip:
                oversample_group.extend(flip_group)
        return oversample_group


class GroupFullResSample(object):
    def __init__(self, crop_size, scale_size=None, flip=True):
        self.crop_size = crop_size if not isinstance(crop_size,
                                                     int) else (crop_size, crop_size)

        if scale_size is not None:
            self.scale_worker = GroupScale(scale_size)
        else:
            self.scale_worker = None
        self.flip = flip

    def __call__(self, img_group):

        if self.scale_worker is not None:
            img_group = self.scale_worker(img_group)

        image_w, image_h = img_group[0].size
        crop_w, crop_h = self.crop_size

        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        offsets = list()
        offsets.append((0 * w_step, 2 * h_step))  # left
        offsets.append((4 * w_step, 2 * h_step))  # right
        offsets.append((2 * w_step, 2 * h_step))  # center

        oversample_group = list()
        for o_w, o_h in offsets:
            normal_group = list()
            flip_group = list()
            for i, img in enumerate(img_group):
                crop = img.crop((o_w, o_h, o_w + crop_w, o_h + crop_h))
                normal_group.append(crop)
                if self.flip:
                    flip_crop = crop.copy().transpose(Image.FLIP_LEFT_RIGHT)

                    if img.mode == 'L' and i % 2 == 0:
                        flip_group.append(ImageOps.invert(flip_crop))
                    else:
                        flip_group.append(flip_crop)

            oversample_group.extend(normal_group)
            oversample_group.extend(flip_group)
        return oversample_group


class GroupMultiScaleCrop(object):
    def __init__(
        self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True
    ):
        self.scales = scales if scales is not None else [1, .875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [
            input_size, input_size
        ]
        self.interpolation = Image.BILINEAR

    def __call__(self, img_group):

        im_size = img_group[0].size

        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = [
            img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h))
            for img in img_group
        ]
        ret_img_group = [
            img.resize((self.input_size[0], self.input_size[1]), self.interpolation)
            for img in crop_img_group
        ]
        return ret_img_group

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]

        # find a crop size
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [
            self.input_size[1] if abs(x - self.input_size[1]) < 3 else x
            for x in crop_sizes
        ]
        crop_w = [
            self.input_size[0] if abs(x - self.input_size[0]) < 3 else x
            for x in crop_sizes
        ]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(
                image_w, image_h, crop_pair[0], crop_pair[1]
            )

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(
            self.more_fix_crop, image_w, image_h, crop_w, crop_h
        )
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

        return ret


class GroupRandomSizedCrop(object):
    """Random crop the given PIL.Image to a random size of (0.08 to 1.0) of the original size
    and and a random aspect ratio of 3/4 to 4/3 of the original aspect ratio
    This is popularly used to train the Inception networks
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img_group):
        for attempt in range(10):
            area = img_group[0].size[0] * img_group[0].size[1]
            target_area = random.uniform(0.08, 1.0) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img_group[0].size[0] and h <= img_group[0].size[1]:
                x1 = random.randint(0, img_group[0].size[0] - w)
                y1 = random.randint(0, img_group[0].size[1] - h)
                found = True
                break
        else:
            found = False
            x1 = 0
            y1 = 0

        if found:
            out_group = list()
            for img in img_group:
                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert (img.size == (w, h))
                out_group.append(img.resize((self.size, self.size), self.interpolation))
            return out_group
        else:
            # Fallback
            scale = GroupScale(self.size, interpolation=self.interpolation)
            crop = GroupRandomCrop(self.size)
            return crop(scale(img_group))


class Stack(object):
    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == 'L':
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                return np.concatenate(
                    [np.array(x)[:, :, ::-1] for x in img_group], axis=2
                )
            else:
                return np.concatenate(img_group, axis=2)


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """

    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255) if self.div else img.float()


class SelectSamples(object):
    """
    given a video 4D array, randomly select sample images within segments
    """

    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __call__(self, pics):
        samples = pics.shape[0]
        if samples <= self.num_samples:
            samples_select = np.linspace(0, samples-1, self.num_samples, dtype=int)
        else:
            samples_select = np.zeros(self.num_samples, dtype=int)
            bounds = np.linspace(0, samples-1, self.num_samples+1, dtype=int)
            for i in range(self.num_samples):
                samples_select[i] = np.random.randint(bounds[i], bounds[i+1])
        # room for improvement: the last frame is never chosen
        return pics[samples_select]


class RandomCropPad(object):
    """
    instead of resizing, crop/pad video to desired shape directly
    """
    def __init__(self, size_des):
        self.size_des = size_des
        print(self.size_des)

    def __call__(self, pics):
        row = pics.shape[1]
        row_des = self.size_des
        col = pics.shape[2]
        col_des = self.size_des

        if row <= row_des: # pad rows
            row_pad_start = int(np.floor((row_des-row)/2))
            row_pad_end = row + int(np.floor((row_des-row)/2))
            row_start = 0
            row_end = row
        else: # crop rows
            row_rand = int(np.random.random() * int(np.floor((row-row_des))))
            row_pad_start = 0
            row_pad_end = row_des
            row_start = row_rand
            row_end = row_des + row_rand
        if col <= col_des: # pad columns
            col_pad_start = int(np.floor((col_des-col)/2))
            col_pad_end = col + int(np.floor((col_des-col)/2))
            col_start = 0
            col_end = col
        else: # crop columns
            col_rand = int(np.random.random() * int(np.floor((col-col_des))))
            col_pad_start = 0
            col_pad_end = col_des
            col_start = col_rand
            col_end = col_des + col_rand

        pics_pad = np.zeros((pics.shape[0], row_des, col_des, pics.shape[3]), dtype=pics.dtype)
        pics_pad[:,row_pad_start:row_pad_end,col_pad_start:col_pad_end,:] = pics[:,row_start:row_end,col_start:col_end,:]
        return pics_pad




class ToPilFormat(object):
    """
    convert from numpy/torch array (B x S_old x H x W x C) to list of PIL images (B x S_new x H x W x C)
    """

    def __call__(self, pics):
        if isinstance(pics, np.ndarray):
            # handle numpy array
            lst = []
            for i in range(len(pics)):
                formatted = (pics[i, ...] * 255).astype('uint8')
                # if we have only one channel, expand it to three channels
                if formatted.shape[-1] is 1:
                    formatted = np.repeat(formatted, 3, axis=-1)
                lst.append(Image.fromarray(formatted))
            return lst
        else:
            # handle torch tensor
            print('torch tensor')
            return [F.to_pil_image(pic) for pic in pics]


class IdentityTransform(object):
    def __call__(self, data):
        return data


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img_group):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img_group[0].size[0]
        w = img_group[0].size[1]

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = int(np.clip(y - self.length // 2, 0, h))
            y2 = int(np.clip(y + self.length // 2, 0, h))
            x1 = int(np.clip(x - self.length // 2, 0, w))
            x2 = int(np.clip(x + self.length // 2, 0, w))

            mask[y1: y2, x1: x2] = 0.

        image_mode = img_group[0].mode
        # for torch tensors
        # mask = torch.from_numpy(mask)
        # mask = mask.expand((len(image_mode), h, w))
        # for np.arrays
        # TODO: Check if PIL.ImageChops.logical_and is faster
        mask = np.repeat(mask[np.newaxis,:, :], 3, axis=0)
        mask =  Image.fromarray(mask, image_mode)
        
        return [ImageChops.multiply(img, mask) for img in img_group]


if __name__ == "__main__":
    trans = torchvision.transforms.Compose(
        [
            GroupScale(256),
            GroupRandomCrop(224),
            Stack(),
            ToTorchFormatTensor(),
            GroupNormalize(mean=[.485, .456, .406], std=[.229, .224, .225])
        ]
    )

    im = Image.open('../tensorflow-model-zoo.torch/lena_299.png')

    color_group = [im] * 3
    rst = trans(color_group)

    gray_group = [im.convert('L')] * 9
    gray_rst = trans(gray_group)

    trans2 = torchvision.transforms.Compose(
        [
            GroupRandomSizedCrop(256),
            Stack(),
            ToTorchFormatTensor(),
            GroupNormalize(mean=[.485, .456, .406], std=[.229, .224, .225])
        ]
    )
    print(trans2(color_group))
