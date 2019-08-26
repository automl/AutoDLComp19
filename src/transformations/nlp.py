# from collections import OrderedDict
import re

# import numpy as np
# import torch
import torch.nn as nn

# from scipy.stats import norm
# from torchvision import transforms
# from utils import LOGGER, AugmentNet, MonkeyNet

MAX_VOCAB_SIZE = 10000

# # ########################################################
# # Transformations and augmentation prepending stacks
# # ########################################################
# def baseline(model, dataset):
#     LOGGER.info('Using ###   BaselineAugmentNet   ### for transformationstack')

#     # Classical transformations performed per sample
#     transf_dict = {
#         'train':
#             {
#                 'samples':
#                     transforms.Compose(
#                         [
#                             CPUDynamicSelectSegmentsUniform(model),
#                             RandomCropPad(model.input_size)
#                         ]
#                     ),
#                 'labels':
#                     transforms.
#                     Lambda(lambda x: x if dataset.is_multilabel else np.argmax(x))
#             },
#         'test':
#             {
#                 'samples':
#                     transforms.Compose(
#                         [
#                             CPUDynamicSelectSegmentsUniform(model),
#                             RandomCropPad(model.input_size)
#                         ]
#                     ),
#                 'labels':
#                     transforms.
#                     Lambda(lambda x: x if dataset.is_multilabel else np.argmax(x))
#             }
#     }

#     # Prepend an augmentation network performing transformations on the gpu
#     # on a whole batch
#     aug_net = AugmentNet(
#         {
#             'train': [],
#             'test': []
#         }
#     )
#     # To expose the original model's attributes use the MonkeyNet(nn.Sequential)
#     model = MonkeyNet(OrderedDict([
#         ('aug_net', aug_net),
#         ('main_net', model),
#     ]))
#     return model, transf_dict


# ########################################################
# Helpers
# ########################################################
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


class CleanENText(nn.Module):
    def __call__(self, x):
        REPLACE_BY_SPACE_RE = re.compile('["/(){}\[\]\|@,;]')
        BAD_SYMBOLS_RE = re.compile('[^0-9a-zA-Z #+_]')

        ret = []
        for line in x:
            # text = text.lower() # lowercase text
            line = REPLACE_BY_SPACE_RE.sub(' ', line)
            line = BAD_SYMBOLS_RE.sub('', line)
            line = line.strip()
            ret.append(line)
        return ret


class CleanZHText(nn.Module):
    def __call__(self, x):
        REPLACE_BY_SPACE_RE = re.compile('[“”【】/（）：！～「」、|，；。"/(){}\[\]\|@,\.;]')

        ret = []
        for line in x:
            line = REPLACE_BY_SPACE_RE.sub(' ', line)
            line = line.strip()
            ret.append(line)
        return ret


# def _tokenize_chinese_words(text):
#     return ' '.join(jieba.cut(text, cut_all=False))

# class VectorizeData(nn.Module):
#     def __call__(self, x_train, x_val=None):
#         vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_features = MAX_VOCAB_SIZE)
#         if x_val:
#             full_text = x_train + x_val
#         else:
#             full_text = x_train
#         vectorizer.fit(full_text)
#         train_vectorized = vectorizer.transform(x_train)
#         if x_val:
#             val_vectorized = vectorizer.transform(x_val)
#             return train_vectorized, val_vectorized, vectorizer
#         return train_vectorized, vectorizer
