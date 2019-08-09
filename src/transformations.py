import numpy as np
import torch
import torchvision

from torchhome.hub.autodlcomp_models_master.video.transforms import SelectSamples, RandomCropPad


# Set the device which torch should use
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Sample/Input
def default_transformations_selector(dataset, model):
    transf_dict = {
        'train': {
            'samples': torchvision.transforms.Compose(
                [
                    SelectSamples(model.model.num_segments),
                    RandomCropPad(model.model.input_size)
                ]
            ),
            'labels': torchvision.transforms.Lambda(
                lambda x: x if dataset.is_multilabel else np.argmax(x)
            )
        },
        'test': {
            'samples': torchvision.transforms.Compose(
                [
                    SelectSamples(model.model.num_segments),
                    RandomCropPad(model.model.input_size)
                ]
            ),
            'labels': torchvision.transforms.Lambda(
                lambda x: x if dataset.is_multilabel else np.argmax(x)
            )
        }
    }
    return transf_dict


# ########################################################
# Helpers
# ########################################################
