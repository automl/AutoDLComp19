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
                    SelectSamples(dataset.num_segments),
                    RandomCropPad(model.input_size)
                ]
            ),
            'labels': torchvision.transforms.Lambda(
                lambda x: torch.Tensor(format_label(x)).unsqueeze(0).to(DEVICE)
            )
        },
        'test': {
            'samples': torchvision.transforms.Compose(
                [
                    SelectSamples(dataset.num_segments),
                    RandomCropPad(model.input_size)
                ]
            ),
            'labels': torchvision.transforms.Lambda(
                lambda x: torch.Tensor(format_label(x)).unsqueeze(0).to(DEVICE)
            )
        }
    }
    return transf_dict


# ########################################################
# Helpers
# ########################################################
def format_label(label, is_multilabel):
    return label if is_multilabel else np.argmax(label, axis=1)
