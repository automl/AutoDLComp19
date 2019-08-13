# Determin modality and select the right model, optimizer and loss
import os
from collections import OrderedDict
import numpy as np
import hjson
import torch
from torch.optim.lr_scheduler import StepLR
from utils import LOGGER
# from torchhome.hub.autodlcomp_models_master.video.load_models import load_model_and_optimizer, load_loss_criterion

HUBNAME = 'autodlcomp/models'
TORCH_HOME = os.path.join(os.path.dirname(__file__), 'torchhome')
os.environ['TORCH_HOME'] = TORCH_HOME
with open(os.path.join(TORCH_HOME, 'manifest.hjson')) as manifest:
    HUBMANIFEST = hjson.load(manifest)


# The idea is get a dataset from our supportdataset(the sets we have pretrained models for)
# which is similar to the dataset given and base our model selection on it. This means
# we would iterate over all available models x datasets and choose/adjust setting saved in the
# manifest file
def master_selector(tfsession, dataset, modelargs):
    models_available = torch.hub.list(HUBNAME)

    LOGGER.info("AVAILABLE MODELS: {0}".format(models_available))
    LOGGER.info("TRAIN SET LENGTH: {0}".format(dataset.num_samples))
    LOGGER.info("INPUT SHAPE MEDIAN: {0}".format(dataset.median_shape))
    LOGGER.info("IS MULTILABEL: {0}".format(dataset.is_multilabel))

    modelargs.update(OrderedDict({
        'num_classes': dataset.num_classes,
        'classification_type': 'multilabel' if dataset.is_multilabel else 'multiclass',
    }))
    if dataset.mean_shape[0] == 1:  # image network
        num_pixel = np.prod(dataset.mean_shape[1:2])
        if num_pixel < 10000:                   # select network based on average number of pixels in the dataset
            model_name, checkpoint_file = ('bninception', 'BnT_Image_Input_64.pth.tar')
        else:
            model_name, checkpoint_file = ('bninception', 'BnT_Image_Input_128.tar')
    else:  # video network
        # model_name, checkpoint_file = ('bninception', 'BnT_Video_input_128.pth.tar')
        model_name, checkpoint_file = ('averagenet', 'Averagenet_RGB_Kinetics_128.pth.tar')
    model, optimizer, loss_fn = torch.hub.load(
        HUBNAME, model_name, pretrained=True, url=checkpoint_file, **modelargs
    )
    model.dropout = modelargs['initial_dropout']
    scheduler = StepLR(optimizer, modelargs['lr_step'], 1 - modelargs['lr_gamma'])
    # If not set, amp will not be initialized for this model
    setattr(model, 'amp_compatible', True)

    return model, loss_fn, optimizer, scheduler


# Model selectors
def default_model_selector(tfsession, dataset, use_wrappernet=False):
    models_available = torch.hub.list(HUBNAME)
    conf = struct(HUBMANIFEST['averagenet']['kinetics'])

    LOGGER.info("AVAILABLE MODELS: {0}".format(models_available))
    LOGGER.info("TRAIN SET LENGTH: {0}".format(dataset.num_samples))
    LOGGER.info("INPUT SHAPE MEDIAN: {0}".format(dataset.median_shape))
    LOGGER.info("IS MULTILABEL: {0}".format(dataset.is_multilabel))

    modeloptimargs = {
        'parser_args': {
            'dropout': 0.2,
            'lr': 0.005,
            'num_classes': dataset.num_classes,
            'num_segments': 2,
            'modality': 'RGB',
            'classification_type': 'multilabel' if dataset.is_multilabel else 'multiclass',
        }
    }

    model, optimizer, loss_fn = torch.hub.load(
        HUBNAME, 'averagenet', pretrained=True, url=conf.checkpoint_file, **modeloptimargs
    )

    # This is an example how you could get the optimizer from the manifest
    # ####
    # optimizer_class = getattr(
    #     torch.optim,
    #     conf.optimizer
    # )
    # optimizer = optimizer_class(
    #     model.parameters(),
    #     optimargs
    # )
    # optimizer = torch.optim.Adam(model.parameters(), **optimargs)

    # Return that this model can be trained using amp
    # Only set this to true if the optimizer and the model can be monkey patched by
    # nvidia apex's amp and amp should be used!
    setattr(model, 'amp_compatible', True)

    return model, loss_fn, optimizer


# ########################################################
# Helpers
# ########################################################
class struct:
    def __init__(self, ordereddict):
        self.__dict__ = ordereddict
