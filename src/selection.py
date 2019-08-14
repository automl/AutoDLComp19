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
        'partial_bn': False,
    }))
    if dataset.mean_shape[0] == 1:  # image network
        num_pixel = np.prod(dataset.mean_shape[1:2])
        if num_pixel < 10000:                   # select network based on average number of pixels in the dataset
            model_name, checkpoint_file = ('bninception', 'BnT_Image_Input_64.pth.tar')
        else:
            model_name, checkpoint_file = ('bninception', 'BnT_Image_Input_128.tar')
    else:  # video network
        # model_name, checkpoint_file = ('bninception', 'BnT_Video_input_128.pth.tar')
        model_name, checkpoint_file = ('bninception', 'BnT_Video_input_128.pth.tar')
    model, optimizer, loss_fn = torch.hub.load(
        HUBNAME, model_name, pretrained=True, url=checkpoint_file, **modelargs
    )
    if hasattr(model, 'partialBN'):
        model.partialBN(False)
    model.dropout = modelargs['initial_dropout']
    scheduler = StepLR(optimizer, modelargs['lr_step'], 1 - modelargs['lr_gamma'])
    # If not set, amp will not be initialized for this model
    setattr(model, 'amp_compatible', False)

    LOGGER.debug('##########################################################')
    LOGGER.debug('MODEL IS IN BIRTHDAY SUIT')
    has_frozen = np.any([not m.training for m in model.modules()])
    LOGGER.debug('MODEL HAS FROZEN MODULES:\t{0}'.format(has_frozen))
    model.eval()
    LOGGER.debug('##########################################################')
    LOGGER.debug('MODEL IS IN EVAL MODE')
    has_unfrozen = np.any([m.training for m in model.modules()])
    LOGGER.debug('MODEL HAS UNFROZEN MODULES:\t{0}'.format(has_unfrozen))
    model.train()
    LOGGER.debug('##########################################################')
    LOGGER.debug('MODEL IS IN TRAIN MODE')
    has_frozen = np.any([not m.training for m in model.modules()])
    LOGGER.debug('MODEL HAS FROZEN MODULES:\t{0}'.format(has_frozen))
    LOGGER.debug('##########################################################')
    LOGGER.debug('All done. You can get back to work now!')
    LOGGER.debug('##########################################################')

    return model, loss_fn, optimizer, scheduler


# Model selectors
def dummy_model_selector(tfsession, dataset):
    '''
    This is a dummy implementation of a selector showing what is expected
    to be performed by it. You are free to add any additional args to it
    as long they are set in the config. The standard ones ar listed above.

    How you decide what model to use given the dataset(TFDataset) is up to you.

    Returns model, loss_fn, optimizer, scheduler

    These should be set accordingly
    setattr(model, 'amp_compatible', True)
    scheduler = None
    '''
    raise NotImplemented


# ########################################################
# Helpers
# ########################################################
class struct:
    def __init__(self, ordereddict):
        self.__dict__ = ordereddict
