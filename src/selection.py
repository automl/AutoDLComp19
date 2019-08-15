
'''
In this module we define algorithms to select the right starting model,
optimizer, loss function and if wanted learning rate scheduler.
Currently switching models mid execution is not supported as that were given
no though as of yet
'''
import os
from collections import OrderedDict
import numpy as np
import hjson
import torch
from torch.optim.lr_scheduler import StepLR
from utils import LOGGER

HUBNAME = 'autodlcomp/models'
TORCH_HOME = os.path.join(os.path.dirname(__file__), 'torchhome')
os.environ['TORCH_HOME'] = TORCH_HOME
models_available = torch.hub.list(HUBNAME)
LOGGER.info("AVAILABLE MODELS: {0}".format(models_available))
with open(os.path.join(TORCH_HOME, 'manifest.hjson')) as manifest:
    HUBMANIFEST = hjson.load(manifest)


def baseline_selector(tfsession, dataset, modelargs):
    LOGGER.info("TRAIN SET LENGTH: {0}".format(dataset.num_samples))
    LOGGER.info("INPUT SHAPE MEDIAN: {0}".format(dataset.median_shape))
    LOGGER.info("IS MULTILABEL: {0}".format(dataset.is_multilabel))

    modelargs.update(OrderedDict({
        'num_classes': dataset.num_classes,
        'classification_type': 'multilabel' if dataset.is_multilabel else 'multiclass',
    }))
    model_name, checkpoint_file = ('averagenet', 'Averagenet_RGB_Kinetics_128.pth.tar')

    # NOTE(Philipp): This is the current video api,
    # in the future we just want the model to be returned
    # NOTE(Philipp): Currently all parameters the model might require
    # but aren't set here are loaded from the parser_args default
    # at 'torchhome/hub/autodlcomp_models_master/video/opts.py' or on the
    # model __init__ args itself
    model, optimizer, loss_fn = torch.hub.load(
        HUBNAME, model_name, pretrained=True, url=checkpoint_file, **modelargs
    )
    if hasattr(model, 'partialBN'):
        model.partialBN(False)
    # Not sure if these parameters reach to model so I set them here
    model.dropout = modelargs['dropout']
    model.num_segments = modelargs['num_segments']
    scheduler = None

    # If not set to true or at all, amp will not be use
    # If I remember correctly, freezing layers might break amp in which
    # case we set this to false. So in a sense, this flag has the final
    # say in the matter of wheter or not to use amp
    setattr(model, 'amp_compatible', False)

    # This is a quick indicator about whether or not some layers are frozen
    # contrary to expectation. Expectation is:
    # train/freshly created: nothing frozen, eval: all frozen
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


def master_selector(tfsession, dataset, modelargs):
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

    # NOTE(Philipp): This is the current video api,
    # in the future we just want the model to be returned
    model, optimizer, loss_fn = torch.hub.load(
        HUBNAME, model_name, pretrained=True, url=checkpoint_file, **modelargs
    )
    if hasattr(model, 'partialBN'):
        model.partialBN(False)
    model.dropout = modelargs['initial_dropout']
    scheduler = StepLR(optimizer, modelargs['lr_step'], 1 - modelargs['lr_gamma'])

    # If not set to true or at all, amp will not be initialized for this model
    setattr(model, 'amp_compatible', False)

    # This is a quick indicator about whether or not some layers are frozen
    # contrary to expectation (train/freshly created: nothing frozen, eval: all frozen)
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
