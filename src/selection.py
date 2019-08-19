'''
In this module we define algorithms to select the right starting model,
optimizer, loss function and if wanted learning rate scheduler.
Currently switching models mid execution is not supported as that were given
no though as of yet
'''
import os
import time
from collections import OrderedDict
from functools import wraps

import hjson
import numpy as np
import torch
import torch.nn as nn  # noqa: F401
from torch.optim.lr_scheduler import StepLR  # noqa: F401
from utils import LOGGER

HUBNAME = 'automl/videomodels'
TORCH_HOME = os.path.join(os.path.dirname(__file__), 'torchhome')
os.environ['TORCH_HOME'] = TORCH_HOME
models_available = torch.hub.list(HUBNAME)
LOGGER.info("AVAILABLE MODELS: {0}".format(models_available))
with open(os.path.join(TORCH_HOME, 'manifest.hjson')) as manifest:
    HUBMANIFEST = hjson.load(manifest)


def baseline_selector(autodl_model, dataset, selection_args):
    LOGGER.info("TRAIN SET LENGTH: {0}".format(dataset.num_samples))
    LOGGER.info("INPUT SHAPE MEDIAN: {0}".format(dataset.median_shape))
    LOGGER.info("IS MULTILABEL: {0}".format(dataset.is_multilabel))

    scheduler = None
    selection_args.update(selection_args.pop('optim_args'))
    selection_args.update(
        OrderedDict(
            {
                'num_classes':
                    dataset.num_classes,
                'classification_type':
                    'multilabel' if dataset.is_multilabel else 'multiclass',
            }
        )
    )
    if dataset.mean_shape[0] == 1:  # image network
        kakaomodel = torch.hub.load(
            'kakaobrain/autoclint', 'KakaoModel', autodl_model.metadata
        )
        LOGGER.info('LETTING SOMEONE ELSE FLY OUR BANANA PLANE!')
        LOGGER.info('BRACE FOR IMPACT!')
        LOGGER.info('AND REMEMBER THAT I AM NOT RESPONSABLE!')
        autodl_model.train = EarlyStop(kakaomodel.train, autodl_model)
        autodl_model.test = EarlyStop(kakaomodel.test, autodl_model)
        autodl_model.train(dataset.dataset, autodl_model.current_remaining_time)
        model = None
        optimizer = None
        loss_fn = None
    else:  # video network
        model_name, checkpoint_file = ('bninception', 'BnT_Video_input_128.pth.tar')
        model, optimizer, loss_fn = torch.hub.load(
            HUBNAME, model_name, pretrained=True, url=checkpoint_file, **selection_args
        )
        # Not sure if these parameters reach to model so I set them here
        model.dropout = selection_args['dropout']
        model.num_segments = selection_args['num_segments']
        model.freeze_portion = selection_args['freeze_portion']

    # If not set to true or at all, amp will not be use
    # If I remember correctly, freezing layers might break amp in which
    # case we set this to false. So in a sense, this flag has the final
    # say in the matter of wheter or not to use amp
    if model is not None:
        setattr(model, 'amp_compatible', False)
        if hasattr(model, 'partialBN'):
            model.partialBN(False)
        CheckModesAndFreezing(model)
    return model, loss_fn, optimizer, scheduler


def kakao_selector(autodl_model, dataset, selection_args):
    LOGGER.info("TRAIN SET LENGTH: {0}".format(dataset.num_samples))
    LOGGER.info("INPUT SHAPE MEDIAN: {0}".format(dataset.median_shape))
    LOGGER.info("IS MULTILABEL: {0}".format(dataset.is_multilabel))

    scheduler = None
    selection_args.update(
        OrderedDict(
            {
                'num_classes':
                    dataset.num_classes,
                'classification_type':
                    'multilabel' if dataset.is_multilabel else 'multiclass',
            }
        )
    )
    if dataset.mean_shape[0] == 1:  # image network
        optim_image = selection_args.pop('image')
        optim_image.update(optim_image.pop('optim_args'))
        selection_args.update(optim_image)
        kakaomodel = torch.hub.load(
            'kakaobrain/autoclint',
            'KakaoModel',
            autodl_model.metadata,
            parser_args=selection_args
        )
        LOGGER.info('LETTING SOMEONE ESLE FLY OUR BANANA PLANE!')
        LOGGER.info('BRACE FOR IMPACT!')
        LOGGER.info('AND REMEMBER THAT I AM NOT RESPONSABLE!')
        autodl_model.train = EarlyStop(kakaomodel.train, autodl_model)
        autodl_model.test = EarlyStop(kakaomodel.test, autodl_model)
        autodl_model.train(dataset.dataset, autodl_model.current_remaining_time)
        model = None
        optimizer = None
        loss_fn = None
    else:  # video network
        # NOTE(Philipp): This is the current video api,
        # in the future we just want the model to be returned
        # NOTE(Philipp): Currently all parameters the model might require
        # but aren't set here are loaded from the parser_args default
        # at 'torchhome/hub/autodlcomp_models_master/video/opts.py' or on the
        # model __init__ args itself
        optim_image = selection_args.pop('video')
        optim_image.update(optim_image.pop('optim_args'))
        selection_args.update(optim_image)
        model_name, checkpoint_file = ('bninception', 'BnT_Video_input_128.pth.tar')
        #model_name, checkpoint_file = (
        #    'averagenet', 'Averagenet_RGB_Kinetics_128.pth.tar'
        #)
        model, optimizer, loss_fn = torch.hub.load(
            HUBNAME, model_name, pretrained=True, url=checkpoint_file, **selection_args
        )
        model.dropout = selection_args['dropout']
        model.num_segments = int(
            dataset.median_shape[0] / selection_args['segment_coeff']
        )

    # If not set to true or at all, amp will not be use
    # If I remember correctly, freezing layers might break amp in which
    # case we set this to false. So in a sense, this flag has the final
    # say in the matter of wheter or not to use amp
    if model is not None:
        setattr(model, 'amp_compatible', False)
        if hasattr(model, 'partialBN'):
            model.partialBN(False)
        CheckModesAndFreezing(model)
    return model, loss_fn, optimizer, scheduler


# ########################################################
# Helpers
# ########################################################
class struct:
    def __init__(self, ordereddict):
        self.__dict__ = ordereddict


def CheckModesAndFreezing(model):
    # This is a quick indicator about whether or not some layers are frozen
    # contrary to expectation. Expectation is:
    # train/freshly created: nothing frozen, eval: all frozen
    LOGGER.debug('##########################################################')
    LOGGER.debug('MODEL IS IN BIRTHDAY SUIT')
    has_frozen = np.all([m.training for m in model.modules()])
    LOGGER.debug('ALL TRAINING FLAGS ARE TRUE:\t{0}'.format(has_frozen))
    model.eval()
    LOGGER.debug('##########################################################')
    LOGGER.debug('MODEL IS IN EVAL MODE')
    has_unfrozen = np.all([not m.training for m in model.modules()])
    LOGGER.debug('ALL TRAINING FLAGS ARE FALSE:\t{0}'.format(has_unfrozen))
    model.train()
    LOGGER.debug('##########################################################')
    LOGGER.debug('MODEL IS IN TRAIN MODE')
    has_frozen = np.all([m.training for m in model.modules()])
    LOGGER.debug('ALL TRAINING FLAGS ARE TRUE:\t{0}'.format(has_frozen))
    LOGGER.debug('##########################################################')
    req_grad_list = np.array([p.requires_grad for p in model.parameters()])
    fp = np.where(np.logical_not(req_grad_list))[0].shape[0] / req_grad_list.shape[0]
    LOGGER.debug('{0:.0f}% OF THE NETWORK IS FROZEN'.format(fp * 100))
    LOGGER.debug('##########################################################')
    LOGGER.debug('All done. You can get back to work now!')
    LOGGER.debug('##########################################################')


def EarlyStop(f, autodl_model):
    @wraps(f)
    def decorated(*args, **kwargs):
        if autodl_model.starting_budget > autodl_model.config.earlystop:
            if 'remaining_time_budget' in kwargs:  # now it's getting silly
                old_budget = kwargs['remaining_time_budget']
                diff = autodl_model.starting_budget - autodl_model.config.earlystop
                new_budget = old_budget - diff
                kwargs.update({'remaining_time_budget': new_budget})
            else:
                old_budget = args[-1]
                diff = autodl_model.starting_budget - autodl_model.config.earlystop
                new_budget = old_budget - diff
                args = tuple([args[0], new_budget])
        if time.time() - autodl_model.birthday > autodl_model.config.earlystop:
            return
        return f(*args, **kwargs)

    return decorated
