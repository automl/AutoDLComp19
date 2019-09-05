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
import pandas as pd
import torch
import torch.nn as nn  # noqa: F401
import transformations
from torch.optim.lr_scheduler import StepLR  # noqa: F401
from utils import LOGGER
import pretrainedmodels

TORCH_HOME = os.path.join(os.path.dirname(__file__), 'torchhome')
os.environ['TORCH_HOME'] = TORCH_HOME

with open(os.path.join(TORCH_HOME, 'manifest.hjson')) as manifest:
    HUBMANIFEST = hjson.load(manifest)


class Selector(object):
    def __init__(self, conf):
        self.conf = conf

    def _determin_modality(self, dataset):
        if dataset.mean_shape[0] == 1:
            LOGGER.info("DETERMINED MODALITY: IMAGE")
            return 'image'
        if dataset.mean_shape[0] > 1:
            LOGGER.info("DETERMINED MODALITY: VIDEO")
            return 'video'
        raise NotImplementedError

    def select(self, autodl_model, dataset):
        LOGGER.info("TRAIN SET LENGTH: {0}".format(dataset.num_samples))
        LOGGER.info("INPUT SHAPE MEDIAN: {0}".format(dataset.median_shape))
        LOGGER.info("IS MULTILABEL: {0}".format(dataset.is_multilabel))
        modality = self._determin_modality(dataset)
        conf = self.conf.pop(modality)
        conf.update(
            OrderedDict(
                {
                    'num_classes':
                        dataset.num_classes,
                    'classification_type':
                        'multilabel' if dataset.is_multilabel else 'multiclass',
                }
            )
        )
        return getattr(self, modality)(autodl_model, dataset, conf)

    def image(self, autodl_model, dataset, conf):
        if 'optim_args' in conf:  # Needed because the api below needs it flat
            conf.update(conf.pop('optim_args'))
        kakaomodel = torch.hub.load(
            'kakaobrain/autoclint', 'KakaoModel', autodl_model.metadata, add_args=conf
        )
        LOGGER.info('LETTING SOMEONE ESLE FLY OUR BANANA PLANE!')
        LOGGER.info('BRACE FOR IMPACT!')
        LOGGER.info('AND REMEMBER THAT I AM NOT RESPONSABLE!')

        autodl_model.train = EarlyStop(kakaomodel.train, autodl_model)
        autodl_model.test = EarlyStop(kakaomodel.test, autodl_model)
        autodl_model.train(dataset.dataset, autodl_model.current_remaining_time)

        updated_attr = {}
        return updated_attr

    def video(self, autodl_model, dataset, conf):
        model = pretrainedmodels.__dict__[conf['model_name']](num_classes=1000, pretrained='imagenet')
        dim_feats = model.last_linear.in_features  # =2048
        model.last_linear = nn.Linear(dim_feats, dataset.num_classes)
        for param in model.parameters():
            param.requires_grad = True

        if conf['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=conf['optim_args']['lr'])
        elif conf['optimizer'] == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=conf['optim_args']['lr'])
        else:
            raise ValueError("Unknown optimizer type")

        if conf['classification_type'] == 'multiclass':
            loss_fn = torch.nn.CrossEntropyLoss().cuda()
        elif conf['classification_type'] == 'multilabel':
            loss_fn = torch.nn.BCEWithLogitsLoss().cuda()
        else:
            raise ValueError("Unknown loss function type")

        # As the transformations are dependend on the modality and maybe even
        # the model we set/apply it here
        get_and_apply_transformations = getattr(
            transformations.video, conf['transformations']
        )
        transf_args = conf['transformation_args'] if 'transformation_args' in conf else {}
        model, transf = get_and_apply_transformations(model, dataset, transf_args)

        # Setting the policy to use
        # In a more sophisticated policy, this could change the
        # execution depth of an ANN (https://arxiv.org/abs/1708.06832)
        # which might help on big test datasets
        policy = None

        LOGGER.info('LR:\t\t\t{0:.5e}'.format(optimizer.param_groups[0]['lr']))

        # Create an update dictionary for the autodl_model
        # this will update the autodl_model attributes which
        # are used to create the dataloaders
        updated_attr = {
            'model':
                model,
            'optimizer':
                optimizer,
            'loss_fn':
                loss_fn,
            'lr_scheduler':
                None,
            'transforms':
                transf,
            'policy_fn':
                None,
            'train_loader_args':
                conf['train_loader_args'] if 'train_loader_args' in conf else {},
            'test_loader_args':
                conf['test_loader_args'] if 'test_loader_args' in conf else {}
        }
        return updated_attr

    def nlp(self, autodl_model, dataset, conf):
        raise NotImplementedError

    def series(self, autodl_model, dataset, conf):
        raise NotImplementedError


# ########################################################
# Helpers
# ########################################################
def append_to_dataframe(frame, val):
    # Convenience function to increase readability
    return frame.append([val.detach().cpu().tolist()], ignore_index=True)


class struct:
    def __init__(self, ordereddict):
        self.__dict__ = ordereddict


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
