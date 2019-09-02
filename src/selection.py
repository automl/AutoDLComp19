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
from utils import (  # noqa: F401
    LOGGER, PREDICT, PREDICT_AND_VALIDATE, TRAIN, VALIDATE, metrics, transform_time_abs
)

TORCH_HOME = os.path.join(os.path.dirname(__file__), 'torchhome')
os.environ['TORCH_HOME'] = TORCH_HOME

with open(os.path.join(TORCH_HOME, 'manifest.hjson')) as manifest:
    HUBMANIFEST = hjson.load(manifest)


class Selector(object):
    def __init__(self, conf):
        self.conf = conf

    def _determin_modality(self, dataset):
        if isinstance(dataset, list):
            LOGGER.info("DETERMINED MODALITY: NLP")
            return 'nlp'
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
        LOGGER.info('LETTING SOMEONE ELSE FLY OUR BANANA PLANE!')
        LOGGER.info('BRACE FOR IMPACT!')
        LOGGER.info('AND REMEMBER THAT I AM NOT RESPONSABLE!')

        autodl_model.train = EarlyStop(kakaomodel.train, autodl_model)
        autodl_model.test = EarlyStop(kakaomodel.test, autodl_model)
        autodl_model.train(dataset.dataset, autodl_model.current_remaining_time)

        updated_attr = {}
        return updated_attr

    def video(self, autodl_model, dataset, conf):
        if 'optim_args' in conf:  # Needed because video models api needs it flat
            conf.update(conf.pop('optim_args'))
        HUBNAME = 'automl/videomodels'
        model_name, checkpoint_file = (
            'averagenet', 'Averagenet_RGB_Kinetics_128.pth.tar'
        )
        model, optimizer, loss_fn = torch.hub.load(
            HUBNAME, model_name, pretrained=True, url=checkpoint_file, **conf
        )
        # Not sure if these parameters reached the model so I set them here
        model.dropout = conf['dropout']
        model.num_segments = int(dataset.mean_shape[0] / conf['segment_coeff'])
        model.freeze_portion = conf['freeze_portion']

        # As the transformations are dependend on the modality and maybe even
        # the model we set/apply it here
        get_and_apply_transformations = getattr(
            transformations.video, conf['transformation']
        )
        transf_args = conf['transformation_args'] if 'transformation_args' in conf else {}
        model, transf = get_and_apply_transformations(model, dataset, **transf_args)

        if hasattr(model, 'partialBN'):
            model.partialBN(False)
        CheckModesAndFreezing(model)

        # Setting the policy to use
        # In a more sophisticated policy, this could change the
        # execution depth of an ANN (https://arxiv.org/abs/1708.06832)
        # which might help on big test datasets
        policy = adaptive_policy

        LOGGER.info(
            'FROZEN LAYERS:\t{}/{}'.format(
                len([x for x in model.parameters() if not x.requires_grad]),
                len([x for x in model.parameters()])
            )
        )
        LOGGER.info('NUM_SEGMENTS:\t{0}'.format(model.num_segments))
        LOGGER.info('DROPOUT:\t\t{0:.4g}'.format(model.dropout))
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
                policy(
                    num_classes=dataset.num_classes,
                    **(conf['policy_args'] if 'policy_args' in conf else {})
                ),
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


################################################################################
# Training Policies
################################################################################
# Different modalities should have different policies
# They might even depend on the model used and how it should change
# depending on the current context during training
################################################################################
class adaptive_policy():
    def __init__(self, t_diff, num_classes):
        self.t_diff = t_diff

        self.t_acc = pd.DataFrame()
        self.t_auc = pd.DataFrame()
        self.v_acc = pd.DataFrame()
        self.v_auc = pd.DataFrame()

        self.labels_seen = np.zeros((num_classes, ))

    def __call__(
        self,
        autodl_model,
        t_start,
        r_budget,
        tlabels: np.array,
        tout: np.array,
        tloss: np.array,
        vlabels: np.array,
        vout: np.array,
        vloss: np.array,
    ):
        is_multilabel = len(tlabels.shape) > 1 and np.any(tlabels.sum(axis=1) > 1)
        if is_multilabel:
            self.labels_seen += tlabels.sum(axis=0)
        else:
            loh = np.zeros_like(tout)
            loh[np.arange(len(tout)), tlabels] = 1
            self.labels_seen += loh.sum(axis=0)
        tacc, tauc = (
            metrics.accuracy(tlabels, tout,
                             is_multilabel), metrics.auc(tlabels, tout, is_multilabel)
        )
        self.t_acc = append_to_dataframe(self.t_acc, tacc)
        self.t_auc = append_to_dataframe(self.t_auc, tauc)
        if vlabels is not None:
            vacc, vauc = (
                metrics.accuracy(vlabels, vout, is_multilabel),
                metrics.auc(vlabels, vout, is_multilabel)
            )
            self.v_acc = append_to_dataframe(self.v_acc, vacc)
            self.v_auc = append_to_dataframe(self.v_auc, vauc)

        # Don't predict unless the last 5 train auc is bigger than 10%
        # In case the dataset is multiclass use acc instead
        if self.t_acc.size < 5 or self.t_acc.iloc[-5:].mean()[0] < 0.1:
            return TRAIN
        if self.t_auc.size < 5 or self.t_auc.iloc[-5:].mean()[0] < 0.1:
            return TRAIN
        # Seen all classes at least once
        # NOTE(Philipp): What about multilabel cases?
        if np.any(self.labels_seen < 1):
            return TRAIN
        # If prev. conditions are fullfilled and it's the first train
        # make a prediction
        if len(autodl_model.test_time) == 0:
            return PREDICT_AND_VALIDATE
        ct_diff = (
            transform_time_abs(time.time() - autodl_model.birthday) -
            transform_time_abs(t_start - autodl_model.birthday)
        )
        if ct_diff < self.t_diff:
            return TRAIN
        if self.v_acc.size > 3 and self.v_acc.iloc[-3:].mean()[0] > 0.4:
            autodl_model.model.eval(
            )  # This will be preserved until the next train/eval step
            return PREDICT
        return PREDICT_AND_VALIDATE


# def get_ema(err_df, ema_win):
#     # If there aren't enough elements shrink the window which is only possible with at least 2
#     # errors to compare.
#     return err_df.ewm(
#         span=np.min([err_df.size - 1, ema_win]),
#         min_periods=np.min([err_df.size - 1, ema_win])
#     ).mean()

# def check_ema_improvement(err, ema, threshold):
#     # Convenience function to increase readability
#     # If threshold == 0 this boils down to lesser operation
#     return (err.iloc[-1, 0] / ema.iloc[-1, 0] < 1 - threshold).all()

# def check_ema_improvement_min(err, ema, threshold):
#     # Convenience function to increase readability
#     # If threshold == 0 this boils down to lesser operation
#     return (err.iloc[-1, 0] / np.min(ema.iloc[:-1, 0]) < 1 - threshold).all()


# ########################################################
# Helpers
# ########################################################
def append_to_dataframe(frame, val):
    # Convenience function to increase readability
    return frame.append([val.tolist()], ignore_index=True)


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
