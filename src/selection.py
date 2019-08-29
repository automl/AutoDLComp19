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
        model_name, checkpoint_file = ('bninception', 'BnT_Video_input_128.pth.tar')
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
            transformations.video, conf['transformations']
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

        self.train_err = pd.DataFrame()
        self.train_acc = pd.DataFrame()
        self.valid_acc = pd.DataFrame()

        self.labels_seen = np.zeros((num_classes, ))

    def __call__(
        self, model, test_num_samples, predictions_made, r_budget, t_start, birthday, out,
        labels, loss, acc, val_acc
    ):
        self.train_err = append_to_dataframe(self.train_err, loss)
        self.train_acc = append_to_dataframe(self.train_acc, acc)
        if val_acc is not None:
            self.valid_acc = append_to_dataframe(self.valid_acc, val_acc)
        self.labels_seen += labels.sum(axis=0)

        # The first 5 batches just train
        if self.train_acc.size < 5:
            return False, False
        # Don't predict unless train acc is bigger than 10%
        if self.train_acc.iloc[-5:].mean()[0] < 0.1:
            return False, False
        # Seen all classes at least 10 times
        # NOTE(Philipp): What about multilabel cases?
        if np.all(self.labels_seen < 5):
            return False, False
        # If prev. conditions are fullfilled and it's the first train
        # make a prediction
        if predictions_made == 0:
            return True, False
        if self.valid_acc.size > 3 and self.valid_acc.iloc[-3:].mean()[0] > 0.4:
            model.eval()  # This will be preserved until the next train/eval step
            return True, False
        return True, False


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
    return frame.append([val.detach().cpu().tolist()], ignore_index=True)


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
