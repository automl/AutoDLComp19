import logging
import subprocess
import time

import numpy as np
import torch
from utils import (  # noqa: F401
    DEVICE, LOGDIR, LOGGER, PREDICT, PREDICT_AND_VALIDATE, SW, TRAIN, VALIDATE,
    memprofile, metrics, transform_time_abs
)


class PolicyTrainer():
    def __init__(self, policy_fn=None):
        self.batch_counter = 0
        self.ele_counter = 0  # keep track of how many batches we trained on
        self.dloader = None
        self.validation_idxs = []
        self.validation_buffer = []
        self.validation_class_dis = None

        # Default policy
        self.check_policy = grid_check_policy(0.02) if policy_fn is None else policy_fn

    @memprofile(precision=2)
    def __call__(self, autodl_model, remaining_time):
        '''
        The policy trainer executes training/validation/prediction making
        according to the given policy-function which returns a pair of bools.
        The first tells the trainer to stop training and to make a prediction
        if true. The second tells the trainer to do a round of validation
        and skip training the current batch.

        If no policy is given, the trainer defaults to a grid prediction policy
        making predictions every 2% on the log time scale
        '''
        self.dloader = autodl_model.train_dl

        t_train = time.time(
        ) if autodl_model.training_round > 0 else autodl_model.birthday
        batch_counter_start = self.batch_counter
        batch_loading_time = 0
        validate = False
        make_prediction = False
        make_final_prediction = False
        while not make_prediction:
            load_start = time.time()
            for i, (data, labels) in enumerate(self.dloader):
                batch_loading_time += time.time() - load_start
                # On second train round gather some validation data

                # Run prechecks whether we abort training or not (earlystop or final pred.)
                make_prediction, make_final_prediction = precheck(
                    autodl_model, t_train, remaining_time
                )
                if make_prediction:
                    break
                if self.dloader.current_idx in self.validation_idxs:
                    continue

                # Copy data to DEVICE for training
                self.batch_counter += 1
                self.ele_counter += np.prod(data.shape[0:2])

                # Train on a batch if we re good to go
                data = data.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                out, loss = train_step(
                    autodl_model.model, autodl_model.loss_fn, data, labels,
                    autodl_model.optimizer, autodl_model.lr_scheduler
                )
                labels, out, loss = (
                    labels.cpu().numpy(), out.detach().cpu().numpy(),
                    loss.detach().cpu().numpy()
                )
                train_acc, train_auc = (
                    metrics.accuracy(labels, out, self.dloader.dataset.is_multilabel),
                    metrics.auc(labels, out, self.dloader.dataset.is_multilabel)
                )
                LOGGER.debug(
                    'STEP #{0}\tTRAINED BATCH #{1}:\t{2:.6f}\t{3:.2f}\t{4:.2f}'.format(
                        i, self.dloader.current_idx, loss, train_acc, train_auc
                    )
                )
                SW.add_scalar('Train_Loss', loss, self.batch_counter)
                SW.add_scalar('Train_Acc', train_acc, self.batch_counter)
                SW.add_scalar('Train_Auc', train_auc, self.batch_counter)
                SW.add_histogram(
                    'Classes',
                    labels if len(labels.shape) == 1 else np.argwhere(labels > 0)[:, 1],
                    self.batch_counter
                )

                vlabels, vout, vloss = None, None, None
                if validate:
                    # Under construction
                    llabels, lout, lloss = [], [], []
                    for tdata, tlabels in self.validation_buffer:
                        tdata = tdata.to(DEVICE, non_blocking=True)
                        tlabels = tlabels.to(DEVICE, non_blocking=True)
                        tout, tloss = eval_step(
                            autodl_model.model, autodl_model.loss_fn, tdata, tlabels
                        )
                        tlabels, tout, tloss = (
                            tlabels.cpu().numpy(), tout.detach().cpu().numpy(),
                            tloss.detach().cpu().numpy()
                        )
                        llabels.append(tlabels)
                        lout.append(tout)
                        lloss(tloss)
                    vlabels, vout, vloss = (
                        np.vstack(llabels), np.vstack(lout), np.vstack(lloss)
                    )
                    val_acc, val_auc = (
                        metrics.accuracy(
                            vlabels, vout, self.dloader.dataset.is_multilabel
                        ), metrics.auc(vlabels, vout, self.dloader.dataset.is_multilabel)
                    )
                    LOGGER.debug(
                        'STEP #{0} VALIDATED ON BATCH #{1}:\t{2:.6f}\t{3:.2f}\t{4:.2f}'.
                        format(i, self.dloader.current_idx, vloss, val_acc, val_auc)
                    )
                    SW.add_scalar('Valid_Loss', vloss, self.batch_counter)
                    SW.add_scalar('Valid_Acc', val_acc, self.batch_counter)
                    SW.add_scalar('Valid_Auc', val_auc, self.batch_counter)
                    validate = False

                # Check if we want to make a prediction/validate after next train or not
                make_prediction, validate = self.check_policy(
                    autodl_model, t_train, remaining_time - (time.time() - t_train),
                    labels, out, loss, vlabels, vout, vloss
                )

                if make_prediction:
                    break
                load_start = time.time()

        if LOGGER.level == logging.DEBUG:
            subprocess.run(['nvidia-smi'])
        LOGGER.debug(
            "MEAN FRAMES PER SEC TRAINED:\t{0:.2f}".format(
                self.ele_counter / (time.time() - autodl_model.birthday)
            )
        )
        if (self.batch_counter - batch_counter_start) > 0:
            LOGGER.debug(
                'SEC PER BATCH LOADING:\t{0:.4f}'.format(
                    batch_loading_time / (self.batch_counter - batch_counter_start)
                )
            )
            LOGGER.debug('SEC TOTAL DATA LOADING:\t{0:.4f}'.format(batch_loading_time))
        else:
            LOGGER.debug('NO BATCH PROCESSED')
        return make_final_prediction


class grid_check_policy():
    '''
    This is the default policy if no other is chosen
    it makes a prediction every t_diff percent along
    the log timescale
    '''
    def __init__(self, t_diff):
        self.t_diff = t_diff
        self.batch_counter = 0

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
        self.batch_counter += 1
        # The first 22 batches just train and make a prediction
        if self.batch_counter <= 21:
            return TRAIN
        else:
            if autodl_model.testing_round == 0:
                return PREDICT
            ct_diff = (
                transform_time_abs(time.time() - autodl_model.birthday) -
                transform_time_abs(t_start - autodl_model.birthday)
            )
            if ct_diff < self.t_diff:
                return TRAIN
            else:
                return PREDICT
        return TRAIN


# ########################################################
# Helpers
# ########################################################
def get_time_wo_final_prediction(autodl_model, remaining_time, train_start):
    '''
    Calculate current remaining time - the time to make a prediction,
    approximated by the last 3 testing durations' average
    '''
    return remaining_time - (
        time.time() - train_start + np.mean(autodl_model.test_time[-3:]) +
        np.std(autodl_model.test_time[-3:]) - 10
    ) if len(autodl_model.test_time) > 3 else None


def precheck(autodl_model, t_train, remaining_time):
    '''
    Checks wether or not there is enough time left to train.
    If not a final prediction request is returned
    '''
    make_prediction, make_final_prediction = False, False
    # Check if we need to early stop according to the config's earlystop
    if (
        autodl_model.config.earlystop is not None and
        time.time() - autodl_model.birthday > autodl_model.config.earlystop
    ):
        make_prediction = True
    # Abort training and make final prediction if not enough time is left
    t_left = get_time_wo_final_prediction(autodl_model, remaining_time, t_train)
    if t_left is not None and t_left < 5:
        LOGGER.info('Making final prediction!')
        make_final_prediction = True
        make_prediction = True
    return make_prediction, make_final_prediction


def train_step(model, criterion, data, labels, optimizer, lr_scheduler):
    '''
    Executes a single train step
    It expects the tensors given to be already cast to the target device
    '''
    model.train()
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()
    if lr_scheduler is not None:
        lr_scheduler.step()
    return output, loss


def eval_step(model, criterion, data, labels):
    '''
    Executes a evaluation train step
    It expects the tensors given to be already cast to the target device
    '''
    with torch.no_grad():
        model.eval()
        output = model(data)
        loss = criterion(output, labels)
        return output, loss
