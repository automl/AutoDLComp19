import logging
import subprocess
import time

import numpy as np
import torch
from sklearn import metrics
from utils import DEVICE, LOGGER

PREDICT_AND_VALIDATE = (True, True)
PREDICT = (True, False)
VALIDATE = (False, True)
TRAIN = (False, False)


class PolicyTrainer():
    def __init__(self, validation_buffer, policy_fn=None):
        self.batch_counter = 0
        self.ele_counter = 0  # keep track of how many batches we trained on
        self.dloader = None
        self.validation_idxs = []

        # Default policy
        self.check_policy = grid_check_policy(0.02) if policy_fn is None else policy_fn

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
            # Uncomment the next line to always start from the beginning
            # although we need to decide if we want to reshuffle or
            # just continue where we left of.
            # The controlling factor is the tfdataset inside the TFDataset object
            load_start = time.time()
            for i, (data, labels) in enumerate(self.dloader):
                batch_loading_time += time.time() - load_start

                # Run prechecks whether we abort training or not (earlystop or final pred.)
                make_prediction, make_final_prediction = precheck(
                    autodl_model, t_train, remaining_time
                )
                if make_prediction:
                    break

                # Copy data to DEVICE for training
                data = data.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                # If the current batch is a validation batch we validate and continue
                train_acc = None
                val_acc = None
                train_auc = None
                val_auc = None

                # Train on a batch if we re good to go
                out, loss = train_step(
                    autodl_model.model, autodl_model.optimizer, autodl_model.loss_fn,
                    autodl_model.lr_scheduler, data, labels
                )
                labels, out, loss = (
                    labels.cpu().numpy(), out.detach().cpu().numpy(),
                    loss.detach().cpu().numpy()
                )

                if validate:
                    raise NotImplementedError
                    # Under construction
                    vdata, vlabels = None, None
                    vout, vloss = eval_step(
                        autodl_model.model, autodl_model.loss_fn, data, labels
                    )
                    vout, vloss = (
                        vout.detach().cpu().numpy(), vloss.detach().cpu().numpy()
                    )
                    val_acc = accuracy(vlabels, vout, self.dloader.dataset.is_multilabel)
                    val_auc = auc(vlabels, vout, self.dloader.dataset.is_multilabel)
                    LOGGER.debug(
                        'STEP #{0} VALIDATED ON BATCH #{1}:\t{2:.6f}\t{3:.2f}\t{4:.2f}'.
                        format(
                            i, self.dloader.current_idx, vloss, val_acc * 100, val_auc
                        )
                    )
                    validate = False

                self.batch_counter += 1
                self.ele_counter += np.prod(data.shape[0:2])
                # ################# No Torch Tensors from here onwards #################
                train_acc = accuracy(labels, out, self.dloader.dataset.is_multilabel)
                train_auc = auc(labels, out, self.dloader.dataset.is_multilabel)
                LOGGER.debug(
                    'STEP #{0} TRAINED BATCH #{1}:\t{2:.6f}\t{3:.2f}\t{4:.2f}'.format(
                        i, self.dloader.current_idx, loss, train_acc * 100, train_auc
                    )
                )

                onehot_labels = labels
                if labels.shape != out.shape:
                    onehot_labels = np.zeros_like(out, dtype=int)
                    onehot_labels[np.arange(len(out)), labels] = 1

                # Check if we want to make a prediction/validate after next train or not
                make_prediction, validate = self.check_policy(
                    autodl_model.model, autodl_model.test_num_samples,
                    autodl_model.testing_round, remaining_time - (time.time() - t_train),
                    t_train, autodl_model.birthday, out, onehot_labels, loss, train_acc,
                    val_acc
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
        self, model, test_num_samples, predictions_made, r_budget, t_start, birthday, out,
        labels, loss, acc, val_acc
    ):
        self.batch_counter += 1
        # The first 22 batches just train and make a prediction
        if self.batch_counter <= 21 or (
            test_num_samples > 1000 and self.batch_counter <= 51
        ):
            return TRAIN
        else:
            if predictions_made == 0:
                return PREDICT
            ct_diff = (
                transform_time_abs(time.time() - birthday) -
                transform_time_abs(t_start - birthday)
            )
            if ct_diff < self.t_diff:
                return TRAIN
            else:
                return PREDICT
        return TRAIN


# ########################################################
# Helpers
# ########################################################
def transform_time_abs(t_abs):
    '''
    conversion from absolute time 0s-1200s to relative time 0-1
    '''
    return np.log(1 + t_abs / 60.0) / np.log(21)


def transform_time_rel(t_rel):
    '''
    convertsion from relative time 0-1 to absolute time 0s-1200s
    '''
    return 60 * (21**t_rel - 1)


def get_time_wo_final_prediction(remaining_time, train_start, model):
    '''
    Calculate current remaining time - the time to make a prediction,
    approximated by the last 3 testing durations' average
    '''
    return remaining_time - (
        time.time() - train_start + np.mean(model.test_time[-3:]) +
        np.std(model.test_time[-3:]) - 10
    ) if len(model.test_time) > 3 else None


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
    t_left = get_time_wo_final_prediction(remaining_time, t_train, autodl_model)
    if t_left is not None and t_left < 5:
        LOGGER.info('Making final prediction!')
        make_final_prediction = True
        make_prediction = True
    return make_prediction, make_final_prediction


def accuracy(labels, out, multilabel):
    '''
    Returns the TPR
    '''
    if multilabel:
        out = (out > 0).astype(int)
    else:
        ooh = np.zeros_like(out)
        ooh[np.arange(len(out)), np.argmax(out, axis=1)] = 1
        out = ooh
        loh = np.zeros_like(out)
        loh[np.arange(len(labels)), labels] = 1
        labels = loh
    nout = out / (np.sum(np.abs(out), axis=1, keepdims=True) + 1e-15)
    true_pos = (labels * nout).sum() / float(labels.shape[0])
    return true_pos


def auc(labels, out, multilabel):
    '''
    Returns the average auc-roc score
    '''
    roc_auc = -1
    if not multilabel:
        loh = np.zeros_like(out)
        loh[np.arange(len(out)), labels] = 1
        ooh = out

        # Remove classes without positive examples
        col_to_keep = (loh.sum(axis=0) > 0)
        loh = loh[:, col_to_keep]
        ooh = out[:, col_to_keep]

        fpr, tpr, _ = metrics.roc_curve(loh.ravel(), ooh.ravel())
        roc_auc = metrics.auc(fpr, tpr)
    else:
        loh = labels
        ooh = out

        # Remove classes without positive examples
        col_to_keep = (loh.sum(axis=0) > 0)
        loh = loh[:, col_to_keep]
        ooh = out[:, col_to_keep]

        roc_auc = metrics.roc_auc_score(loh, ooh, average='macro')
    return 2 * roc_auc - 1


def train_step(model, optimizer, criterion, lr_scheduler, data, labels):
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


def evaluate_on(model, dl_val):
    '''
    Evaluates the given model on a dataset given as a dataloader
    '''
    err = np.Inf
    dl_val.reset()
    for i, (vdata, vlabels) in enumerate(dl_val):
        _, loss = eval_step(
            model.model, model.loss_fn, vdata.to(DEVICE, non_blocking=True),
            vlabels.to(DEVICE, non_blocking=True)
        )
        err = loss if np.isinf(err) else err + loss
    return err


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
