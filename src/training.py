import logging
import subprocess
import time
from functools import reduce

import numpy as np
import pandas as pd
import torch
from utils import DEVICE, LOGGER


class baseline_trainer():
    def __init__(self, t_diff):
        self.batch_counter = 0
        self.ele_counter = 0  # keep track of how many batches we trained on
        self.train_err = pd.DataFrame()  # collect train error
        self.train_time = 0
        self.dl_train = None  # just for convinience

        self.t_diff = t_diff

    def __call__(self, autodl_model, remaining_time):
        '''
        This is called from the model.py and just seperates the
        training routine from the unchaning code
        '''
        # This is one way to split the tedious stuff from
        # making the decision to continue training or not
        # Maybe move this stuff and just define a policy api?
        LOGGER.info("TRAINING COUNTER:\t" + str(self.ele_counter))

        self.dl_train = autodl_model.train_dl

        t_train = time.time(
        ) if autodl_model.training_round > 0 else autodl_model.birthday
        batch_counter_start = self.batch_counter
        batch_loading_time = 0
        make_prediction = False
        make_final_prediction = False
        while not make_prediction:
            # Uncomment the next line to always start from the beginning
            # althought we need to decide if we want to reshuffle or
            # just continue where we left of.
            # The controlling factor is the tfdataset inside the TFDataset object
            load_start = time.time()
            for i, (data, labels) in enumerate(self.dl_train):
                batch_loading_time += time.time() - load_start
                # Run prechecks whether we abort training or not (earlystop or final pred.)
                make_prediction, make_final_prediction = precheck(
                    autodl_model, t_train, remaining_time
                )
                if make_prediction:
                    break

                data = data.to(DEVICE)
                labels = labels.to(DEVICE)

                # Train on a batch if we re good to go
                out, loss = train_step(
                    autodl_model.model, autodl_model.optimizer, autodl_model.loss_fn,
                    autodl_model.lr_scheduler, data, labels
                )
                train_acc = accuracy(labels, out, self.dl_train.dataset.is_multilabel)

                LOGGER.debug(
                    'TRAINED BATCH #{0}:\t{1:.6f}\t{2:.2f}'.format(
                        i, loss, train_acc * 100
                    )
                )

                self.batch_counter += 1
                self.ele_counter += np.prod(data.shape[0:1])

                # Check if we want to make a prediciton or not
                if self.grid_check_policy(autodl_model, i, t_train, out, labels, loss):
                    make_prediction = True
                    if hasattr(autodl_model.model, 'unlock_next'):
                        autodl_model.model.unlock_next()
                    break
                load_start = time.time()

        if LOGGER.level == logging.debug:
            subprocess.run(['nvidia-smi'])
        LOGGER.info('NUM_SEGMENTS:\t\t\t{0}'.format(autodl_model.model.num_segments))
        LOGGER.info(
            'LR:\t\t\t\t{0:.5e}'.format(autodl_model.optimizer.param_groups[0]['lr'])
        )
        LOGGER.info('DROPOUT:\t\t\t{0:.4g}'.format(autodl_model.model.dropout))
        LOGGER.info(
            "MEAN TRAINING FRAMES PER SEC:\t{0:.2f}".format(
                self.ele_counter / (time.time() - autodl_model.birthday)
            )
        )
        LOGGER.info("TRAINING COUNTER:\t\t" + str(self.ele_counter))
        if (self.batch_counter - batch_counter_start) > 0:
            LOGGER.debug(
                'SEC PER BATCH LOADING:\t{0:.4f}'.format(
                    batch_loading_time / (self.batch_counter - batch_counter_start)
                )
            )
            LOGGER.debug('SEC TOTAL DATA LOADING:\t{0:.4f}'.format(batch_loading_time))
        else:
            LOGGER.info('NO BATCH PROCESSED')
        return make_final_prediction

    def grid_check_policy(self, autodl_model, i, t_train_start, out, labels, loss):
        '''
        return True - make a prediction
        return False - continue training another batch

        NOTE(Philipp): Maybe extend this with a third option - change/update model
        '''
        self.train_err = append_to_dataframe(self.train_err, loss)

        # The first 22 batches just train and make a prediction
        if self.batch_counter <= 21 or (
            autodl_model.num_test_samples > 1000 and self.batch_counter <= 51
        ):
            pass
        else:
            if autodl_model.testing_round == 0:
                return True
            ct_diff = (
                transform_time_abs(time.time() - autodl_model.birthday) -
                transform_time_abs(t_train_start - autodl_model.birthday)
            )
            if ct_diff < self.t_diff:
                pass
            else:
                return True
        return False


class validation_trainer():
    def __init__(self, t_diff, validate_on):
        self.batch_counter = 0
        self.ele_counter = 0  # keep track of how many batches we trained on
        self.train_err = pd.DataFrame()  # collect train error
        self.train_time = 0
        self.dl_train = None

        self._late_init_done = False

        self.t_diff = t_diff
        self.validate_on = validate_on

        self.validation_idxs = []

        self.train_acc = pd.DataFrame()
        self.valid_acc = pd.DataFrame()

        self.labels_seen = None

    def _late_init(self, autodl_model):
        if self.labels_seen is None:
            self.labels_seen = np.zeros(autodl_model.train_dl.dataset.num_classes)
        self._late_init_done = True

    def _should_validate(self, autodl_model):
        if self.validate_on <= 0.:
            return False
        if (
            self.dl_train.dataset.current_idx in self.validation_idxs or (
                np.random.random() < self.validate_on and len(self.validation_idxs) <
                autodl_model.num_train_samples * self.validate_on
            )
        ):
            if self.dl_train.dataset.current_idx not in self.validation_idxs:
                self.validation_idxs.append(self.dl_train.dataset.current_idx)
            return True
        return False

    def __call__(self, autodl_model, remaining_time):
        '''
        This is called from the model.py and just seperates the
        training routine from the unchaning code
        '''
        # This is one way to split the tedious stuff from
        # making the decision to continue training or not
        # Maybe move this stuff and just define a policy api?
        LOGGER.info("TRAINING COUNTER:\t" + str(self.ele_counter))
        if not self._late_init_done:
            self._late_init(autodl_model)
        self.dl_train = autodl_model.train_dl

        t_train = time.time(
        ) if autodl_model.training_round > 0 else autodl_model.birthday
        batch_counter_start = self.batch_counter
        batch_loading_time = 0
        make_prediction = False
        make_final_prediction = False
        while not make_prediction:
            # Uncomment the next line to always start from the beginning
            # althought we need to decide if we want to reshuffle or
            # just continue where we left of.
            # The controlling factor is the tfdataset inside the TFDataset object
            load_start = time.time()
            for i, (data, labels) in enumerate(self.dl_train):
                batch_loading_time += time.time() - load_start
                # Run prechecks whether we abort training or not (earlystop or final pred.)
                make_prediction, make_final_prediction = precheck(
                    autodl_model, t_train, remaining_time
                )
                if make_prediction:
                    break

                data = data.to(DEVICE)
                labels = labels.to(DEVICE)

                # If the current batch is a validation batch we validate and continue
                if self._should_validate(autodl_model):
                    out, loss = eval_step(
                        autodl_model.model, autodl_model.loss_fn, data, labels
                    )
                    val_acc = accuracy(labels, out, self.dl_train.dataset.is_multilabel)
                    self.valid_acc = append_to_dataframe(self.valid_acc, val_acc)

                    LOGGER.debug(
                        'VALIDATED ON BATCH #{0}:\t{1:.6f}\t{2:.2f}'.format(
                            i, loss, val_acc * 100
                        )
                    )
                    continue

                # Train on a batch if we re good to go
                out, loss = train_step(
                    autodl_model.model, autodl_model.optimizer, autodl_model.loss_fn,
                    autodl_model.lr_scheduler, data, labels
                )
                train_acc = accuracy(labels, out, self.dl_train.dataset.is_multilabel)
                self.train_acc = append_to_dataframe(self.train_acc, train_acc)

                LOGGER.debug(
                    'TRAINED BATCH #{0}:\t{1:.6f}\t{2:.2f}'.format(
                        i, loss, train_acc * 100
                    )
                )

                self.batch_counter += 1
                self.ele_counter += np.prod(data.shape[0:1])
                onehot_labels = np.zeros(
                    (len(labels), self.dl_train.dataset.num_classes), dtype=int
                )
                onehot_labels[np.arange(labels.shape[0]), labels.cpu().numpy()] = 1
                self.labels_seen += onehot_labels.sum(axis=0)

                # Check if we want to make a prediciton or not
                if self.grid_check_policy(autodl_model, i, t_train, out, labels, loss):
                    make_prediction = True
                    if hasattr(autodl_model.model, 'unlock_next'):
                        autodl_model.model.unlock_next()
                    break
                load_start = time.time()

        if LOGGER.level == logging.debug:
            subprocess.run(['nvidia-smi'])
        LOGGER.info('NUM_SEGMENTS:\t\t\t{0}'.format(autodl_model.model.num_segments))
        LOGGER.info(
            'LR:\t\t\t\t{0:.5e}'.format(autodl_model.optimizer.param_groups[0]['lr'])
        )
        LOGGER.info('DROPOUT:\t\t\t{0:.4g}'.format(autodl_model.model.dropout))
        LOGGER.info(
            "MEAN TRAINING FRAMES PER SEC:\t{0:.2f}".format(
                self.ele_counter / (time.time() - autodl_model.birthday)
            )
        )
        LOGGER.info("TRAINING COUNTER:\t\t" + str(self.ele_counter))
        if (self.batch_counter - batch_counter_start) > 0:
            LOGGER.debug(
                'SEC PER BATCH LOADING:\t{0:.4f}'.format(
                    batch_loading_time / (self.batch_counter - batch_counter_start)
                )
            )
            LOGGER.debug('SEC TOTAL DATA LOADING:\t{0:.4f}'.format(batch_loading_time))
        else:
            LOGGER.info('NO BATCH PROCESSED')
        return make_final_prediction

    def grid_check_policy(self, autodl_model, i, t_train_start, out, labels, loss):
        self.train_err = append_to_dataframe(self.train_err, loss)

        # The first 5 batches just train
        if self.train_acc.size < 5:
            return False
        # Don't predict unless train acc is bigger than 10%
        if self.train_acc.iloc[-5:].mean()[0] < 0.1:
            return False
        # Seen all classes at least 10 times
        # NOTE(Philipp): What about multilabel cases?
        if np.all(self.labels_seen < 5):
            return False
        # If prev. conditions are fullfilled and it's the first train
        # make a prediction
        if autodl_model.testing_round == 0:
            return True
        if self.val_acc.size > 3 and self.val_acc.iloc[-3:].mean()[0] > 0.4:
            autodl_model.model.eval()
        # Continue with grid like predictions
        ct_diff = (
            transform_time_abs(time.time() - autodl_model.birthday) -
            transform_time_abs(t_train_start - autodl_model.birthday)
        )
        return ct_diff > self.t_diff


# ########################################################
# Helpers
# ########################################################
def precheck(autodl_model, t_train, remaining_time):
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
        LOGGER.info('Making final prediciton!')
        make_final_prediction = True
        make_prediction = True
    return make_prediction, make_final_prediction


def train_step(model, optimizer, criterion, lr_scheduler, data, labels):
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
    err = np.Inf
    dl_val.dataset.reset()
    with torch.no_grad():
        for i, (vdata, vlabels) in enumerate(dl_val):
            _, loss = eval_step(
                model.model, model.loss_fn, vdata.to(DEVICE), vlabels.to(DEVICE)
            )
            err = loss if np.isinf(err) else err + loss
    return err


def eval_step(model, criterion, data, labels):
    model.eval()
    output = model(data)
    loss = criterion(output, labels)
    return output, loss


def accuracy(labels, out, multilabel):
    out = out > 0 if multilabel else torch.argmax(out, dim=1)
    return labels.eq(out).sum().float() / float(labels.shape[0])


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
    # Calculate current remaining time - time to make a prediction
    return remaining_time - (
        time.time() - train_start + np.mean(model.test_time[-3:]) +
        np.std(model.test_time[-3:])
    ) if len(model.test_time) > 3 else None


def append_to_dataframe(frame, val):
    # Convenience function to increase readability
    return frame.append([val.detach().cpu().tolist()], ignore_index=True)


def get_ema(err_df, ema_win):
    # If there aren't enough elements shrink the window which is only possible with at least 2
    # errors to compare.
    return err_df.ewm(
        span=np.min([err_df.size - 1, ema_win]),
        min_periods=np.min([err_df.size - 1, ema_win])
    ).mean()


def check_ema_improvement(err, ema, threshold):
    # Convenience function to increase readability
    # If threshold == 0 this boils down to lesser operation
    return (err.iloc[-1, 0] / ema.iloc[-1, 0] < 1 - threshold).all()


def check_ema_improvement_min(err, ema, threshold):
    # Convenience function to increase readability
    # If threshold == 0 this boils down to lesser operation
    return (err.iloc[-1, 0] / np.min(ema.iloc[:-1, 0]) < 1 - threshold).all()
