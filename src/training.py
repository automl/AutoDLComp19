import logging
import subprocess
import time

import numpy as np
import torch
from utils import DEVICE, LOGGER


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

                data = data.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                # If the current batch is a validation batch we validate and continue
                train_acc = None
                val_acc = None
                # Train on a batch if we re good to go
                out, loss = train_step(
                    autodl_model.model, autodl_model.optimizer, autodl_model.loss_fn,
                    autodl_model.lr_scheduler, data, labels
                )
                train_acc = accuracy(labels, out, self.dloader.dataset.is_multilabel)
                LOGGER.debug(
                    'STEP #{0} TRAINED BATCH #{1}:\t{2:.6f}\t{3:.2f}'.format(
                        i, self.dloader.current_idx, loss, train_acc * 100
                    )
                )
                self.batch_counter += 1
                self.ele_counter += np.prod(data.shape[0:1])

                onehot_labels = np.zeros(
                    (len(labels), self.dloader.dataset.num_classes), dtype=int
                )
                onehot_labels[np.arange(labels.shape[0]), labels.cpu().numpy()] = 1

                if validate:
                    raise NotImplementedError
                    # Under construction
                    vout, vloss = eval_step(
                        autodl_model.model, autodl_model.loss_fn, data, labels
                    )
                    val_acc = accuracy(labels, vout, self.dloader.dataset.is_multilabel)
                    LOGGER.debug(
                        'STEP #{0} VALIDATED ON BATCH #{1}:\t{2:.6f}\t{3:.2f}'.format(
                            i, self.dloader.dataset.current_idx, vloss, val_acc * 100
                        )
                    )
                    validate = False

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

        if LOGGER.level == logging.debug:
            subprocess.run(['nvidia-smi'])
        LOGGER.info(
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
            LOGGER.info('NO BATCH PROCESSED')
        return make_final_prediction


class grid_check_policy():
    def __init__(self, t_diff):
        self.t_diff = t_diff
        self.batch_counter = 0

    def __call__(
        self, model, test_num_samples, predictions_made, r_budget, t_start, birthday, out,
        labels, loss, acc, val_acc
    ):
        '''
        This is the default policy if no other is chosen

        return make_a_prediction, skip_next_trainbatch_for_validation
        '''
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

        self.batch_counter += 1
        # The first 22 batches just train and make a prediction
        if self.batch_counter <= 21 or (
            test_num_samples > 1000 and self.batch_counter <= 51
        ):
            pass
        else:
            if predictions_made == 0:
                return True, False
            ct_diff = (
                transform_time_abs(time.time() - birthday) -
                transform_time_abs(t_start - birthday)
            )
            if ct_diff < self.t_diff:
                pass
            else:
                return True, False
        return False, False


# ########################################################
# Helpers
# ########################################################
def get_time_wo_final_prediction(remaining_time, train_start, model):
    # Calculate current remaining time - the time to make a prediction,
    # approximated by the last 3 testing durations' average
    return remaining_time - (
        time.time() - train_start + np.mean(model.test_time[-3:]) +
        np.std(model.test_time[-3:])
    ) if len(model.test_time) > 3 else None


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
        LOGGER.info('Making final prediction!')
        make_final_prediction = True
        make_prediction = True
    return make_prediction, make_final_prediction


def accuracy(labels, out, multilabel):
    out = out > 0 if multilabel else torch.argmax(out, dim=1)
    return labels.eq(out).sum().float() / float(labels.shape[0])


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
                model.model, model.loss_fn, vdata.to(DEVICE, non_blocking=True),
                vlabels.to(DEVICE, non_blocking=True)
            )
            err = loss if np.isinf(err) else err + loss
    return err


def eval_step(model, criterion, data, labels):
    with torch.no_grad():
        model.eval()
        output = model(data)
        loss = criterion(output, labels)
        return output, loss
