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
                abort, final = precheck(autodl_model, t_train, remaining_time)
                if abort:
                    make_final_prediction = final
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
        self.train_err = append_loss(self.train_err, loss)

        # The first 22 batches just train and make a prediction
        if self.batch_counter <= 51 or (
            autodl_model.num_test_samples > 1000 and self.batch_counter <= 101
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
    def __init__(self, t_diff, validate_every):
        self.batch_counter = 0
        self.ele_counter = 0  # keep track of how many batches we trained on
        self.train_err = pd.DataFrame()  # collect train error
        self.train_time = 0
        self.dl_train = None

        self.t_diff = t_diff
        self.validate_every = validate_every

        self.validation_idxs = []

        self.train_acc = pd.DataFrame()
        self.valation_acc = pd.DataFrame()

    def __call__(self, autodl_model, remaining_time):
        '''
        This is called from the model.py and just seperates the
        training routine from the unchaning code
        '''
        if self.validate_every > 0 and len(self.validation_idxs) == 0:
            num_evals = int(1 / self.validate_every)
            self.validation_idxs = np.linspace(
                0, autodl_model.num_train_samples - 1, num_evals + 1, dtype=int
            )[1:-1]
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
                abort, final = precheck(autodl_model, t_train, remaining_time)
                if abort:
                    make_final_prediction = final
                    break

                data = data.to(DEVICE)
                labels = labels.to(DEVICE)

                # If the current batch is a validation batch we validate and continue
                if self.dl_train.dataset.current_idx in self.validation_idxs:
                    out, loss = eval_step(
                        autodl_model.model, autodl_model.loss_fn, data, labels
                    )
                    val_acc = accuracy(labels, out, self.dl_train.dataset.is_multilabel)
                    self.val_acc.append(val_acc, ignore_index=True)

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
                self.train_acc.append(train_acc, ignore_index=True)

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
        self.train_err = append_loss(self.train_err, loss)

        # The first 22 batches just train and make a prediction
        if self.batch_counter <= 51 or (
            autodl_model.num_test_samples > 1000 and self.batch_counter <= 101
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


def append_loss(err_list, loss):
    # Convenience function to increase readability
    return err_list.append([loss.detach().cpu().tolist()], ignore_index=True)


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


################################################################################
# From the scoring program
################################################################################
# Metric used to compute the score of a point on the learning curve
def autodl_auc(solution, prediction, valid_columns_only=True):
    """Compute normarlized Area under ROC curve (AUC).
    Return Gini index = 2*AUC-1 for  binary classification problems.
    Should work for a vector of binary 0/1 (or -1/1)"solution" and any discriminant values
    for the predictions. If solution and prediction are not vectors, the AUC
    of the columns of the matrices are computed and averaged (with no weight).
    The same for all classification problems (in fact it treats well only the
    binary and multilabel classification problems). When `valid_columns` is not
    `None`, only use a subset of columns for computing the score.
    """
    if valid_columns_only:
        valid_columns = get_valid_columns(solution)
        if len(valid_columns) < solution.shape[-1]:
            LOGGER.warning(
                "Some columns in solution have only one class, " +
                "ignoring these columns for evaluation."
            )
        solution = solution[:, valid_columns].copy()
        prediction = prediction[:, valid_columns].copy()
    label_num = solution.shape[1]
    auc = np.empty(label_num)
    for k in range(label_num):
        r_ = tiedrank(prediction[:, k])
        s_ = solution[:, k]
        if sum(s_) == 0:
            LOGGER.warning("WARNING: no positive class example in class {}".format(k + 1))
        npos = sum(s_ == 1)
        nneg = sum(s_ < 1)
        auc[k] = (sum(r_[s_ == 1]) - npos * (npos + 1) / 2) / (nneg * npos)
    return 2 * mvmean(auc) - 1


def autodl_accuracy(solution, prediction):
    """Get accuracy of 'prediction' w.r.t true labels 'solution'."""
    epsilon = 1e-15
    # normalize prediction
    prediction_normalized =\
        prediction / (np.sum(np.abs(prediction), axis=1, keepdims=True) + epsilon)
    return np.sum(solution * prediction_normalized) / solution.shape[0]


def get_valid_columns(solution):
    """Get a list of column indices for which the column has more than one class.
    This is necessary when computing BAC or AUC which involves true positive and
    true negative in the denominator. When some class is missing, these scores
    don't make sense (or you have to add an epsilon to remedy the situation).

    Args:
        solution: array, a matrix of binary entries, of shape
        (num_examples, num_features)
    Returns:
        valid_columns: a list of indices for which the column has more than one
        class.
    """
    num_examples = solution.shape[0]
    col_sum = np.sum(solution, axis=0)
    valid_columns = np.where(
        1 - np.isclose(col_sum, 0) - np.isclose(col_sum, num_examples)
    )[0]
    return valid_columns


def is_one_hot_vector(x, axis=None, keepdims=False):
    """Check if a vector 'x' is one-hot (i.e. one entry is 1 and others 0)."""
    norm_1 = np.linalg.norm(x, ord=1, axis=axis, keepdims=keepdims)
    norm_inf = np.linalg.norm(x, ord=np.inf, axis=axis, keepdims=keepdims)
    return np.logical_and(norm_1 == 1, norm_inf == 1)


def is_multiclass(solution):
    """Return if a task is a multi-class classification task, i.e.  each example
    only has one label and thus each binary vector in `solution` only has
    one '1' and all the rest components are '0'.

    This function is useful when we want to compute metrics (e.g. accuracy) that
    are only applicable for multi-class task (and not for multi-label task).

    Args:
        solution: a numpy.ndarray object of shape [num_examples, num_classes].
    """
    return all(is_one_hot_vector(solution, axis=1))


def tiedrank(a):
    ''' Return the ranks (with base 1) of a list resolving ties by averaging.
     This works for numpy arrays.'''
    m = len(a)
    # Sort a in ascending order (sa=sorted vals, i=indices)
    i = a.argsort()
    sa = a[i]
    # Find unique values
    uval = np.unique(a)
    # Test whether there are ties
    R = np.arange(m, dtype=float) + 1  # Ranks with base 1
    if len(uval) != m:
        # Average the ranks for the ties
        oldval = sa[0]
        newval = sa[0]
        k0 = 0
        for k in range(1, m):
            newval = sa[k]
            if newval == oldval:
                # moving average
                R[k0:k + 1] = R[k - 1] * (k - k0) / (k - k0 + 1) + R[k] / (k - k0 + 1)
            else:
                k0 = k
                oldval = newval
    # Invert the index
    S = np.empty(m)
    S[i] = R
    return S


def mvmean(R, axis=0):
    ''' Moving average to avoid rounding errors. A bit slow, but...
    Computes the mean along the given axis, except if this is a vector, in which case the mean is returned.
    Does NOT flatten.'''
    if len(R.shape) == 0:
        return R

    def average(x):
        reduce(
            lambda i, j: (0, (j[0] / (j[0] + 1.)) * i[1] + (1. / (j[0] + 1)) * j[1]),
            enumerate(x)
        )[1]

    R = np.array(R)
    if len(R.shape) == 1:
        return average(R)
    if axis == 1:
        return np.array(map(average, R))
    else:
        return np.array(map(average, R.transpose()))
