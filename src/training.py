import time
import logging
import subprocess
import pandas as pd
import numpy as np
import torch
from functools import reduce
from utils import LOGGER, DEVICE


class baseline_trainer():
    def __init__(self, t_diff):
        self.t_diff = t_diff

        self.train_err = pd.DataFrame()  # collect train error

        self.batch_counter = 0
        self.ele_counter = 0  # keep track of how many batches we trained on

        self.train_time = 0

    def __call__(self, autodl_model, remaining_time):
        '''
        This is called from the model.py and just seperates the
        training routine from the unchaning code
        '''
        # This is one way to split the tedious stuff from
        # making the decision to continue training or not
        # Maybe move this stuff and just define a policy api?
        LOGGER.info("TRAINING COUNTER:\t" + str(self.ele_counter))

        dl_train = autodl_model.train_dl['train']
        dl_val = autodl_model.train_dl['val']

        t_train = time.time() if autodl_model.training_round > 0 else autodl_model.birthday
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
            for i, (data, labels) in enumerate(dl_train):
                batch_loading_time += time.time() - load_start
                # Check if we need to early stop according to the config's earlystop
                if (
                    autodl_model.config.earlystop is not None
                    and time.time() - autodl_model.birthday > autodl_model.config.earlystop
                ):
                    make_prediction = True
                    break

                # Abort training and make final prediction if not enough time is left
                t_left = get_time_wo_final_prediction(
                    remaining_time,
                    t_train,
                    autodl_model
                )
                if t_left is not None and t_left < 5:
                    LOGGER.info('Making final prediciton!')
                    make_final_prediction = True
                    make_prediction = True
                    break

                # Train on a batch if we re good to go
                _, loss = train_step(
                    autodl_model.model,
                    autodl_model.optimizer,
                    autodl_model.loss_fn,
                    autodl_model.lr_scheduler,
                    data.to(DEVICE),
                    labels.to(DEVICE)
                )
                LOGGER.debug('TRAINED BATCH #{0}:\t{1}'.format(i, loss))

                self.batch_counter += 1
                self.ele_counter += np.prod(data.shape[0:1])

                # Check if we want to make a prediciton or not
                if self.grid_check_policy(autodl_model, i, t_train, loss, dl_val):
                    make_prediction = True
                    break
                load_start = time.time()

        if LOGGER.level == logging.debug:
            subprocess.run(['nvidia-smi'])
        LOGGER.info('NUM_SEGMENTS:\t\t\t{0}'.format(autodl_model.model.num_segments))
        LOGGER.info('LR:\t\t\t\t{0:.5e}'.format(autodl_model.optimizer.param_groups[0]['lr']))
        LOGGER.info('DROPOUT:\t\t\t{0:.4g}'.format(autodl_model.model.dropout))
        LOGGER.info("MEAN TRAINING FRAMES PER SEC:\t{0:.2f}".format(
            self.ele_counter / (time.time() - autodl_model.birthday))
        )
        LOGGER.info("TRAINING COUNTER:\t\t" + str(self.ele_counter))
        if (self.batch_counter - batch_counter_start) > 0:
            LOGGER.debug('SEC PER BATCH LOADING:\t{0:.4f}'.format(
                batch_loading_time
                / (self.batch_counter - batch_counter_start)
            ))
            LOGGER.debug('SEC TOTAL DATA LOADING:\t{0:.4f}'.format(
                batch_loading_time
            ))
        else:
            LOGGER.info('NO BATCH PROCESSED')
        return make_final_prediction

    def grid_check_policy(self, autodl_model, i, t_train_start, loss, dl_val):
        '''
        return True - make a prediction
        return False - continue training another batch

        NOTE(Philipp): Maybe extend this with a third option - change/update model
        '''
        self.train_err = append_loss(self.train_err, loss)

        # The first 22 batches just train and make a prediction
        if self.batch_counter <= 21:
            pass
        else:
            if autodl_model.testing_round == 0:
                return True
            ct_diff = (
                transform_time_abs(time.time() - autodl_model.birthday)
                - transform_time_abs(t_train_start - autodl_model.birthday)
            )
            if ct_diff < self.t_diff:
                pass
            else:
                return True
        return False


class policy_trainer():
    def __init__(self, t_diff, dropout_diff, num_segments_step, bn_prod_limit):
        self.t_diff = t_diff
        self.dropout_diff = dropout_diff
        self.num_segments_step = num_segments_step
        self.bn_prod_limit = bn_prod_limit

        self.train_err = pd.DataFrame()  # collect train error
        self.train_ewm_window = 1  # window size of the exponential moving average
        self.val_err = pd.DataFrame()  # collect train error
        self.val_ewm_window = 1  # window size of the exponential moving average

        self.batch_counter = 0
        self.ele_counter = 0  # keep track of how many batches we trained on

        self.train_ewm_window = None  # np.ceil(split_num[0] / self.parser_args.batch_size)
        self.val_ewm_window = None  # np.ceil(split_num[0] / self.parser_args.batch_size)

        self.train_time = 0

    def __call__(self, autodl_model, remaining_time):
        self.update_hyperparams(autodl_model, autodl_model.train_dl['train'])
        self.update_batch_size(autodl_model)

        LOGGER.info("TRAINING COUNTER:\t" + str(self.ele_counter))
        LOGGER.info('BATCH SIZE:\t' + str(autodl_model.train_dl['train'].batch_size))

        dl_train = autodl_model.train_dl['train']
        dl_val = autodl_model.train_dl['val']

        t_train = time.time() if autodl_model.training_round > 0 else autodl_model.birthday
        make_prediction = False
        make_final_prediction = False
        while not make_prediction:
            # Set train mode before we go into the train loop over an epoch
            for i, (data, labels) in enumerate(dl_train):
                t_left = get_time_wo_final_prediction(
                    remaining_time,
                    t_train,
                    autodl_model
                )
                # Abort training and make final prediction if not enough time is left
                if t_left is not None and t_left < 5:
                    LOGGER.info('Making final prediciton!')
                    make_final_prediction = True
                    make_prediction = True
                    break

                _, loss = train_step(
                    autodl_model.model,
                    autodl_model.optimizer,
                    autodl_model.loss_fn,
                    autodl_model.lr_scheduler,
                    data.to(DEVICE),
                    labels.to(DEVICE)
                )
                LOGGER.debug('TRAINED BATCH #{0}:\t{1}'.format(i, loss))
                # ### Set dropout
                autodl_model.model.dropout = autodl_model.model.dropout + self.dropout_diff
                autodl_model.model.dropout = min(autodl_model.model.dropout, 0.9)
                autodl_model.model.alphadrop = torch.nn.AlphaDropout(p=autodl_model.model.dropout)

                self.batch_counter += 1
                self.ele_counter += np.prod(data.shape[0:1])

                if self.grid_check_policy(autodl_model, i, t_train, loss, dl_val):
                    make_prediction = True
                    break

        if LOGGER.level == logging.debug:
            subprocess.run(['nvidia-smi'])
        LOGGER.info('NUM_SEGMENTS:\t' + str(autodl_model.model.num_segments))
        LOGGER.info('LR:\t{0:.4e}'.format(autodl_model.optimizer.param_groups[0]['lr']))
        LOGGER.info('DROPOUT:\t{0:.4g}'.format(autodl_model.model.dropout))
        LOGGER.info("MEAN TRAINING FRAMES PER SEC:\t{0:.2f}".format(
            self.ele_counter / (time.time() - autodl_model.birthday))
        )
        LOGGER.info("TRAINING COUNTER:\t" + str(self.ele_counter))
        return make_final_prediction

    def update_batch_size(self, autodl_model):
        batch_size_max = int(self.bn_prod_limit / autodl_model.model.num_segments)
        trainloader_args = {**autodl_model.config.dataloader_args['train']}

        if autodl_model.train_dl['val'] is not None:
            trainloader_args['batch_size'] = batch_size_max
            autodl_model.train_dl['val'] = torch.utils.data.DataLoader(
                autodl_model.train_dl['val'].dataset,
                **trainloader_args
            )

        batch_size_des = autodl_model.config.dataloader_args['train']['batch_size']
        trainloader_args['batch_size'] = min(batch_size_des, batch_size_max)
        autodl_model.train_dl['train'] = torch.utils.data.DataLoader(
            autodl_model.train_dl['train'].dataset,
            **trainloader_args
        )

    def update_hyperparams(self, autodl_model, dl):
        # ### Set number segments
        num_segments = 1
        if dl.dataset.max_shape[0] > 1:
            # video dataset
            num_segments = 2**int(self.ele_counter / self.num_segments_step + 1)
            avg_frames = dl.dataset.mean_shape[0]
            if avg_frames > 64:
                upper_limit = 16
            else:
                upper_limit = 8
            num_segments = min(max(num_segments, 2), upper_limit)
            autodl_model.model.num_segments = num_segments

    def validation_check_policy(self, autodl_model, i, t_train_start, loss, dl_val):
        make_prediction = False
        self.train_err = append_loss(self.train_err, loss)
        LOGGER.info('TRAIN BATCH #{0}:\t{1}'.format(i, loss))

        t_current = time.time() - t_train_start
        # The first 15 seconds just train and make a prediction
        if t_current <= 15:
            pass
        elif (
            autodl_model.config.earlystop is not None
            and time.time() - autodl_model.birthday > autodl_model.config.earlystop
        ):
            make_prediction = True
        elif t_current < 300 and dl_val is not None:
            if autodl_model.testing_round == 0:
                make_prediction = True
            # The first 5min do grid-like predictions
            ct_diff = (
                transform_time_abs(time.time() - autodl_model.birthday)
                - transform_time_abs(t_train_start - autodl_model.birthday)
            )
            if ct_diff < self.t_diff:
                pass
            else:
                make_prediction = True
        else:
            # If the last train error improves upon the minimum of it's exponential moving average
            # by at least 5% with window self.train_err_ewm = 5, escalate to validation.
            if self.train_err.size > 1:
                train_err_ewm = get_ema(self.train_err, self.train_ewm_window)
                if check_ema_improvement(self.train_err, train_err_ewm, 0.1):
                    self.val_err = append_loss(self.val_err, self.evaluate_on(dl_val))
                    if self.val_err.size > 1:
                        val_err_ewm = get_ema(self.val_err, self.val_ewm_window)
                        LOGGER.info('VALIDATION EMA:\t{0}'.format(val_err_ewm.iloc[-1, 0]))
                        LOGGER.info('VALIDATION ERR:\t{0}'.format(self.val_err.iloc[-1, 0]))
                        LOGGER.info('VALIDATION MIN:\t{0}'.format(np.min(self.val_err.iloc[:-1, 0])))
                        # If the last validation error improves upon the minimum of it's exponential moving average
                        # by 10% with window self.train_err_ewm = 5, escalate to test.
                        # if self.check_ema_improvement(self.val_err, val_err_ewm, 0.1):
                        if check_ema_improvement_min(self.val_err, val_err_ewm, 0.15):
                            make_prediction = True
                        LOGGER.info('BACK TO THE GYM')
        return make_prediction

    def grid_check_policy(self, autodl_model, i, t_train_start, loss, dl_val):
        make_prediction = False
        self.train_err = append_loss(self.train_err, loss)

        # The first 20 batches just train and make a prediction
        if self.batch_counter <= 21:
            pass
        elif (
            autodl_model.config.earlystop is not None
            and time.time() - autodl_model.birthday > autodl_model.config.earlystop
        ):
            make_prediction = True
        else:
            if autodl_model.testing_round == 0:
                make_prediction = True
            ct_diff = (
                transform_time_abs(time.time() - autodl_model.birthday)
                - transform_time_abs(t_train_start - autodl_model.birthday)
            )
            if ct_diff < self.t_diff:
                pass
            else:
                make_prediction = True
        return make_prediction


# ########################################################
# Helpers
# ########################################################
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


def eval_step(model, criterion, data, labels):
    model.eval()
    output = model(data)
    loss = criterion(output, labels)

    return output, loss


def transform_time_abs(t_abs):
    '''
    conversion from absolute time 0s-1200s to relative time 0-1
    '''
    return np.log(1 + t_abs / 60.0) / np.log(21)


def transform_time_rel(t_rel):
    '''
    convertsion from relative time 0-1 to absolute time 0s-1200s
    '''
    return 60 * (21 ** t_rel - 1)


def get_time_wo_final_prediction(remaining_time, train_start, model):
            # Calculate current remaining time - time to make a prediction
    return remaining_time - (
        time.time() - train_start
        + np.mean(model.test_time[-3:])
        + np.std(model.test_time[-3:])
    ) if len(model.test_time) > 3 else None


def append_loss(err_list, loss):
    # Convenience function to increase readability
    return err_list.append(
        [loss.detach().cpu().tolist()],
        ignore_index=True
    )


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
    return (
        err.iloc[-1, 0] / ema.iloc[-1, 0]
        < 1 - threshold
    ).all()


def check_ema_improvement_min(err, ema, threshold):
    # Convenience function to increase readability
    # If threshold == 0 this boils down to lesser operation
    return (
        err.iloc[-1, 0] / np.min(ema.iloc[:-1, 0])
        < 1 - threshold
    ).all()


def evaluate_on(model, dl_val):
    err = np.Inf
    dl_val.dataset.reset()
    with torch.no_grad():
        for i, (vdata, vlabels) in enumerate(dl_val):
            _, loss = eval_step(
                model.model,
                model.loss_fn,
                vdata,
                vlabels
            )
            err = loss if np.isinf(err) else err + loss
    return err


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
                "Some columns in solution have only one class, "
                + "ignoring these columns for evaluation.")
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


def accuracy(solution, prediction):
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
        1 - np.isclose(col_sum, 0)
        - np.isclose(col_sum, num_examples))[0]
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
            lambda i, j: (
                0,
                (j[0] / (j[0] + 1.)) * i[1] + (1. / (j[0] + 1)) * j[1]
            ),
            enumerate(x)
        )[1]
    R = np.array(R)
    if len(R.shape) == 1:
        return average(R)
    if axis == 1:
        return np.array(map(average, R))
    else:
        return np.array(map(average, R.transpose()))
