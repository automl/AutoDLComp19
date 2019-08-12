import time
import subprocess
import pandas as pd
import numpy as np
import torch
from utils import LOGGER


# Example for a trainer class
# which is like the loop but has methods and variables
# and is prefered
class default_trainer():
    def __init__(self):
        self.train_err = pd.DataFrame()  # collect train error
        self.train_ewm_window = 1  # window size of the exponential moving average
        self.val_err = pd.DataFrame()  # collect train error
        self.val_ewm_window = 1  # window size of the exponential moving average

        self.batch_counter = 0  # keep track of how many batches we trained on

        self.train_ewm_window = None  # np.ceil(split_num[0] / self.parser_args.batch_size)
        self.val_ewm_window = None  # np.ceil(split_num[0] / self.parser_args.batch_size)

        self.train_time = 0

    def __call__(self, autodl_model, remaining_time, t_diff, dropout_diff):
        LOGGER.info("TRAINING START: " + str(time.time()))
        LOGGER.info("REMAINING TIME: " + str(remaining_time))
        dl_train = autodl_model.train_dl['train']
        dl_val = autodl_model.train_dl['val']
        t_train = time.time() if autodl_model.training_round > 0 else autodl_model.birthday
        make_prediction = False
        make_final_prediction = False
        while not make_prediction:
            # Set train mode before we go into the train loop over an epoch
            dl_train.dataset.reset()
            for i, (data, labels) in enumerate(dl_train):
                t_left = get_time_wo_final_prediction(
                    remaining_time,
                    t_train,
                    autodl_model
                )
                # Abort training and make final prediction if not enough time is left
                if t_left is not None and t_left < 10:
                    LOGGER.info('Making final prediciton!')
                    make_final_prediction = True
                    make_prediction = True
                    break

                _, loss = train_step(
                    autodl_model.model,
                    autodl_model.optimizer,
                    autodl_model.loss_fn,
                    data,
                    labels
                )
                self.batch_counter += 1
                self.update_model(
                    autodl_model, dl_train.dataset,
                    dropout_diff, self.batch_counter
                )
                if self.check_policy(autodl_model, i, t_train, loss, dl_val, t_diff):
                    make_prediction = True
                    break
        subprocess.run(['nvidia-smi'])
        LOGGER.info("TRAINING END: " + str(time.time()))
        return make_final_prediction

    def check_policy(self, autodl_model, i, t_train_start, loss, dl_val, t_diff):
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
            if ct_diff < t_diff:
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
                        LOGGER.info('VALIDATION EMA: {0}'.format(val_err_ewm.iloc[-1, 0]))
                        LOGGER.info('VALIDATION ERR: {0}'.format(self.val_err.iloc[-1, 0]))
                        LOGGER.info('VALIDATION MIN: {0}'.format(np.min(self.val_err.iloc[:-1, 0])))
                        # If the last validation error improves upon the minimum of it's exponential moving average
                        # by 10% with window self.train_err_ewm = 5, escalate to test.
                        # if self.check_ema_improvement(self.val_err, val_err_ewm, 0.1):
                        if check_ema_improvement_min(self.val_err, val_err_ewm, 0.15):
                            make_prediction = True
                        LOGGER.info('BACK TO THE GYM')
        return make_prediction

    def update_model(self, autodl_model, dataset, dropout_diff, batches_seen):
        # ### Set number segments
        num_segments = 1
        if autodl_model.get_sequence_size() > 1:
            # video dataset
            # num_segments = 2**int(self.train_counter/self.parser_args.num_segments_step+1)
            num_segments = 8
            avg_frames = dataset.mean_shape[0]
            if avg_frames > 64:
                upper_limit = 16
            else:
                upper_limit = 8
            num_segments = min(max(num_segments, 2), upper_limit)

        # ### Set dropout
        autodl_model.model.dropout = autodl_model.model.dropout + dropout_diff
        autodl_model.model.dropout = min(autodl_model.model.dropout, 0.9)


# ########################################################
# Helpers
# ########################################################
def train_step(model, optimizer, criterion, data, labels):
    model.train()
    optimizer.zero_grad()
    output = model(data.cuda())
    loss = criterion(output, labels.cuda())
    loss.backward()
    optimizer.step()

    return output, loss


def eval_step(model, criterion, data, labels):
    model.eval()
    output = model(data.cuda())
    loss = criterion(output, labels.cuda())

    return output, loss


def get_batch_size(num_segments, istraining):
    if is_training:
        bn_prod_des = self.parser_args.batch_size_train*num_segments
        if bn_prod_des <= self.parser_args.bn_prod_limit:
            batch_size = self.parser_args.batch_size_train
        else:
            batch_size = int(self.parser_args.bn_prod_limit / num_segments)
    else:
        batch_size = int(self.parser_args.bn_prod_limit / num_segments)

    logger.info('SET BATCH SIZE: ' + str(batch_size))

    return batch_size


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
        + np.mean(model.test_time)
        + np.std(model.test_time)
    ) if len(model.test_time) > 0 else None


def append_loss(self, err_list, loss):
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
    val_err = np.Inf
    dl_val.dataset.reset()
    with torch.no_grad():
        for i, (vdata, vlabels) in enumerate(dl_val):
            _, loss = eval_step(
                model.model,
                model.loss_fn,
                vdata,
                vlabels
            )
            val_err = loss if np.isinf(val_err) else val_err + loss
    return val_err


def accuracy(output, labels):
    # Right choices +1 wrong choises -1
    # If multiclass we use a sigmoid
    # If multilabel ?
    pass
