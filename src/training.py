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

        self.train_ewm_window = None  # np.ceil(split_num[0] / self.parser_args.batch_size)
        self.val_ewm_window = None  # np.ceil(split_num[0] / self.parser_args.batch_size)

        self.train_time = 0

    def transform_time_abs(self, t_abs):
        '''
        conversion from absolute time 0s-1200s to relative time 0-1
        '''
        return np.log(1 + t_abs / 60.0) / np.log(21)

    def transform_time_rel(self, t_rel):
        '''
        convertsion from relative time 0-1 to absolute time 0s-1200s
        '''
        return 60 * (21 ** t_rel - 1)

    def get_time_wo_final_prediction(self, remaining_time, train_start, model):
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

    def get_ema(self, err_df, ema_win):
        # If there aren't enough elements shrink the window which is only possible with at least 2
        # errors to compare.
        return err_df.ewm(
            span=np.min([err_df.size - 1, ema_win]),
            min_periods=np.min([err_df.size - 1, ema_win])
        ).mean()

    def check_ema_improvement(self, err, ema, threshold):
        # Convenience function to increase readability
        # If threshold == 0 this boils down to lesser operation
        return (
            err.iloc[-1, 0] / ema.iloc[-1, 0]
            < 1 - threshold
        ).all()

    def check_ema_improvement_min(self, err, ema, threshold):
        # Convenience function to increase readability
        # If threshold == 0 this boils down to lesser operation
        return (
            err.iloc[-1, 0] / np.min(ema.iloc[:-1, 0])
            < 1 - threshold
        ).all()

    def evaluate_on(self, model, dl_val):
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

    def check_policy(self, model, i, t_train_start, loss, dl_val, t_diff):
        make_prediction = False
        self.train_err = self.append_loss(self.train_err, loss)
        LOGGER.info('TRAIN BATCH #{0}:\t{1}'.format(i, loss))

        t_current = time.time() - t_train_start
        # The first 15 seconds just train and make a prediction
        if t_current <= 15:
            pass
        elif (
            model.config.earlystop is not None
            and time.time() - model.birthday > model.config.earlystop
        ):
            make_prediction = True
        elif t_current < 300 and dl_val is not None:
            if model.testing_round == 0:
                make_prediction = True
            # The first 5min do grid-like predictions
            ct_diff = (
                self.transform_time_abs(time.time() - model.birthday)
                - self.transform_time_abs(t_train_start - model.birthday)
            )
            if ct_diff < t_diff:
                pass
            else:
                make_prediction = True
        else:
            # If the last train error improves upon the minimum of it's exponential moving average
            # by at least 5% with window self.train_err_ewm = 5, escalate to validation.
            if self.train_err.size > 1:
                train_err_ewm = self.get_ema(self.train_err, self.train_ewm_window)
                if self.check_ema_improvement(self.train_err, train_err_ewm, 0.1):
                    self.val_err = self.append_loss(self.val_err, self.evaluate_on(dl_val))
                    if self.val_err.size > 1:
                        val_err_ewm = self.get_ema(self.val_err, self.val_ewm_window)
                        LOGGER.info('VALIDATION EMA: {0}'.format(val_err_ewm.iloc[-1, 0]))
                        LOGGER.info('VALIDATION ERR: {0}'.format(self.val_err.iloc[-1, 0]))
                        LOGGER.info('VALIDATION MIN: {0}'.format(np.min(self.val_err.iloc[:-1, 0])))
                        # If the last validation error improves upon the minimum of it's exponential moving average
                        # by 10% with window self.train_err_ewm = 5, escalate to test.
                        # if self.check_ema_improvement(self.val_err, val_err_ewm, 0.1):
                        if self.check_ema_improvement_min(self.val_err, val_err_ewm, 0.15):
                            make_prediction = True
                        LOGGER.info('BACK TO THE GYM')
        return make_prediction

    def __call__(self, model, dl_train, dl_val, remaining_time, **kwargs):
        LOGGER.info("TRAINING START: " + str(time.time()))
        LOGGER.info("REMAINING TIME: " + str(remaining_time))
        t_train = time.time() if model.training_round > 0 else model.birthday
        make_prediction = False
        make_final_prediction = False
        while not make_prediction:
            # Set train mode before we go into the train loop over an epoch
            dl_train.dataset.reset()
            for i, (data, labels) in enumerate(dl_train):
                t_left = self.get_time_wo_final_prediction(
                    remaining_time,
                    t_train,
                    model
                )
                # Abort training and make final prediction if not enough time is left
                if t_left is not None and t_left < 10:
                    LOGGER.info('Making final prediciton!')
                    make_final_prediction = True
                    make_prediction = True
                    break

                _, loss = train_step(
                    model.model,
                    model.optimizer,
                    model.loss_fn,
                    data,
                    labels
                )
                if self.check_policy(model, i, t_train, loss, dl_val, **kwargs):
                    make_prediction = True
                    break
        subprocess.run(['nvidia-smi'])
        LOGGER.info("TRAINING END: " + str(time.time()))
        return make_final_prediction


# Example for a training loop
def example_loop(model, dl_train, dl_val, remaining_time):
    pass


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
