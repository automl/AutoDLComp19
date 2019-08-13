import time
import logging
import subprocess
import pandas as pd
import numpy as np
import torch
from utils import LOGGER


# Example for a trainer class
# which is like the loop but has methods and variables
# and is prefered
class default_trainer():
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
        LOGGER.info("REMAINING TIME: " + str(remaining_time))

        self.update_hyperparams(autodl_model, autodl_model.train_dl['train'])
        self.update_batch_size(autodl_model)

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
                    autodl_model.lr_scheduler,
                    data,
                    labels
                )
                # ### Set dropout
                autodl_model.model.dropout = autodl_model.model.dropout + self.dropout_diff
                autodl_model.model.dropout = min(autodl_model.model.dropout, 0.9)
                autodl_model.model.alphadrop = torch.nn.AlphaDropout(p=autodl_model.model.dropout)

                self.batch_counter += 1
                self.ele_counter += data.shape[0]
                if self.check_policy2(autodl_model, i, t_train, loss, dl_val):
                    make_prediction = True
                    break
        if LOGGER.level == logging.debug:
            subprocess.run(['nvidia-smi'])
        LOGGER.info('DROPOUT: {0:.4g}'.format(autodl_model.model.dropout))
        LOGGER.info('LR: {0:.4e}'.format(autodl_model.optimizer.param_groups[0]['lr']))
        LOGGER.info("MEAN TRAINING FRAMES PER SEC: {0:.2f}".format(
            self.ele_counter / (time.time() - autodl_model.birthday))
        )
        LOGGER.info("TRAINING COUNTER: " + str(self.ele_counter))
        LOGGER.info("TRAINING END: " + str(time.time()))
        return make_final_prediction

    def check_policy(self, autodl_model, i, t_train_start, loss, dl_val):
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

    def check_policy2(self, autodl_model, i, t_train_start, loss, dl_val):
        make_prediction = False
        self.train_err = append_loss(self.train_err, loss)
        LOGGER.debug('TRAIN BATCH #{0}:\t{1}'.format(i, loss))

        # The first 20 batches just train and make a prediction
        if self.batch_counter <= 20:
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

    def update_batch_size(self, autodl_model):
        batch_size = int(self.bn_prod_limit / autodl_model.model.num_segments)
        trainloader_args = autodl_model.config.dataloader_args['train']
        trainloader_args['batch_size'] = batch_size
        autodl_model.train_dl['val'] = torch.utils.data.DataLoader(
            autodl_model.train_dl['val'].dataset,
            **trainloader_args
        )

        batch_size_des = autodl_model.config.dataloader_args['train']['batch_size']
        if autodl_model.model.training:
            bn_prod_des = batch_size_des * autodl_model.model.num_segments
            if bn_prod_des <= self.bn_prod_limit:
                batch_size = batch_size_des

        trainloader_args['batch_size'] = batch_size
        autodl_model.train_dl['train'] = torch.utils.data.DataLoader(
            autodl_model.train_dl['train'].dataset,
            **trainloader_args
        )
        LOGGER.info('SET BATCH SIZE: ' + str(batch_size))

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
            # TODO(Philipp): set num_segments in the transforms


# ########################################################
# Helpers
# ########################################################
def train_step(model, optimizer, criterion, lr_scheduler, data, labels):
    model.train()
    optimizer.zero_grad()
    output = model(data.cuda())
    loss = criterion(output, labels.cuda())
    loss.backward()
    optimizer.step()
    lr_scheduler.step()

    return output, loss


def eval_step(model, criterion, data, labels):
    model.eval()
    output = model(data.cuda())
    loss = criterion(output, labels.cuda())

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
        + np.mean(model.test_time)
        + np.std(model.test_time)
    ) if len(model.test_time) > 0 else None


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
