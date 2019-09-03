import logging
import subprocess
import time

import numpy as np
import torch
from utils import DEVICE, LOGGER


class PolicyTrainer():
    def __init__(self, validation_buffer, policy_fn=None, **trainer_args):
        self.batch_counter = 0
        self.ele_counter = 0  # keep track of how many batches we trained on
        self.dloader = None
        self.trainer_args = trainer_args
        self.validation_idxs = []

    def __call__(self, autodl_model, remaining_time, birthday):
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

        t_start = time.time()
        for i, (data, labels) in enumerate(self.dloader):
            data = data.to(DEVICE)
            labels = labels.to(DEVICE)

            out, loss = train_step(
                autodl_model.model, autodl_model.optimizer, autodl_model.loss_fn,
                autodl_model.lr_scheduler, data, labels
            )

            LOGGER.debug('TRAINING ' + str(i))

            self.batch_counter += 1
            self.ele_counter += np.prod(data.shape[0:1])

            t_diff = (transform_time_abs(time.time() - birthday)
                      - transform_time_abs(t_start - birthday))

            if t_diff > self.trainer_args['t_diff']:
                self.make_prediction = True
                break

def transform_time_abs(t_abs):
    '''
    conversion from absolute time 0s-1200s to relative time 0-1
    '''
    return np.log(1 + t_abs / 60.0) / np.log(21)

def transform_time_rel(t_rel):
    '''
    conversion from relative time 0-1 to absolute time 0s-1200s
    '''
    return 60 * (21**t_rel - 1)


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

