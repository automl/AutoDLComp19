import time

import numpy as np
import psutil
import torch
from utils import (  # noqa: F401
    DEVICE, KEEP_AVAILABLE, LOGDIR, LOGGER, PREDICT, PREDICT_AND_VALIDATE, SW, TRAIN,
    VALIDATE, memprofile, metrics, print_vram_usage, transform_time_abs
)


class PolicyTrainer():
    def __init__(self, use_validation_cache, preserve_ram_for_nele=0, policy_fn=None):
        self.batch_counter = 0
        self.training_round = 0
        self.last_batch_size = 0
        self.dloader = None

        # Default policy
        self.check_policy = grid_check_policy(0.02) if policy_fn is None else policy_fn

        # Validation cache
        self.use_validation_cache = use_validation_cache
        self.validation_min_dist = 3
        self.validation_min_sample_per_class = 10
        self.validation_idxs = []
        self.validation_cache = []
        self.validation_class_dis = None
        self.validation_cache_valid = False
        self.validate = False
        # How many elements to consider when allocating ram
        # During the first training round
        self.preserve_ram_for_nele = preserve_ram_for_nele

    def _get_batch_class_dist(self, labels):
        if self.dloader.dataset.is_multilabel:
            loh = labels
        else:
            loh = np.zeros((len(labels), self.dloader.dataset.num_classes))
            loh[np.arange(len(labels)), labels] = 1
        return loh.sum(axis=0)

    def _reset_val_cache(self):
        self.validation_idxs = []
        self.validation_cache = []
        self.validation_class_dis = None
        self.validation_cache_valid = False

    @memprofile(precision=2)
    def _check_and_cache_val_batch(self, data, labels):
        batch_idx = int(self.dloader.next_idx / self.dloader.batch_size)
        if batch_idx in self.validation_idxs:
            return True
        if (
            not self.use_validation_cache or self.validation_cache_valid or (
                len(self.validation_idxs) > 0 and
                batch_idx - self.validation_idxs[-1] <= self.validation_min_dist
            )
        ):
            return False

        if self.batch_counter > batch_idx:
            # Well, that's all folks no new data for you
            self.validation_cache_valid = True

            LOGGER.debug(
                'TRAIN DATASET HAS NO UNSEEN BATCHES\n' +
                'CURRENT CACHE CLASS HIST:\n{}'.format(self.validation_class_dis)
            )

            return False

        data_mem_size = (data.element_size() * data.nelement())
        labels_mem_size = (labels.element_size() * labels.nelement())
        # Inflate estimated ram-usage by 3 GB to not hog all memory available
        available_mem = psutil.virtual_memory().available - KEEP_AVAILABLE
        if self.preserve_ram_for_nele > 0 and self.training_round <= 0:
            # If no prediction has been made and the test cache is enabled
            # reserve sufficient space for it
            available_mem -= self.preserve_ram_for_nele * data_mem_size / len(data)
        # Add 1 to max_count for the current one
        max_count = available_mem / (data_mem_size + labels_mem_size) + 1
        if max_count < 2:
            # Ram is full so declare cache as fit for use even if it isn't
            # Fit for use. This means every class is sufficiently represented in the cache
            self.validation_cache_valid = True

            LOGGER.debug(
                (
                    '###################################\n'
                    'NOT ENOUGH RAM LEFT TO ADD ANOTHER SAMPLE TO THE VALIDATION CACHE\n'
                    'CURRENT CACHE CLASS HIST:\n{}'
                ).format(self.validation_class_dis)
            )

            return False

        if self.validation_class_dis is None:
            self.validation_class_dis = np.zeros(self.dloader.dataset.num_classes)
        batch_class_dist = self._get_batch_class_dist(labels)
        new_dist = (batch_class_dist + self.validation_class_dis)
        if (
            (new_dist[new_dist > 0]**-2).sum() <
            (self.validation_class_dis[self.validation_class_dis > 0]**-2).sum()
        ):
            return False
        self.validation_class_dis += batch_class_dist
        self.validation_cache.append((data, labels))
        self.validation_idxs.append(batch_idx)
        if self.validation_class_dis.min() >= self.validation_min_sample_per_class:
            self.validation_cache_valid = True

        LOGGER.debug('GRABBING BATCH FOR VALIDATION')

        return True

    @memprofile(precision=2)
    def _train(self, i, autodl_model, data, labels):
        # Train on a batch if we re good to go
        # Copy data to DEVICE for training
        data = data.to(DEVICE, dtype=torch.float32, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        out, loss = train_step(
            autodl_model.model, autodl_model.loss_fn, data, labels,
            autodl_model.optimizer, autodl_model.lr_scheduler
        )
        labels, out, loss = (
            labels.cpu().numpy(), out.detach().cpu().numpy(), loss.detach().cpu().numpy()
        )
        train_acc, train_auc = (
            metrics.accuracy(labels, out, self.dloader.dataset.is_multilabel),
            metrics.auc(labels, out, self.dloader.dataset.is_multilabel)
        )
        SW.add_scalar('Train_Loss', loss, self.batch_counter)
        SW.add_scalar('Train_Acc', train_acc, self.batch_counter)
        SW.add_scalar('Train_Auc', train_auc, self.batch_counter)
        SW.add_histogram(
            'Train_Classes',
            np.argwhere(labels > 0)[:, 1]
            if self.dloader.dataset.is_multilabel else labels, self.batch_counter
        )

        LOGGER.debug(
            'STEP #{0}\tNEXT TRAIN IDX #{1}:\t{2:.6f}\t{3:.2f}\t{4:.2f}'.format(
                i, self.dloader.next_idx, loss, train_acc, train_auc
            )
        )

        return labels, out, loss

    @memprofile(precision=2)
    def _validate(self, autodl_model):
        if not self.validate or not self.validation_cache_valid:
            return None, None, None
        # Under construction
        v_start = time.time()
        llabels, lout, lloss = [], [], []
        for tdata, tlabels in self.validation_cache:
            tdata = tdata.to(DEVICE, dtype=torch.float32, non_blocking=True)
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
            lloss.append(tloss)
        vlabels, vout, vloss = (
            np.vstack(llabels) if self.dloader.dataset.is_multilabel else
            np.hstack(llabels), np.vstack(lout), np.vstack(lloss).mean()
        )
        val_acc, val_auc = (
            metrics.accuracy(vlabels, vout, self.dloader.dataset.is_multilabel),
            metrics.auc(vlabels, vout, self.dloader.dataset.is_multilabel)
        )
        SW.add_scalar('Valid_Loss', vloss, self.batch_counter)
        SW.add_scalar('Valid_Acc', val_acc, self.batch_counter)
        SW.add_scalar('Valid_Auc', val_auc, self.batch_counter)
        SW.add_histogram(
            'Valid_Classes',
            np.argwhere(vlabels > 0)[:, 1]
            if self.dloader.dataset.is_multilabel else vlabels, self.batch_counter
        )
        self.validate = False

        LOGGER.debug(
            'VALIDATION TOOK {0:.4f} s:\t\t{1:.6f}\t{2:.2f}\t{3:.2f}'.format(
                time.time() - v_start, vloss, val_acc, val_auc
            )
        )

        return vlabels, vout, vloss

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
        if self.last_batch_size != self.dloader.batch_size:
            self._reset_val_cache()
            self.last_batch_size = self.dloader.batch_size

        t_train = time.time()
        batch_counter_start = self.batch_counter
        batch_loading_time = 0
        make_prediction, make_final_prediction = False, False
        while not make_prediction:
            load_start = time.time()
            for i, (data, labels) in enumerate(self.dloader):
                batch_loading_time += time.time() - load_start
                self.batch_counter += 1

                # Run prechecks whether we abort training or not (earlystop or final pred.)
                make_prediction, make_final_prediction = precheck(
                    autodl_model, t_train, remaining_time
                )
                if make_prediction:
                    break

                # Check if the current batch is in the validation cache and skip if yes
                # If not and we want to add it skip as well
                if self._check_and_cache_val_batch(data, labels):
                    continue

                labels, out, loss = self._train(i, autodl_model, data, labels)
                vlabels, vout, vloss = self._validate(autodl_model)

                # Check if we want to make a prediction/validate after next train or not
                make_prediction, self.validate = self.check_policy(
                    autodl_model, t_train, remaining_time - (time.time() - t_train),
                    labels, out, loss, vlabels, vout, vloss
                )

                if make_prediction:
                    break
                load_start = time.time()
        self.training_round += 1

        if self.batch_counter - batch_counter_start > 0:
            spbl = batch_loading_time / (self.batch_counter - batch_counter_start)
            LOGGER.debug(
                (
                    '###################################\n'
                    'SEC PER BATCH LOADING:\t{0:.4f}\n'
                    'SEC TOTAL DATA LOADING:\t{0:.4f}'
                ).format(spbl, batch_loading_time)
            )
        else:
            LOGGER.debug('NO BATCH PROCESSED')
        print_vram_usage()

        return make_final_prediction


class grid_check_policy():
    '''
    This is the default policy if no other is chosen
    it makes a prediction every t_diff percent along
    the log timescale with T=1200
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
            if len(autodl_model.test_time) == 0:
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
