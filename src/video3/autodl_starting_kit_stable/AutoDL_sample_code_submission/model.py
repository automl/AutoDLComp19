import _pickle as pickle
import logging
import os
import subprocess
import sys
import time
from functools import partial

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torchvision
from autoclint.model import Model as kakaobrain_model
from dataloader_kakaobrain import FixedSizeDataLoader
from dataset_kakaobrain import TFDataset
from ops.load_models import load_loss_criterion, load_model_and_optimizer
from opts import parser
from transforms import RandomCropPad, SelectSamples
from wrapper_net import WrapperNet

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.logging.set_verbosity(tf.logging.ERROR)


class ParserMock():
    # mock class for handing over the correct arguments
    def __init__(self):
        self._parser_args = parser.parse_known_args()[0]
        self.load_manual_parameters()
        self.load_bohb_parameters()
        self.load_apex()

    def load_manual_parameters(self):
        # manually set parameters
        rootpath = os.path.dirname(__file__)
        print('ROOT PATH: ' + str(rootpath))
        setattr(
            self._parser_args, 'finetune_model',
            os.path.join(rootpath, 'pretrained_models/')
        )
        setattr(self._parser_args, 'arch', 'bninception')  # Averagenet or bninception
        setattr(self._parser_args, 'batch_size_train', 16)
        setattr(self._parser_args, 'batch_size_test', 96)
        setattr(self._parser_args, 'num_segments', 4)
        setattr(self._parser_args, 'optimizer', 'SGD')
        setattr(self._parser_args, 'modality', 'RGB')
        setattr(self._parser_args, 'print', True)
        setattr(self._parser_args, 't_diff', 1.0 / 35)
        setattr(self._parser_args, 'splits', [85, 15])
        setattr(self._parser_args, 'lr', 0.01)
        setattr(self._parser_args, 'weight-decay', 0.001)

    def load_bohb_parameters(self):
        # parameters from bohb_auc
        path = os.path.join(os.getcwd(), 'smac_config.txt')
        if os.path.isfile(path):
            with open(path, 'rb') as file:
                logger.info('FOUND BOHB CONFIG, OVERRIDING PARAMETERS')
                bohb_cfg = pickle.load(file)
                logger.info('BOHB_CFG: ' + str(bohb_cfg))
                for key, value in bohb_cfg.items():
                    logger.info(
                        'OVERRIDING PARAMETER ' + str(key) + ' WITH ' + str(value)
                    )
                    setattr(self._parser_args, key, value)
            os.remove(path)

    def load_apex(self):
        # apex
        if torch.cuda.device_count() == 1:
            try:
                from apex import amp
                setattr(self._parser_args, 'apex_available', True)
            except Exception:
                pass
            logger.info('Apex = ' + str(self._parser_args.apex_available))

    def set_attr(self, attr, val):
        setattr(self._parser_args, attr, val)

    def parse_args(self):
        return self._parser_args


class Model(object):
    """Trivial example of valid model. Returns all-zero predictions."""

    def __init__(self, metadata):
        """
        Args:
          metadata: an AutoDLMetadata object. Its definition can be found in
              AutoDL_ingestion_program/dataset.py
        """
        logger.info("INIT START: " + str(time.time()))
        super().__init__()

        # This flag allows you to enable the inbuilt cudnn auto-tuner to find the best
        # algorithm to use for your hardware. Benchmark mode is good whenever your input sizes
        # for your network do not vary
        # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
        torch.backends.cudnn.benchmark = True

        self.time_start = time.time()
        self.train_time = []
        self.test_time = []

        self.done_training = False
        self.make_prediction = False
        self.make_final_prediction = False
        self.final_prediction_made = False

        self.metadata = metadata
        self.num_classes = self.metadata.get_output_size()
        self.num_examples_train = self.metadata.size()

        row_count, col_count = self.metadata.get_matrix_size(0)
        channel = self.metadata.get_num_channels(0)
        sequence_size = self.metadata.get_sequence_size()
        self.sequence_size = sequence_size
        print('INPUT SHAPE : ', row_count, col_count, channel, sequence_size)

        parser = ParserMock()
        parser.set_attr('num_classes', self.num_classes)
        parser.load_bohb_parameters()

        self.parser_args = parser.parse_args()

        if sequence_size == 1:
            #self.parser_args.finetune_model = self.parser_args.finetune_model + 'BnT_Image_Input_128.tar'
            #self.parser_args.num_segments = 1
            self.model_obj = kakaobrain_model(metadata, self.parser_args)
            self.parser_args.finetune_model = 'KAKAOBRAIN_MODEL'
        else:
            self.parser_args.finetune_model = self.parser_args.finetune_model + 'bnt_kinetics_input_128.pth.tar'
        print('USED MODEL: ' + str(self.parser_args.finetune_model))

        self.train_err = pd.DataFrame()  # collect train error
        self.train_ewm_window = 1  # window size of the exponential moving average
        self.val_err = pd.DataFrame()  # collect train error
        self.val_ewm_window = 1  # window size of the exponential moving average

        self.training_round = 0  # flag indicating if we are in the first round of training
        self.testing_round = 0  # flag indicating if we are in the first round of testing
        self.num_samples_training = None  # number of training samples
        self.num_samples_testing = None  # number of test samples

        self.session = tf.Session()
        logger.info("INIT END: " + str(time.time()))

    def train(self, dataset, remaining_time_budget=None):
        logger.info("TRAINING START: " + str(time.time()))
        logger.info("REMAINING TIME: " + str(remaining_time_budget))

        #########################################
        if self.sequence_size == 1:
            return self.model_obj.train(dataset, remaining_time_budget)

        self.training_round += 1

        t1 = time.time()

        # initial config during first round
        if int(self.training_round) == 1:
            self.late_init(dataset)

        t2 = time.time()

        transform = torchvision.transforms.Compose(
            [
                SelectSamples(self.parser_args.num_segments),
                RandomCropPad(self.model_main.input_size)
            ]
        )

        t3 = time.time()

        # [train_percent, validation_percent, ...]
        dl_train = self.split_dataset(dataset, transform)

        t4 = time.time()

        t_train = time.time()
        self.finish_loop = False
        while not self.finish_loop:
            # Set train mode before we go into the train loop over an epoch
            for i, (data, labels) in enumerate(dl_train):
                self.model.train()
                self.optimizer.zero_grad()

                output = self.model(data.cuda())
                labels = format_labels(labels, self.parser_args).cuda()

                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()

                self.train_err = self.append_loss(self.train_err, loss)
                logger.info('TRAIN BATCH #{0}:\t{1}'.format(i, loss))

                t_diff = (
                    transform_time_abs(time.time() - self.time_start) -
                    transform_time_abs(t_train - self.time_start)
                )

                if t_diff > self.parser_args.t_diff:
                    self.finish_loop = True
                    # NOTE: increase time per loop after each iteration
                    self.parser_args.t_diff *= 1.05
                    break

            #subprocess.run(['nvidia-smi'])
            self.training_round += 1

        t5 = time.time()

        logger.info(
            '\nTIMINGS TRAINING: ' + '\n t2-t1 ' + str(t2 - t1) + '\n t3-t2 ' +
            str(t3 - t2) + '\n t4-t3 ' + str(t4 - t3) + '\n t5-t4 ' + str(t5 - t4)
        )

        logger.info("TRAINING END: " + str(time.time()))
        self.train_time.append(t5 - t1)

    def late_init(self, dataset):
        logger.info('TRAINING: FIRST ROUND')
        # show directory structure
        # for root, subdirs, files in os.walk(os.getcwd()):
        #     logger.info(root)
        # get multiclass/multilabel information
        ds_temp = TFDataset(
            session=self.session, dataset=dataset, num_samples=self.num_examples_train
        )
        scan_start = time.time()
        info = ds_temp.scan2(10)
        logger.info('TRAIN SCAN TIME: {0}'.format(time.time() - scan_start))
        if info['is_multilabel']:
            setattr(self.parser_args, 'classification_type', 'multilabel')
        else:
            setattr(self.parser_args, 'classification_type', 'multiclass')

        self.model_main, self.optimizer = load_model_and_optimizer(
            self.parser_args, 0.3, 0.001
        )
        self.model = WrapperNet(self.model_main)
        self.model.cuda()

        # load proper criterion for multiclass/multilabel
        self.criterion = load_loss_criterion(self.parser_args)
        if self.parser_args.apex_available:
            from apex import amp

            def scaled_loss_helper(loss, optimizer):
                with amp.scale_loss(loss, optimizer) as scale_loss:
                    scale_loss.backward()

            def amp_loss(predictions, labels, loss_fn, optimizer):
                loss = loss_fn(predictions, labels)
                if hasattr(optimizer, '_amp_stash'):
                    loss.backward = partial(
                        scaled_loss_helper, loss=loss, optimizer=optimizer
                    )
                return loss

            self.criterion = partial(
                amp_loss, loss_fn=self.criterion, optimizer=self.optimizer
            )

    def split_dataset(self, dataset, transform):
        # [train_percent, validation_percent, ...]
        split_percentages = self.parser_args.splits / np.sum(self.parser_args.splits)
        split_num = np.round((self.num_examples_train * split_percentages))
        assert (sum(split_num) == self.num_examples_train)

        dataset.shuffle(self.num_examples_train)

        dataset_remaining = dataset
        dataset_train = dataset_remaining.take(split_num[0])
        dataset_remaining = dataset.skip(split_num[0])
        # dataset_val = dataset_remaining.take(split_num[1])
        # dataset_remaining = dataset_remaining.skip(split_num[1])

        ds_train = TFDataset(
            session=self.session,
            dataset=dataset_train,
            num_samples=int(split_num[0]),
            transform=transform
        )

        dl_train = FixedSizeDataLoader(
            ds_train,
            steps=int(split_num[0]),
            batch_size=self.parser_args.batch_size_train,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=False
        )

        return dl_train

    def append_loss(self, err_list, loss):
        # Convenience function to increase readability
        return err_list.append([loss.detach().cpu().tolist()], ignore_index=True)

    def evaluate_on(self, dl_val):
        val_error = np.Inf

        tempargs = self.parser_args
        tempargs.evaluate = True

        self.model.eval()
        with torch.no_grad():
            for i, (vdata, vlabels) in enumerate(dl_val):
                vlabels = format_labels(vlabels, tempargs).cuda()
                voutput = self.model(vdata.cuda())

                if np.isinf(val_error):
                    val_err = self.criterion(voutput, vlabels)
                else:
                    val_err += self.criterion(voutput, vlabels)
        return val_err

    def test(self, dataset, remaining_time_budget=None):
        logger.info("TESTING START: " + str(time.time()))
        logger.info("REMAINING TIME: " + str(remaining_time_budget))

        if self.sequence_size == 1:
            return self.model_obj.test(dataset, remaining_time_budget)

        self.testing_round += 1

        t1 = time.time()

        if int(self.testing_round) == 1:
            scan_start = time.time()
            ds_temp = TFDataset(
                session=self.session, dataset=dataset, num_samples=10000000
            )
            info = ds_temp.scan2()
            self.num_samples_testing = info['num_samples']
            logger.info('SCAN TIME: {0}'.format(time.time() - scan_start))
            logger.info('TESTING: FIRST ROUND')

        t2 = time.time()

        transform = torchvision.transforms.Compose(
            [
                SelectSamples(self.parser_args.num_segments),
                RandomCropPad(self.model_main.input_size)
            ]
        )
        predictions = None

        t3 = time.time()
        ds = TFDataset(
            session=self.session,
            dataset=dataset,
            num_samples=self.num_samples_testing,
            transform=transform
        )

        dl = torch.utils.data.DataLoader(
            ds, batch_size=self.parser_args.batch_size_test, drop_last=False
        )

        t4 = time.time()
        self.model.eval()
        with torch.no_grad():
            for i, (data, _) in enumerate(dl):
                logger.info('test: ' + str(i))
                data = data.cuda()
                output = self.model(data)
                if predictions is None:
                    predictions = output
                else:
                    predictions = torch.cat((predictions, output), 0)

        # remove if needed: Only train for 5 mins in order to save time on the submissions
        #if remaining_time_budget < 60:
        #    self.done_training = True
        #    return None

        t5 = time.time()

        logger.info(
            '\nTIMINGS TESTING: ' + '\n t2-t1 ' + str(t2 - t1) + '\n t3-t2 ' +
            str(t3 - t2) + '\n t4-t3 ' + str(t4 - t3) + '\n t5-t4 ' + str(t5 - t4)
        )

        logger.info("TESTING END: " + str(time.time()))
        self.test_time.append(t5 - t1)
        return predictions.cpu().numpy()

    ##############################################################################
    #### Above 3 methods (__init__, train, test) should always be implemented ####
    ##############################################################################


def format_labels(labels, parser_args):
    if parser_args.classification_type == 'multiclass':
        return np.argmax(labels, axis=1)
    else:
        return labels


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


def get_logger(verbosity_level):
    """Set logging format to something like:
         2019-04-25 12:52:51,924 INFO model.py: <message>
    """
    logger = logging.getLogger(__file__)
    logging_level = getattr(logging, verbosity_level)
    logger.setLevel(logging_level)
    formatter = logging.Formatter(
        fmt='%(asctime)s %(levelname)s %(filename)s: %(message)s'
    )
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging_level)
    stdout_handler.setFormatter(formatter)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)
    logger.propagate = False
    return logger


logger = get_logger('INFO')
