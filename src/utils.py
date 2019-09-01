import atexit
import logging
import multiprocessing
import os
import re
import sys
import time
import webbrowser
from functools import partial, wraps

import hjson
import numpy as np
import psutil
import torch
import torch.nn as nn
from sklearn import metrics as skmetrics

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASEDIR = os.path.dirname(os.path.abspath(__file__))
LOGDIR = os.path.abspath(
    os.path.join(
        BASEDIR, os.pardir, 'logs', time.strftime("%Y%m%d_%H%M%S", time.localtime())
    )
)
if not os.path.isdir(LOGDIR):
    os.mkdir(LOGDIR)

PREDICT_AND_VALIDATE = (True, True)
PREDICT = (True, False)
VALIDATE = (False, True)
TRAIN = (False, False)


def kill_if_alive(p):
    if p.is_alive():
        psutil.Process(p.pid).terminate()


try:
    from tensorboardX import SummaryWriter
    SW = SummaryWriter(LOGDIR)

    def spawn_tb(dir):
        webbrowser.open_new_tab('http://localhost:6006/')
        os.system(
            'export TF_CPP_MIN_LOG_LEVEL=3; tensorboard --logdir ' + dir + ' >> /dev/null'
        )

    tbp = multiprocessing.Process(target=spawn_tb, kwargs={'dir': LOGDIR}, daemon=True)
    tbp.start()
    atexit.register(kill_if_alive, tbp)
    print('Started tensorboard with PID \033[92m{}\033[0m'.format(tbp.pid))
except Exception as e:

    class BlackHole():
        '''
        Has everything, swallows all and returns nothing...
        and very handy to replace non-existing loggers but keeping
        the lines of code for it the same
        '''
        def __getattr__(self, attr):
            def f(*args, **kwargs):
                return

            return f

        def __setattr__(self, attr, val):
            return

    SW = BlackHole()


class Config:
    def __init__(self, config_path):
        with open(config_path) as config_file:
            self.__dict__ = hjson.load(config_file)

    def write(self, save_path):
        with open(save_path, 'w') as save_file:
            save_file.write(hjson.dumps(self.__dict__))


CONFIG = Config(os.path.join(BASEDIR, 'config.hjson'))


def get_logger():
    '''
    Set logging format to something like:
            2019-04-25 12:52:51,924 INFO model.py: <message>
    '''
    class LessThanFilter(logging.Filter):
        def __init__(self, exclusive_maximum, name=''):
            super(LessThanFilter, self).__init__(name)
            self.max_level = exclusive_maximum

        def filter(self, record):
            # non-zero return means we log this message
            return 1 if record.levelno < self.max_level else 0

    logger = logging.getLogger(__file__)
    logging_level = getattr(logging, CONFIG.log_level)
    logger.setLevel(logging_level)
    formatter = logging.Formatter(
        fmt='%(asctime)s %(levelname)s %(filename)s: %(message)s'
    )
    fileout_handler = logging.FileHandler(os.path.join(LOGDIR, 'run.log'), mode='w')
    fileout_handler.setLevel(logging.DEBUG)
    fileout_handler.setFormatter(formatter)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging_level)
    stdout_handler.addFilter(LessThanFilter(logging.WARNING))
    stdout_handler.setFormatter(formatter)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(formatter)
    logger.addHandler(fileout_handler)
    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)
    logger.propagate = False
    return logger


LOGGER = get_logger()

try:
    if CONFIG.profile_mem:
        from memory_profiler import profile as memprofile
        fp = open(os.path.join(LOGDIR, 'memory_profiler.log'), 'w+')
        memprofile = partial(memprofile, stream=fp)
    else:
        raise Exception
except Exception:
    # Fake version of the memory_profiler's profile decorator
    def memprofile(func=None, stream=None, precision=1, backend='psutil'):
        '''
        Stripped version of memory_profile's profile decorator
        '''
        if func is not None:

            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper
        else:

            def inner_wrapper(f):
                return memprofile(f, stream=stream, precision=precision, backend=backend)

            return inner_wrapper


def parse_cumem_error(err_str):
    mem_search = re.search(
        r'Tried to allocate ([0-9].*? [G|M])iB.*\; ([0-9]*.*? [G|M])iB free', err_str
    )
    tried, free = mem_search.groups()
    tried = float(tried[:-2]) * 1024 if tried[-1] == 'G' else float(tried[:-2])
    free = float(free[:-2]) * 1024 if free[-1] == 'G' else float(free[:-2])
    return tried, free


def BSGuard(f, loader, reset_on_fail):
    '''
    Decorator to safe-guard the execution of f against cuda's out of memory error
    and to recover from it if possible by reducing the batch-size of the given
    dataloader
    '''
    @wraps(f)
    def decorated(*args, **kwargs):
        while True:
            try:
                return f(*args, **kwargs)
            except RuntimeError as e:
                LOGGER.warn('CAUGHT VMEM ERROR! SCALING DOWN BATCH-SIZE!')
                if ('CUDA out of memory.' not in e.args[0] or loader.batch_size == 1):
                    raise e
                tried_mem, free_mem = parse_cumem_error(e.args[0])
                # Not working as intended. I blame fragmantation!
                # mem_downscale = min(1, free_mem / (tried_mem + 1e-8)) if free_mem < tried_mem else 0.9
                mem_downscale = 0.5
                loader.batch_size = max(1, int(loader.batch_size * mem_downscale))
                if reset_on_fail:
                    loader.reset()

                LOGGER.warn(
                    'TRIED TO ALLOCATE {0:.2f} MiB WITH {1:.2f} MiB FREE'.format(
                        tried_mem, free_mem
                    )
                )
                LOGGER.warn('BATCH-SIZE NOW IS {}'.format(loader.batch_size))

    return decorated


def transform_time_abs(t_abs, T=1200):
    '''
    conversion from absolute time 0s-1200s to relative time 0-1
    '''
    return np.log(1 + t_abs / 60.0) / np.log((T / 60.0) + 1)


def transform_time_rel(t_rel):
    '''
    convertsion from relative time 0-1 to absolute time 0s-1200s
    '''
    return 60 * (21**t_rel - 1)


class metrics():
    @staticmethod
    def accuracy(labels, out, multilabel):
        '''
        Returns the TPR
        '''
        if not multilabel:
            ooh = np.zeros_like(out)
            ooh[np.arange(len(out)), np.argmax(out, axis=1)] = 1
            out = ooh
            loh = np.zeros_like(out)
            loh[np.arange(len(labels)), labels] = 1
            labels = loh
        out = (out > 0).astype(int)
        nout = out / (np.sum(out, axis=1, keepdims=True) + 1e-15)
        true_pos = (labels * nout).sum() / float(labels.shape[0])
        return true_pos

    @staticmethod
    def auc(labels, out, multilabel):
        '''
        Returns the average auc-roc score
        '''
        roc_auc = -1
        if not multilabel:
            loh = np.zeros_like(out)
            loh[np.arange(len(out)), labels] = 1
            ooh = out

            # Remove classes without positive examples
            col_to_keep = (loh.sum(axis=0) > 0)
            loh = loh[:, col_to_keep]
            ooh = out[:, col_to_keep]

            fpr, tpr, _ = skmetrics.roc_curve(loh.ravel(), ooh.ravel())
            roc_auc = skmetrics.auc(fpr, tpr)
        else:
            loh = labels
            ooh = out

            # Remove classes without positive examples
            col_to_keep = (loh.sum(axis=0) > 0)
            loh = loh[:, col_to_keep]
            ooh = out[:, col_to_keep]

            roc_auc = skmetrics.roc_auc_score(loh, ooh, average='macro')
        return 2 * roc_auc - 1


class AugmentNet(nn.Module):
    '''
    The augment net's purpose is to perform augmentations on a whole batch at once
    on the GPU instead of per element on the cpu. If this has benefit still needs
    to be determined
    '''
    def __init__(self, transfs):
        super().__init__()
        self.augmentation = {
            'train': (nn.Sequential(*transfs['train'])),
            'eval': (nn.Sequential(*transfs['test']))
        }

    def forward(self, x):
        mode = 'train' if self.training else 'eval'
        x = self.augmentation[mode](x)
        return x


class MonkeyNet(nn.Sequential):
    '''
    The idea of the monkeynet is to expose all attributes of the networks
    it's given and is therefore a special kind of Sequential network
    '''
    def __init__(self, *nets):
        super().__init__(*nets)
        super().__setattr__('__finished_init__', True)

    def __getattr__(self, attr):
        ret = None
        try:
            ret = super(nn.Sequential, self).__getattribute__(attr)
        except AttributeError:
            super(nn.Sequential, self).__getattribute__('__finished_init__')
        try:
            ret = super(nn.Sequential, self).__getattr__(attr)
        except AttributeError:
            pass
        for m in self.children():
            try:
                ret = getattr(m, attr)
            except AttributeError:
                continue
        if ret is None:
            raise AttributeError(
                'The monkey is sorry because it could not find '
                '{0}'
                ''.format(attr)
            )
        else:
            return ret

    def __setattr__(self, attr, val):
        try:
            super(nn.Sequential, self).__getattribute__(attr)
            super(nn.Sequential, self).__setattr__(attr, val)
            return
        except AttributeError:
            try:
                super(nn.Sequential, self).__getattribute__('__finished_init__')
            except AttributeError:
                super(nn.Sequential, self).__setattr__(attr, val)
                return
        for m in self.children():
            try:
                getattr(m, attr)
                setattr(m, attr, val)
                return
            except AttributeError:
                continue
        raise AttributeError(
            'The monkey is sorry because it could not set '
            '{0}'
            ''.format(attr)
        )
