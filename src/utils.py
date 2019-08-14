import os
import sys
import torch
import torch.nn as nn
import logging
import hjson


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASEDIR = os.path.dirname(os.path.abspath(__file__))


class MonkeyNet(nn.Sequential):
    '''
    The idea of the monkeynet is to expose all attributes of the networks
    it's given and is therefore a special kind of Sequential network
    '''
    def __init__(self, *nets):
        super().__init__(*nets)
        super().__setattr__('__finished_init__', True)

    def __getattr__(self, attr):
        try:
            super(nn.Sequential, self).__getattribute__(attr)
        except AttributeError:
            super(nn.Sequential, self).__getattribute__('__finished_init__')
        for m in self.children():
            try:
                return getattr(m, attr)
            except AttributeError:
                continue
        raise AttributeError('The monkey is sorry because it could not find ''{0}'''.format(attr))

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
        raise AttributeError('The monkey is sorry because it could not set ''{0}'''.format(attr))


class Config:
    def __init__(self, config_path):
        with open(config_path) as config_file:
            self.__dict__ = hjson.load(config_file)

    def write(self, save_path):
        with open(save_path, "w") as save_file:
            save_file.write(hjson.dumps(self.__dict__))


def get_logger():
    """Set logging format to something like:
            2019-04-25 12:52:51,924 INFO model.py: <message>
    """
    conf = Config(os.path.join(BASEDIR, "config.hjson"))
    logger = logging.getLogger(__file__)
    logging_level = getattr(logging, conf.log_level)
    logger.setLevel(logging_level)
    formatter = logging.Formatter(
        fmt='%(asctime)s %(levelname)s %(filename)s: %(message)s')
    fileout_handler = logging.FileHandler(os.path.join(BASEDIR, 'run.log'), mode='w')
    fileout_handler.setLevel(logging.DEBUG)
    fileout_handler.setFormatter(formatter)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging_level)
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
