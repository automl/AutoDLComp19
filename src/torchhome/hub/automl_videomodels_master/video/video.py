# This file wraps around all known models which should be presented
# to torch.hub.list

import os

from .load_models import load_loss_criterion, load_model_and_optimizer
from .opts import parser

# HACK(Philipp J.): We can't use the normal load_state_dict_from_url because we got our own
TORCH_HOME = os.environ['TORCH_HOME']

__all__ = [
    'eco', 'ecofull', 'ecofull_py', 'ecofull_efficient_py', 'bninception', 'tsm',
    'averagenet', 'averagenet_feature', 'stagedaveragenet'
]

model_urls = {
    'ECO': None,
    'ECOfull': None,
    'ECOfull_py': None,
    'ECOfull_efficient_py': None,
    'ECObninception': None,
    'TSM': None,
    'Averagenet': None,
    'Averagenet_feature': None,
    'StagedAveragenet': None,
}


def eco(pretrained=False, url=None, **kwargs):
    pargs = parser.parse_known_args()[0]
    for k, v in kwargs.items():
        setattr(pargs, k, v)
    if pretrained:
        if url is not None:
            url = os.path.join(TORCH_HOME, 'checkpoints', url)
        setattr(pargs, 'finetune_model', url)
    setattr(pargs, 'arch', 'ECO')
    model, optimizer = load_model_and_optimizer(pargs)
    loss_fn = load_loss_criterion(pargs)
    return model, optimizer, loss_fn


def ecofull(pretrained=False, url=None, **kwargs):
    pargs = parser.parse_known_args()[0]
    for k, v in kwargs.items():
        setattr(pargs, k, v)
    if pretrained:
        if url is not None:
            url = os.path.join(TORCH_HOME, 'checkpoints', url)
        setattr(pargs, 'finetune_model', url)
    setattr(pargs, 'arch', 'ECOfull')
    model, optimizer = load_model_and_optimizer(pargs)
    loss_fn = load_loss_criterion(pargs)
    return model, optimizer, loss_fn


def ecofull_py(pretrained=False, url=None, **kwargs):
    pargs = parser.parse_known_args()[0]
    for k, v in kwargs.items():
        setattr(pargs, k, v)
    if pretrained:
        if url is not None:
            url = os.path.join(TORCH_HOME, 'checkpoints', url)
        setattr(pargs, 'finetune_model', url)
    setattr(pargs, 'arch', 'ECOfull_py')
    model, optimizer = load_model_and_optimizer(pargs)
    loss_fn = load_loss_criterion(pargs)
    return model, optimizer, loss_fn


def ecofull_efficient_py(pretrained=False, url=None, **kwargs):
    pargs = parser.parse_known_args()[0]
    for k, v in kwargs.items():
        setattr(pargs, k, v)
    if pretrained:
        if url is not None:
            url = os.path.join(TORCH_HOME, 'checkpoints', url)
        setattr(pargs, 'finetune_model', url)
    setattr(pargs, 'arch', 'ECOfull_efficient_py')
    model, optimizer = load_model_and_optimizer(pargs)
    loss_fn = load_loss_criterion(pargs)
    return model, optimizer, loss_fn


def bninception(pretrained=False, url=None, **kwargs):
    pargs = parser.parse_known_args()[0]
    for k, v in kwargs.items():
        setattr(pargs, k, v)
    if pretrained:
        if url is not None:
            url = os.path.join(TORCH_HOME, 'checkpoints', url)
        setattr(pargs, 'finetune_model', url)
    setattr(pargs, 'arch', 'bninception')
    model, optimizer = load_model_and_optimizer(pargs)
    loss_fn = load_loss_criterion(pargs)
    return model, optimizer, loss_fn


def tsm(pretrained=False, url=None, **kwargs):
    pargs = parser.parse_known_args()[0]
    for k, v in kwargs.items():
        setattr(pargs, k, v)
    if pretrained:
        if url is not None:
            url = os.path.join(TORCH_HOME, 'checkpoints', url)
        setattr(pargs, 'finetune_model', url)
    setattr(pargs, 'arch', 'TSM')
    model, optimizer = load_model_and_optimizer(pargs)
    loss_fn = load_loss_criterion(pargs)
    return model, optimizer, loss_fn


def averagenet(pretrained=False, url=None, **kwargs):
    pargs = parser.parse_known_args()[0]
    for k, v in kwargs.items():
        setattr(pargs, k, v)
    if pretrained:
        if url is not None:
            url = os.path.join(TORCH_HOME, 'checkpoints', url)
        setattr(pargs, 'finetune_model', url)
    setattr(pargs, 'arch', 'Averagenet')
    model, optimizer = load_model_and_optimizer(pargs)
    loss_fn = load_loss_criterion(pargs)
    return model, optimizer, loss_fn


def averagenet_feature(pretrained=False, url=None, **kwargs):
    pargs = parser.parse_known_args()[0]
    for k, v in kwargs.items():
        setattr(pargs, k, v)
    if pretrained:
        if url is not None:
            url = os.path.join(TORCH_HOME, 'checkpoints', url)
        setattr(pargs, 'finetune_model', url)
    setattr(pargs, 'arch', 'Averagenet_feature')
    model, optimizer = load_model_and_optimizer(pargs)
    loss_fn = load_loss_criterion(pargs)
    return model, optimizer, loss_fn


def stagedaveragenet(pretrained=False, url=None, **kwargs):
    pargs = parser.parse_known_args()[0]
    for k, v in kwargs.items():
        setattr(pargs, k, v)
    if pretrained:
        if url is not None:
            url = os.path.join(TORCH_HOME, 'checkpoints', url)
        setattr(pargs, 'finetune_model', url)
    setattr(pargs, 'arch', 'StagedAveragenet')
    model, optimizer = load_model_and_optimizer(pargs)
    loss_fn = load_loss_criterion(pargs)
    return model, optimizer, loss_fn
