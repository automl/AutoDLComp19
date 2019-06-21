import numpy as np
from torch import randperm

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    lam = np.random.beta(alpha, alpha) if alpha > 0. else 1.
    batch_size = x.size()[0]
    index = randperm(batch_size).cuda() if use_cuda else randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

