import numpy as np
from sklearn import metrics as skmetrics


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


def auc(labels, out, multilabel):
    '''
    Returns the average auc-roc score
    '''
    roc_auc = -1
    if multilabel:
        loh = labels
        ooh = out

        # Remove classes without positive examples
        col_to_keep = (((ooh > 0).astype(int) + loh).sum(axis=0) > 0)
        loh = loh[:, col_to_keep]
        ooh = out[:, col_to_keep]

        roc_auc = skmetrics.roc_auc_score(loh, ooh, average='macro')
    else:
        loh = np.zeros_like(out)
        loh[np.arange(len(out)), labels] = 1
        ooh = out

        # Remove classes without positive examples
        col_to_keep = (((ooh > 0).astype(int) + loh).sum(axis=0) > 0)
        loh = loh[:, col_to_keep]
        ooh = out[:, col_to_keep]

        fpr, tpr, _ = skmetrics.roc_curve(loh.ravel(), ooh.ravel())
        roc_auc = skmetrics.auc(fpr, tpr)

    return 2 * roc_auc - 1
