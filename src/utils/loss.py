from torch.nn.modules.loss import _Loss


class ClassRectificationLoss(_Loss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, predictions, labels):
        n_bs = len(labels)
        pn_bs = 0.5 * n_bs
        h = torch.sum(labels, dim=0)
