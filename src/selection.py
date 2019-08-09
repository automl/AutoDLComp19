# Determin modality and select the right model, optimizer and loss
import os

import hjson
import torch
from wrapper_net import WrapperNet
from utils import LOGGER
# from torchhome.hub.autodlcomp_models_master.video.load_models import load_model_and_optimizer, load_loss_criterion

HUBNAME = 'autodlcomp/models'
TORCH_HOME = os.path.join(os.path.dirname(__file__), 'torchhome')
os.environ['TORCH_HOME'] = TORCH_HOME
with open(os.path.join(TORCH_HOME, 'manifest.hjson')) as manifest:
    HUBMANIFEST = hjson.load(manifest)

# The idea is get a dataset from our supportdataset(the sets we have pretrained models for)
# which is similar to the dataset given and base our model selection on it. This means
# we would iterate over all available models x datasets and choose/adjust setting saved in the
# manifest file


# Model selectors
def default_model_selector(tfsession, dataset):
    models_available = torch.hub.list(HUBNAME)
    conf = struct(HUBMANIFEST['averagenet']['kinetics'])

    LOGGER.info("AVAILABLE MODELS: {0}".format(models_available))
    LOGGER.info("TRAIN SET LENGTH: {0}".format(dataset.num_samples))
    LOGGER.info("INPUT SHAPE MEDIAN: {0}".format(dataset.median_shape))
    LOGGER.info("IS MULTILABEL: {0}".format(dataset.is_multilabel))

    modeloptimargs = {
        'dropout': 0.2,
        'lr': 0.005,
        'parser_args': {
            'num_classes': dataset.num_classes,
            'num_segments': 2,
            'modality': 'RGB',
            'classification_type': 'multilabel' if dataset.is_multilabel else 'multiclass',
        }
    }

    model, optimizer, loss_fn = torch.hub.load(
        HUBNAME, 'averagenet', pretrained=True, url=conf.checkpoint_file, **modeloptimargs
    )
    model = WrapperNet(model)

    # This is an example how you could get the optimizer from the manifest
    # ####
    # optimizer_class = getattr(
    #     torch.optim,
    #     conf.optimizer
    # )
    # optimizer = optimizer_class(
    #     model.parameters(),
    #     optimargs
    # )
    # optimizer = torch.optim.Adam(model.parameters(), **optimargs)

    # Return that this model can be trained using amp
    # Only set this to true if the optimizer and the model can be monkey patched by
    # nvidia apex's amp and amp should be used!
    amp_compatible = True

    return model, loss_fn, optimizer, amp_compatible


# ########################################################
# Helpers
# ########################################################
class struct:
    def __init__(self, ordereddict):
        self.__dict__ = ordereddict
