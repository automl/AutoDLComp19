import torch
import torch.nn as nn
import image.models as models
import functools
import utils


def _rgetattr(obj, attr, *args):
    """See https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects"""

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def _rsetattr(obj, attr, val):
    """See https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects"""
    pre, _, post = attr.rpartition(".")
    return setattr(_rgetattr(obj, pre) if pre else obj, post, val)


def _get_trainable_submodules(module):
    names = []
    params = []
    for name, param in module.named_modules():
        is_trainable_module = list(param.parameters()) and len(list(param.modules())) == 1
        if is_trainable_module:
            names.append(name)
            params.append(param)
    return names, params


def _adapt_last_layer(model, output_dim):
    # TODO(Danny): make this less complicated
    trainable_submodules_names, _ = _get_trainable_submodules(model)
    last_layer_name = trainable_submodules_names[-1]
    last_layer = _rgetattr(model, last_layer_name)
    num_ftrs = last_layer.in_features

    _rsetattr(model, last_layer_name, nn.Linear(num_ftrs, output_dim))
    model = nn.Sequential(model, nn.LogSoftmax(dim=1))
    return model


class OnlineMeta:
    def __init__(self, config, metadata_):
        self.config = config
        self.model = None

        # TODO(Danny): Document what metadata_ contains
        self.output_dim = metadata_.get_output_size()

    def select_model(self):
        if not self.model:
            model = models.models[self.config.model]
            model, self._initialize_model(model)
            # TODO(Danny): Use torchsummary
            utils.print_log(
                "Selected model before adapting last layer:\n {}".format(model)
            )
            model = _adapt_last_layer(model, self.output_dim)
            utils.print_log(
                "Selected model after adapting last layer:\n {}".format(model)
            )

            if torch.cuda.is_available():
                model.cuda()

            self.model = model

        return self.model, self.model_input_size

    def select_unfrozen_parameter(self, model):
        return None

    def _initialize_model(self, model):
        state_dict, self.model_input_size = models.get_parameters(
            self.config.pretrained_parameters, self.config
        )
        trainable_submodules_names, _ = _get_trainable_submodules(model)

        print(state_dict.keys())

        model.load_state_dict(state_dict)
        return model  # , self.model_input_size
