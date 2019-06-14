import functools

import image.models as models
import torch
import torch.nn as nn
import utils


def _recursive_getattr(obj, attr, *args):
    """See https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects"""

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def _recursive_setattr(obj, attr, val):
    """See https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects"""
    pre, _, post = attr.rpartition(".")
    return setattr(_recursive_getattr(obj, pre) if pre else obj, post, val)


def _get_ordered_trainable_submodules(module):
    names = []
    submodules = []
    for name, submodule in module.named_modules():
        is_trainable_module = (
            list(submodule.parameters()) and len(list(submodule.modules())) == 1
        )
        if is_trainable_module:
            names.append(name)
            submodules.append(submodule)
    return names, submodules


def _adapt_last_layer(model, output_dim):
    # TODO(Danny): make this less complicated
    trainable_submodules_names, _ = _get_ordered_trainable_submodules(model)
    last_layer_name = trainable_submodules_names[-1]
    last_layer = _recursive_getattr(model, last_layer_name)
    num_ftrs = last_layer.in_features

    _recursive_setattr(model, last_layer_name, nn.Linear(num_ftrs, output_dim))
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
            utils.print_log("Select the model architecture {}".format(self.config.model))
            model = models.models[self.config.model]
            model = self._initialize_model(model)
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
        if self.config.finetune_all:
            utils.print_log("Selected all modules to be unfrozen")
            return model.parameters()

        # Compute frozen / unfrozen modules using first_j and last_k configparameters
        submodules_names, submodules = _get_ordered_trainable_submodules(model)

        first_j = submodules[0:self.config.finetune_first_j]
        first_j_names = submodules_names[0:self.config.finetune_first_j]
        last_k = submodules[-self.config.finetune_last_k:]
        last_k_names = submodules_names[-self.config.finetune_last_k:]

        unfrozen_modules = first_j + last_k
        unfrozen_modules_names = first_j_names + last_k_names
        frozen_modules = submodules[self.config.
                                    finetune_first_j:-self.config.finetune_last_k]
        frozen_modules_names = submodules_names[self.config.finetune_first_j:-self.config.
                                                finetune_last_k]

        utils.print_log(
            "Selected the unfrozen modules:\n{}".format(unfrozen_modules_names)
        )
        utils.print_log("Selected the frozen modules:\n{}".format(frozen_modules_names))

        # Extract parameters from modules
        def parameters_from_module_list(module_list):
            return [
                parameter for module in module_list for parameter in module.parameters()
            ]

        unfrozen_parameters = parameters_from_module_list(unfrozen_modules)
        frozen_parameters = parameters_from_module_list(frozen_modules)

        # For efficiency reasons, disable gradient computation for frozen parameters
        # "Only if all inputs don’t require gradient, the output also won’t require it"
        # --- https://pytorch.org/docs/stable/notes/autograd.html
        for parameter in frozen_parameters:
            parameter.requires_grad = False

        return unfrozen_parameters

    def _initialize_model(self, model):
        # TODO(Danny): Allow random initialization
        # TODO(Danny): Initialize some unfrozen parameters to random?
        state_dict, self.model_input_size = models.get_parameters(
            self.config.pretrained_parameters, self.config
        )
        trainable_submodules_names, _ = _get_ordered_trainable_submodules(model)

        model.load_state_dict(state_dict)
        return model  # , self.model_input_size
