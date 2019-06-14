import re
import os
import torch
import torchvision

try:
    import utils
except ImportError:
    # This file is used as a script
    # TODO(Danny): Nicer solution to this
    import sys
    sys.path.append("src")
    import utils


def download_all(config):
    for model in config.models:
        model_family = re.sub("[^a-z]", "", re.search("^[^_]*",
                                                      model).group(0))
        model_family = getattr(torchvision.models, model_family)
        model_url = model_family.model_urls[model]
        if not os.path.basename(model_url) in os.listdir(config.model_dir):
            torch.utils.model_zoo.load_url(model_url, config.model_dir)


if __name__ == "__main__":
    config = utils.Config("src/image/pretrained_models.hjson")
    download_all(config)
