import torch
import torchvision
import hjson
import utils

def download_all(config):
    resnet18_url = torchvision.models.resnet.model_urls["resnet18"]
    torch.utils.model_zoo.load_url(resnet18_url, config.model_dir)


models = {"resnet18": torchvision.models.resnet18()}


def get_parameters(model_type, config):
    if model_type == "resnet18_imagenet_224":
        input_size = (224, 224)
        resnet18_url = torchvision.models.resnet.model_urls["resnet18"]
        state_dict = torch.utils.model_zoo.load_url(resnet18_url, config.model_dir)
        return state_dict, input_size
    else:
        raise  # TODO(Danny): Error message


if __name__ == "__main__":
    config = utils.Config("src/config.hjson")
    download_all(config)
