import torch
import torchvision


def download_all():
    resnet18_url = torchvision.models.resnet.model_urls["resnet18"]
    torch.utils.model_zoo.load_url(resnet18_url, "models")


models = {"resnet18": torchvision.models.resnet18()}


def get_parameters(model_type):
    if model_type == "resnet18_imagenet_224":
        input_size = (224, 224)
        resnet18_url = torchvision.models.resnet.model_urls["resnet18"]
        state_dict = torch.utils.model_zoo.load_url(resnet18_url, "models")
        return state_dict, input_size
    else:
        raise


if __name__ == "__main__":
    download_all()
