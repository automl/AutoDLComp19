import os

import torchvision
import torchvision.transforms.functional as F

from PIL import Image

from dataset_kakaobrain import *


def get_model(model_name, model_dir, dropout, num_classes):
    """
    select proper model based on information from the dataset (image/video, etc.)
    """
    LOGGER.info("+++++++++++++ ARCH ++++++++++++++")
    LOGGER.info(model_name)

    if "squeezenet" in model_name:
        from torchvision.models import squeezenet1_1

        model = squeezenet1_1(pretrained=False, num_classes=num_classes).cuda()
        if "32" in model_name:
            save_file = "cifar100_squeezenet_input_32_bs_32_SGD.pth"
        elif "64" in model_name:
            save_file = "imagenet_squeezenet_epochs_87_input_64_bs_256_SGD_ACC_33_7.pth"
        elif "128" in model_name:
            save_file = "imagenet_squeezenet_epochs_128_input_128_bs_256_SGD_ACC_51.pth"
        elif "224" in model_name:
            save_file = "imagenet_squeezenet_epochs_0_input_224_bs_256_SGD_ACC_58_3.pth"

    elif "shufflenet05" in model_name:
        from torchvision.models import shufflenet_v2_x0_5

        model = shufflenet_v2_x0_5(pretrained=False, num_classes=num_classes).cuda()
        if "32" in model_name:
            save_file = "cifar100_shufflenet05_input_32_bs_128_SGD.pth"
        elif "64" in model_name:
            save_file = "imagenet_shufflenet05_epochs_48_input_64_bs_512_SGD_ACC_25_8.pth"
        elif "128" in model_name:
            save_file = "imagenet_shufflenet05_epochs_111_input_128_bs_512_SGD_ACC_47_3.pth"
        elif "224" in model_name:
            save_file = "imagenet_shufflenet05_epochs_0_input_224_bs_512_SGD_ACC_48_9.pth"

    elif "shufflenet10" in model_name:
        from torchvision.models import shufflenet_v2_x1_0

        model = shufflenet_v2_x1_0(pretrained=False, num_classes=num_classes).cuda()
        if "32" in model_name:
            save_file = "cifar100_shufflenet10_input_32_bs_64_SGD.pth"
        elif "64" in model_name:
            save_file = "imagenet_shufflenet10_epochs_5_input_64_bs_512_SGD_ACC_30_5.pth"
        elif "128" in model_name:
            save_file = "imagenet_shufflenet10_epochs_10_input_128_bs_512_SGD_ACC_54_8.pth"
        elif "224" in model_name:
            save_file = "imagenet_shufflenet10_epochs_0_input_224_bs_512_SGD_ACC_63_4.pth"

    elif "shufflenet20" in model_name:
        from torchvision.models import shufflenet_v2_x2_0

        model = shufflenet_v2_x2_0(pretrained=False, num_classes=num_classes).cuda()
        if "32" in model_name:
            save_file = "cifar100_shufflenet20_input_32_bs_32_SGD.pth"
        elif "64" in model_name:
            save_file = "imagenet_shufflenet20_epochs_114_input_64_bs_512_SGD_ACC_46_8.pth"
        elif "128" in model_name:
            save_file = "imagenet_shufflenet20_epochs_115_input_128_bs_512_SGD_ACC_61_5.pth"
        elif "224" in model_name:
            save_file = "imagenet_shufflenet20_epochs_110_input_224_bs_512_SGD_ACC_68.pth"

    elif "resnet18" in model_name:
        from torchvision.models import resnet18

        model = resnet18(pretrained=False, num_classes=num_classes).cuda()
        if "32" in model_name:
            save_file = "cifar100_resnet18_input_32_bs_128_SGD.pth"
        elif "64" in model_name:
            save_file = "imagenet_resnet18_epochs_88_input_64_bs_256_SGD_ACC_41_4.pth"
        elif "128" in model_name:
            save_file = "imagenet_resnet18_epochs_67_input_128_bs_256_SGD_ACC_63.pth"
        elif "224" in model_name:
            save_file = "imagenet_resnet18_epochs_0_input_224_bs_256_SGD_ACC_68_8.pth"

    elif "mobilenetv2_64" in model_name:
        from torchvision.models import mobilenet_v2

        model = mobilenet_v2(pretrained=False, width_mult=0.25).cuda()
        model.classifier[1] = torch.nn.Linear(1280, num_classes)
        save_file = "imagenet_mobilenetv2_epochs_83_input_64_bs_256_SGD_ACC_24_8.pth"

    elif "efficientnet" in model_name:
        if "_pytorch" in model_name:
            if "32" in model_name:
                save_file = "cifar100_efficientnet_pytorch_input_32_bs_128_SGD.pth"
            elif "64" in model_name:
                save_file = "imagenet_efficientnet_pytorch_input_64_bs_256_SGD.pth"
            elif "128" in model_name:
                save_file = "imagenet_efficientnet_pytorch_input_128_bs_256_SGD.pth"
            elif "224" in model_name:
                save_file = "imagenet_efficientnet_pytorch_input_224_bs_256_SGD.pth"
            from efficientnet_pytorch import EfficientNet

            model = EfficientNet.from_name(
                "efficientnet-b0", override_params={"num_classes": num_classes}
            ).cuda()
        else:
            if "b07" in model_name:
                scale = 0.7
                if "32" in model_name:
                    save_file = "cifar100_efficientnet_b07_input_32_bs_64_SGD.pth"
                elif "64" in model_name:
                    save_file = (
                        "imagenet_efficientnet_b07_epochs_104_input_64_bs_256_SGD_ACC_33_8.pth"
                    )
                elif "128" in model_name:
                    save_file = (
                        "imagenet_efficientnet_b07_epochs_129_input_128_bs_256_SGD_ACC_53_3.pth"
                    )
                elif "224" in model_name:
                    save_file = (
                        "imagenet_efficientnet_b07_epochs_128_input_224_bs_256_SGD_ACC_62_6.pth"
                    )
            elif "b05" in model_name:
                scale = 0.5
                if "32" in model_name:
                    save_file = "cifar100_efficientnet_b05_input_32_bs_128_SGD.pth"
                elif "64" in model_name:
                    save_file = (
                        "imagenet_efficientnet_b05_epochs_129_input_64_bs_256_SGD_ACC_24.pth"
                    )
                elif "128" in model_name:
                    save_file = (
                        "imagenet_efficientnet_b05_epochs_125_input_128_bs_256_SGD_ACC43_8_.pth"
                    )
                elif "224" in model_name:
                    save_file = (
                        "imagenet_efficientnet_b05_epochs_126_input_224_bs_256_SGD_ACC_54_2.pth"
                    )
            elif "b03" in model_name:
                scale = 0.3
                if "32" in model_name:
                    save_file = "cifar100_efficientnet_b03_input_32_bs_256_SGD.pth"
                elif "64" in model_name:
                    save_file = (
                        "imagenet_efficientnet_b03_epochs_128_input_64_bs_256_SGD_ACC_15.pth"
                    )
                elif "128" in model_name:
                    save_file = (
                        "imagenet_efficientnet_b03_epochs_127_input_128_bs_256_SGD_ACC_28_5.pth"
                    )
                elif "224" in model_name:
                    save_file = (
                        "imagenet_efficientnet_b03_epochs_129_input_224_bs_256_SGD_ACC_38_2.pth"
                    )
            elif "b0" in model_name:
                scale = 1
                if "32" in model_name:
                    save_file = "cifar100_efficientnet_b0_input_32_bs_128_SGD.pth"
                elif "64" in model_name:
                    save_file = (
                        "imagenet_efficientnet_b0_epochs_27_input_64_bs_256_SGD_ACC_41_5.pth"
                    )
                elif "128" in model_name:
                    save_file = (
                        "imagenet_efficientnet_b0_epochs_115_input_128_bs_256_SGD_ACC_52.pth"
                    )
                elif "224" in model_name:
                    save_file = (
                        "imagenet_efficientnet_b0_epochs_185_input_224_bs_256_SGD_ACC_67_5.pth"
                    )
            from common.models_efficientnet import EfficientNet

            model = EfficientNet(
                num_classes=num_classes,
                width_coef=scale,
                depth_coef=scale,
                scale=scale,
                dropout_ratio=dropout,
                pl=0.2,
                arch="fullEfficientnet",
            ).cuda()

    elif "densenet05" in model_name:
        from torchvision.models import DenseNet

        model = DenseNet(
            growth_rate=16,
            block_config=(3, 6, 12, 8),
            num_init_features=64,
            bn_size=2,
            drop_rate=dropout,
            num_classes=num_classes,
        ).cuda()
        if "32" in model_name:
            save_file = "cifar100_densenet05_input_32_bs_32_SGD.pth"
        elif "64" in model_name:
            save_file = "imagenet_densenet05_epochs_117_input_64_bs_256_SGD_ACC_24.pth"
        elif "128" in model_name:
            save_file = "imagenet_densenet05_epochs_124_input_128_bs_256_SGD_ACC_44_7.pth"
        elif "224" in model_name:
            save_file = "imagenet_densenet05_epochs_122_input_224_bs_256_SGD_ACC_50.pth"

    elif "densenet025" in model_name:
        from torchvision.models import DenseNet

        model = DenseNet(
            growth_rate=8,
            block_config=(2, 4, 8, 4),
            num_init_features=32,
            bn_size=2,
            drop_rate=dropout,
            num_classes=num_classes,
        ).cuda()
        if "32" in model_name:
            save_file = "cifar100_densenet025_input_32_bs_32_SGD.pth"
        elif "64" in model_name:
            save_file = "imagenet_densenet025_epochs_113_input_64_bs_256_SGD_ACC_17_3.pth"
        elif "128" in model_name:
            save_file = "imagenet_densenet025_epochs_133_input_128_bs_256_SGD_ACC_23_9.pth"
        elif "224" in model_name:
            save_file = "imagenet_densenet025_input_224_bs_256_SGD.pth"

    elif "densenet" in model_name:
        from torchvision.models import densenet121

        model = densenet121(pretrained=False, num_classes=num_classes, drop_rate=dropout).cuda()
        if "32" in model_name:
            save_file = "cifar100_densenet_input_32_bs_256_SGD.pth"
        elif "64" in model_name:
            save_file = "imagenet_densenet_epochs_139_input_64_bs_256_SGD_ACC_52_8.pth"
        elif "128" in model_name:
            save_file = "imagenet_densenet_epochs_90_input_128_bs_256_SGD_ACC_63_8.pth"
        elif "224" in model_name:
            save_file = "imagenet_densenet_epochs_0_input_224_bs_256_SGD_ACC_72_7.pth"

    else:
        raise TypeError("Unknown model type")

    model = load_model(model, os.path.join(model_dir, save_file)).cuda()

    return model


def get_input_size(model_name):
    if "32" in model_name:
        return 32
    elif "64" in model_name:
        return 64
    elif "128" in model_name:
        return 128
    elif "224" in model_name:
        return 224


def get_loss_criterion(classification_type):
    if classification_type == "multiclass":
        return torch.nn.CrossEntropyLoss().cuda()
    elif classification_type == "multilabel":
        return torch.nn.BCEWithLogitsLoss().cuda()
    else:
        raise ValueError("Unknown loss type")


def get_optimizer(model, optimizer_type, lr):
    if optimizer_type == "SGD":
        return torch.optim.SGD(model.parameters(), lr, momentum=0.9, nesterov=True)
    elif optimizer_type == "Adam":
        return torch.optim.Adam(model.parameters(), lr)
    else:
        raise ValueError("Unknown optimizer type")


def load_model(model, save_file):
    #################################################################################################
    pretrained_dict = torch.load(save_file)
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys and if different classes
    new_state_dict = {}
    for k1, v in pretrained_dict.items():
        for k2 in model_dict.keys():
            k = k1.replace("module.", "")
            if k2 in k and (v.size() == model_dict[k2].size()):
                new_state_dict[k2] = v
    # If explicitely delete last fully connected layed(like finetuning with new
    # dataset that has equal amount of classes
    # if remove_last_fc:
    #     temp_keys = []
    #     for k in new_state_dict.keys():
    #         if 'last' in k:
    #             temp_keys.append(k)
    #     for k in temp_keys:
    #         del new_state_dict[k]

    un_init_dict_keys = [k for k in model_dict.keys() if k not in new_state_dict]

    print("un_init_dict_keys: ", un_init_dict_keys)
    print("\n------------------------------------")

    for k in un_init_dict_keys:
        new_state_dict[k] = torch.DoubleTensor(model_dict[k].size()).zero_()
        if "weight" in k:
            if "bn" in k:
                print("{} init as: 1".format(k))
                torch.nn.init.constant_(new_state_dict[k], 1)
            else:

                print("{} init as: xavier".format(k))
                try:
                    torch.nn.init.xavier_uniform_(new_state_dict[k])
                except Exception:
                    torch.nn.init.constant_(new_state_dict[k], 1)
        elif "bias" in k:

            print("{} init as: 0".format(k))
            torch.nn.init.constant_(new_state_dict[k], 0)
        print("------------------------------------")
    model.load_state_dict(new_state_dict)
    print("loaded model")

    return model


def get_transform(is_training, input_size, scale=0.7, ratio=0.75):
    if is_training:
        return torchvision.transforms.Compose(
            [
                SelectSample(),
                AlignAxes(),
                FormatChannels(channels_des=3),
                ToPilFormat(),
                torchvision.transforms.RandomResizedCrop(
                    size=input_size, scale=(scale, 1.0), ratio=(ratio, 1 / ratio)
                ),
                torchvision.transforms.RandomHorizontalFlip(),
                ToTorchFormat(),
            ]
        )
    else:
        return torchvision.transforms.Compose(
            [
                SelectSample(),
                AlignAxes(),
                FormatChannels(channels_des=3),
                ToPilFormat(),
                torchvision.transforms.Resize(size=(input_size, input_size)),
                ToTorchFormat(),
            ]
        )


def get_dataloader(
    model, dataset, session, is_training, first_round, batch_size, input_size, num_samples
):
    transform = get_transform(is_training=is_training, input_size=input_size)

    ds = TFDataset(session=session, dataset=dataset, num_samples=num_samples, transform=transform)

    # reduce batch size until it fits into memory
    if first_round:
        batch_size_ok = False

        while not batch_size_ok and batch_size > 1:
            ds.reset()
            try:
                dl = torch.utils.data.DataLoader(
                    ds, batch_size=batch_size, shuffle=False, drop_last=False
                )

                data, labels = next(iter(dl))
                model(data.cuda())

                batch_size_ok = True

            except RuntimeError as e:
                LOGGER.info(str(e))
                batch_size = int(batch_size / 2)
                if is_training:
                    LOGGER.info("REDUCING BATCH SIZE FOR TRAINING TO: " + str(batch_size))
                else:
                    LOGGER.info("REDUCING BATCH SIZE FOR TESTING TO: " + str(batch_size))

    LOGGER.info("USING BATCH SIZE: " + str(batch_size))
    ds.reset()
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)

    return dl, batch_size


def format_labels(labels, classification_type):
    if classification_type == "multiclass":
        return np.argmax(labels, axis=1)
    else:
        return labels


def transform_to_time_rel(t_abs):
    """
    conversion from absolute time 0s-1200s to relative time 0-1
    """
    return np.log(1 + t_abs / 60.0) / np.log(21)


def transform_to_time_abs(t_rel):
    """
    convertsion from relative time 0-1 to absolute time 0s-1200s
    """
    return 60 * (21 ** t_rel - 1)


class ToTorchFormat(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """

    def __init__(self):
        self.trans = torchvision.transforms.ToTensor()

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic).permute(2, 0, 1).float().div(255)
        else:
            # handle PIL Image
            img = self.trans(pic).float()
        return img


class SaveImage(object):
    def __init__(self, save_dir, suffix):
        self.save_dir = save_dir
        self.suffix = suffix
        self.it = 0

    def __call__(self, pic):
        if isinstance(pic, torch.Tensor):
            pic_temp = F.to_pil_image(pic, mode="RGB")
        elif isinstance(pic, np.ndarray):
            pic_temp = Image.fromarray(np.uint8(pic), mode="RGB")
        else:
            pic_temp = pic

        self.it += 1
        pic_temp.save(os.path.join(self.save_dir, str(self.it) + str(self.suffix) + ".jpg"))
        return pic


class Stats(object):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        print("shape   " + str(x.shape))
        print("min val " + str(np.array(x).min()))
        print("max val " + str(np.array(x).max()))
        return x


class SelectSample(object):
    """
    given a video 4D array, randomly select sample images within segments
    """

    def __init__(self):
        super().__init__()

    def __call__(self, x):
        return x[np.random.randint(0, x.shape[0])]


class AlignAxes(object):
    """
    Swap axes if necessary
    """

    def __init__(self):
        super().__init__()
        pass

    def __call__(self, x):
        if x.shape[0] < min(x.shape[1], x.shape[2]):
            x = np.transpose(x, (1, 2, 0))
        return x


class FormatChannels(object):
    """
    Adapt number of channels. If there are more than desired, use only the first n channels.
    If there are less, copy existing channels
    """

    def __init__(self, channels_des):
        super().__init__()
        self.channels_des = channels_des

    def __call__(self, x):
        channels = x.shape[2]
        if channels < self.channels_des:
            x = np.tile(x, (1, 1, int(np.ceil(self.channels_des / channels))))
        x = x[:, :, 0 : self.channels_des]
        return x


class ToPilFormat(object):
    """
    convert from numpy/torch array (H x W x C) to PIL images (H x W x C)
    """

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            return Image.fromarray(np.uint8(pic * 255), mode="RGB")
        elif isinstance(pic, torch.Tensor):
            return F.to_pil_image(pic)
        else:
            raise TypeError("unknown input type")


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
