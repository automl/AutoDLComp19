import os

import torchvision
import torchvision.transforms.functional as F

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

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
