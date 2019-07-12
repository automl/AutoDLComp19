from dataset_reza import TSNDataSet
from transforms import Stack, ToTorchFormatTensor, GroupScale
from transforms import GroupCenterCrop, IdentityTransform, GroupNormalize
# from transforms import GroupMultiScaleCrop
# from transforms import GroupRandomHorizontalFlip
import torch, torchvision

def get_model_for_loader(parser_args):
    model = ()
    if parser_args.arch == "ECO" or parser_args.arch == "ECOfull":
        from models_eco import TSN
        model = TSN(parser_args.num_classes,
                    parser_args.num_segments,
                    parser_args.modality,
                    base_model=parser_args.arch,
                    consensus_type=parser_args.consensus_type,
                    dropout=parser_args.dropout,
                    partial_bn=not parser_args.no_partialbn,
                    freeze_eco=parser_args.freeze_eco)
    elif "resnet" in parser_args.arch:
        from models_tsm import TSN
        fc_lr5_temp = not (
            parser_args.finetune_model
            and parser_args.dataset in parser_args.finetune_model)
        model = TSN(
            parser_args.num_classes,
            parser_args.num_segments,
            parser_args.modality,
            base_model=parser_args.arch,
            consensus_type=parser_args.consensus_type,
            dropout=parser_args.dropout,
            img_feature_dim=parser_args.img_feature_dim,
            partial_bn=not parser_args.no_partialbn,
            pretrain=parser_args.pretrain,
            is_shift=parser_args.shift,
            shift_div=parser_args.shift_div,
            shift_place=parser_args.shift_place,
            fc_lr5=fc_lr5_temp,
            temporal_pool=parser_args.temporal_pool,
            non_local=parser_args.non_local)
    elif parser_args.arch == "ECOfull_py":
        from models_ecopy import ECOfull
        model = ECOfull(
            num_classes=parser_args.num_classes,
            num_segments=parser_args.num_segments,
            modality=parser_args.modality,
            freeze_eco=parser_args.freeze_eco,
            freeze_interval=parser_args.freeze_interval)
    elif parser_args.arch == "ECOfull_efficient_py":
        from models_ecopy import ECOfull_efficient
        model = ECOfull_efficient(
            num_classes=parser_args.num_classes,
            num_segments=parser_args.num_segments,
            modality=parser_args.modality,
            freeze_eco=parser_args.freeze_eco,
            freeze_interval=parser_args.freeze_interval)
    return model


def get_train_and_testloader(parser_args):
    ############################################################
    # Model choosing
    model = get_model_for_loader(parser_args)

    ############################################################
    # Data loading code
    # TODO: Data loading to first model initialization
    train_augmentation = model.get_augmentation()
    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    if parser_args.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    if parser_args.modality == 'RGB':
        data_length = 1
    elif parser_args.modality in ['Flow', 'RGBDiff']:
        data_length = 5

    if parser_args.dataset == 'yfcc100m' or \
        parser_args.dataset == 'youtube8m':
        parser_args.classification_type = 'multilabel'
    else:
        parser_args.classification_type = 'multiclass'

    train_loader = torch.utils.data.DataLoader(
        TSNDataSet(parser_args.root_path,
                   parser_args.train_list,
                   num_segments=parser_args.num_segments,
                   new_length=data_length,
                   modality=parser_args.modality,
                   image_tmpl=parser_args.prefix,
                   classification_type=parser_args.classification_type,
                   num_labels=parser_args.num_classes,
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=True),
                       ToTorchFormatTensor(div=False),
                       normalize,
                   ])),
        batch_size=parser_args.batch_size, shuffle=True,
        num_workers=parser_args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        TSNDataSet(parser_args.root_path,
                   parser_args.val_list,
                   num_segments=parser_args.num_segments,
                   new_length=data_length,
                   modality=parser_args.modality,
                   image_tmpl=parser_args.prefix,
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=True),
                       ToTorchFormatTensor(div=False),
                       normalize,
                   ])),
        batch_size=parser_args.batch_size, shuffle=False,
        num_workers=parser_args.workers, pin_memory=True)
    return train_loader, val_loader
