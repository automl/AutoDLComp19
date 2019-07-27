import os
import torch
from torch.nn.init import constant_, xavier_uniform_


# from opts imporwwt parser
# parser_args = parser.parse_args()

def load_model_and_optimizer(parser_args, config):
    parser_args = parser_args
    ############################################################       
    # Apex usable?
    if parser_args.apex_available == True:
        from apex import amp

    ############################################################        
    if parser_args.arch == "ECO" or parser_args.arch == "ECOfull":
        from models_eco import TSN
        model = TSN(parser_args.num_classes,
                    parser_args.num_segments,
                    parser_args.modality,
                    base_model=parser_args.arch,
                    consensus_type=parser_args.consensus_type,
                    dropout=config['dropout'],
                    partial_bn=not parser_args.no_partialbn,
                    freeze_eco=parser_args.freeze_eco,
                    input_size=224)
    elif "TSM" in parser_args.arch:
        from models_tsm import TSN
        fc_lr5_temp = (not (parser_args.finetune_model
                            and parser_args.dataset
                            in parser_args.finetune_model))
        model = TSN(parser_args.num_classes,
                    parser_args.num_segments,
                    parser_args.modality,
                    base_model=parser_args.arch,
                    consensus_type=parser_args.consensus_type,
                    dropout=config['dropout'],
                    img_feature_dim=parser_args.img_feature_dim,
                    partial_bn=not parser_args.no_partialbn,
                    pretrain=parser_args.pretrain,
                    is_shift=parser_args.shift,
                    shift_div=parser_args.shift_div,
                    shift_place=parser_args.shift_place,
                    fc_lr5=fc_lr5_temp,
                    temporal_pool=parser_args.temporal_pool,
                    non_local=parser_args.non_local,
                    input_size=128)
    elif parser_args.arch == "ECOfull_py":
        from models_ecopy import ECOfull
        model = ECOfull(
            dropout=config['dropout'],
            num_classes=parser_args.num_classes,
            num_segments=parser_args.num_segments,
            modality=parser_args.modality,
            freeze_eco=parser_args.freeze_eco,
            freeze_interval=parser_args.freeze_interval,
            input_size=224)
    elif parser_args.arch == "ECOfull_efficient_py":
        from models_ecopy import ECOfull_efficient
        model = ECOfull_efficient(
            dropout=config['dropout'],
            num_classes=parser_args.num_classes,
            num_segments=parser_args.num_segments,
            modality=parser_args.modality,
            freeze_eco=parser_args.freeze_eco,
            freeze_interval=parser_args.freeze_interval,
            input_size=224)
    elif parser_args.arch == "Averagenet":
        from models_averagenet import Averagenet
        model = Averagenet(
            dropout=config['dropout'],
            num_classes=parser_args.num_classes,
            num_segments=parser_args.num_segments,
            modality=parser_args.modality,
            freeze=parser_args.freeze_eco,
            freeze_interval=parser_args.freeze_interval,
            input_size=128)
    elif parser_args.arch == "Averagenet_feature":
        from models_averagenet import Averagenet_feature
        model = Averagenet_feature(
            dropout=config['dropout'],
            num_classes=parser_args.num_classes,
            num_segments=parser_args.num_segments,
            modality=parser_args.modality,
            freeze=parser_args.freeze_eco,
            freeze_interval=parser_args.freeze_interval,
            input_size=128)
    ############################################################
    # Model Parameters

    # Optimizer s also support specifying per-parameter options.
    # To do this, pass in an iterable of dict s.
    # Each of them will define a separate parameter group,
    # and should contain a params key, containing a list of parameters
    # belonging to it.
    # Other keys should match the keyword arguments accepted by
    # the optimizers, and will be used as optimization options for this
    # group.
    policies = model.get_optim_policies()
    ############################################################
    # Load optimizer
    if parser_args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(policies,
                                    config['lr'],
                                    momentum=parser_args.momentum,
                                    weight_decay=parser_args.weight_decay,
                                    nesterov=parser_args.nesterov)
    if parser_args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(policies,
                                     config['lr'])

    if not parser_args.apex_available:
        model = torch.nn.DataParallel(model).cuda()
    ############################################################
    # Model Training with resume option
    model_dict = model.state_dict()
    #######################
    # Resume from checkpoint
    if parser_args.resume:
        if os.path.isfile(parser_args.resume):
            if parser_args.print:
                print(("=> loading checkpoint '{}'".format(
                    parser_args.resume)))
            checkpoint = torch.load(parser_args.resume)
            # if not checkpoint['lr']:
            if "lr" not in checkpoint.keys():
                text1 = "No 'lr' attribute found in resume model"
                text2 = ", please input the 'lr' manually: "
                parser_args.lr = input(text1 + text2)
                parser_args.lr = float(parser_args.lr)
            else:
                parser_args.lr = checkpoint['lr']
            parser_args.start_epoch = checkpoint['epoch']
            if checkpoint['best_prec1'] > parser_args.best_prec1:
                parser_args.best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            if parser_args.print:
                print(("=> loaded checkpoint '{}'"
                       " (epoch: {}, lr: {})".format(
                    parser_args.resume,
                    checkpoint['epoch'],
                    parser_args.lr)))
            else:
                if parser_args.print:
                    print(("=> no checkpoint found at '{}'".format(
                        parser_args.resume)))
    #######################
    # Load pretrained
    else:
        ###########
        # ECO
        if "ECO" in parser_args.arch:
            new_state_dict = init_ECO(model_dict, parser_args)
            un_init_dict_keys = [k for k in model_dict.keys() if k
                                 not in new_state_dict]
            if parser_args.print:
                print("un_init_dict_keys: ", un_init_dict_keys)
                print("\n------------------------------------")

            for k in un_init_dict_keys:
                new_state_dict[k] = torch.DoubleTensor(
                    model_dict[k].size()).zero_()
                if 'weight' in k:
                    if 'bn' in k:
                        if parser_args.print:
                            print("{} init as: 1".format(k))
                        constant_(new_state_dict[k], 1)
                    else:
                        if parser_args.print:
                            print("{} init as: xavier".format(k))
                        try:
                            xavier_uniform_(new_state_dict[k])
                        except Exception:
                            constant_(new_state_dict[k], 1)
                elif 'bias' in k:
                    if parser_args.print:
                        print("{} init as: 0".format(k))
                    constant_(new_state_dict[k], 0)
            if parser_args.print:
                print("------------------------------------")
            model.load_state_dict(new_state_dict)
        ###########
        # Resnets
        if "TSM" in parser_args.arch:
            if parser_args.print:
                print(("=> fine-tuning from '{}'".format(
                    parser_args.finetune_model)))
            sd = torch.load(parser_args.finetune_model)
            sd = sd['state_dict']
            model_dict = model.state_dict()
            replace_dict = []
            for k, v in sd.items():
                if k not in model_dict and k.replace(
                        '.net', '') in model_dict:
                    if parser_args.print:
                        print('=> Load after remove .net: ', k)
                    replace_dict.append((k, k.replace('.net', '')))
            for k, v in model_dict.items():
                if k not in sd and k.replace('.net', '') in sd:
                    if parser_args.print:
                        print('=> Load after adding .net: ', k)
                    replace_dict.append((k.replace('.net', ''), k))

            for k, k_new in replace_dict:
                sd[k_new] = sd.pop(k)
            keys1 = set(list(sd.keys()))
            keys2 = set(list(model_dict.keys()))
            set_diff = (keys1 - keys2) | (keys2 - keys1)
            if parser_args.print:
                print(
                    '#### Notice: keys that failed to load: {}'.format(
                        set_diff))
            if parser_args.dataset not in parser_args.finetune_model:
                if parser_args.print:
                    print('=> New dataset, do not load fc weights')
                sd = {k: v for k, v in sd.items() if 'fc' not in k}
            if (parser_args.modality == 'Flow'
                    and 'Flow' not in parser_args.finetune_model):
                sd = {k: v for k,
                               v in sd.items() if 'conv1.weight' not in k}
            model_dict.update(sd)
            model.load_state_dict(model_dict)
        ###########
        # Averagenet
        if "Averagenet" in parser_args.arch:
            new_state_dict = init_Averagenet(model_dict, parser_args)
            un_init_dict_keys = [k for k in model_dict.keys() if k
                                 not in new_state_dict]
            if parser_args.print:
                print("un_init_dict_keys: ", un_init_dict_keys)
                print("\n------------------------------------")

            for k in un_init_dict_keys:
                new_state_dict[k] = torch.DoubleTensor(
                    model_dict[k].size()).zero_()
                if 'weight' in k:
                    if 'bn' in k:
                        if parser_args.print:
                            print("{} init as: 1".format(k))
                        constant_(new_state_dict[k], 1)
                    else:
                        if parser_args.print:
                            print("{} init as: xavier".format(k))
                        try:
                            xavier_uniform_(new_state_dict[k])
                        except Exception:
                            constant_(new_state_dict[k], 1)
                elif 'bias' in k:
                    if parser_args.print:
                        print("{} init as: 0".format(k))
                    constant_(new_state_dict[k], 0)
            if parser_args.print:
                print("------------------------------------")
            model.load_state_dict(new_state_dict)

    # TODO: APEX WITH Dataparalell!?!
    if parser_args.apex_available:
        model = model.cuda()
        model, optimizer = amp.initialize(
            model, optimizer, opt_level="O2",
            keep_batchnorm_fp32=True, loss_scale="dynamic"
        )
        # Apex seems to have a problem loading pretrained
        # therefore again load model to gpu
        # return model.cuda(), optimizer
    return model, optimizer


def load_loss_criterion(parser_args):
    if (parser_args.loss_type == 'nll' and parser_args.classification_type == 'multiclass'):
        criterion = torch.nn.CrossEntropyLoss().cuda()
        if parser_args.print:
            print("Using CrossEntropyLoss")
    elif (parser_args.loss_type == 'nll' and parser_args.classification_type == 'multilabel'):
        criterion = torch.nn.BCEWithLogitsLoss().cuda()
        if parser_args.print:
            print("Using SigmoidBinaryCrossEntropyLoss")
    else:
        raise ValueError("Unknown loss type")

    return criterion


def init_ECO(model_dict, parser_args):
    weight_url_2d = ('https://yjxiong.blob.core.windows.net/ssn-models'
                     '/bninception_rgb_kinetics_init-d4ee618d3399.pth')

    if not os.path.exists(parser_args.finetune_model):
        print('Path or model file does not exist, can not load pretrained')
        new_state_dict = {}

    else:
        print(parser_args.finetune_model)
        print("88" * 40)
        if parser_args.finetune_model is not None:
            pretrained_dict = torch.load(parser_args.finetune_model)
            if parser_args.print:
                print(("=> loading model-finetune: '{}'".format(
                    parser_args.finetune_model)))
        else:
            pretrained_dict = torch.load("pretrained_models"
                                         "/eco_fc_rgb_kinetics.pth.tar")

            if parser_args.print:
                print(
                    (
                        "=> loading model-finetune-url: '{}'".
                            format("pretrained_models/eco_fc_rgb_kinetics.pth.tar")
                    )
                )
        new_state_dict = {
            k: v
            for k, v in pretrained_dict['state_dict'].items()
            if (k in model_dict) and (v.size() == model_dict[k].size())
        }
        print("*" * 50)
        print("Start finetuning ..")

    return new_state_dict


def init_Averagenet(model_dict, parser_args):
    weight_url_2d = ('https://yjxiong.blob.core.windows.net/ssn-models'
                     '/bninception_rgb_kinetics_init-d4ee618d3399.pth')

    if not os.path.exists(parser_args.finetune_model):
        print('Path or model file does not exist, can not load pretrained')
        new_state_dict = {}

    else:
        print(parser_args.finetune_model)
        print("88" * 40)
        if parser_args.finetune_model is not None:
            pretrained_dict = torch.load(parser_args.finetune_model)
            if parser_args.print:
                print(("=> loading model-finetune: '{}'".format(
                    parser_args.finetune_model)))
        else:
            pretrained_dict = torch.load("pretrained_models"
                                         "/bninception_rgb_kinetics_init-d4ee618d3399.pth")

            if parser_args.print:
                print(
                    (
                        "=> loading model-finetune-url: '{}'".
                            format("pretrained_models/eco_fc_rgb_kinetics.pth.tar")
                    )
                )
        new_state_dict = {}
        for k1, v in pretrained_dict['state_dict'].items():
            for k2 in model_dict.keys():
                k = k1.replace('module.base_model.', 'base.')
                if k2 in k and (v.size() == model_dict[k2].size()):
                    new_state_dict[k2] = v
        print("*" * 50)
        print("Start finetuning ..")

    return new_state_dict
