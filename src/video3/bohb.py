import os
import time
import shutil
# BOHB
from hpbandster.core.worker import Worker
# Project
from ops.temporal_shift import make_temporal_pool
from ops import dataset_config
from dataset import TSNDataSet
from transforms import Stack, ToTorchFormatTensor, GroupScale
from transforms import GroupCenterCrop, IdentityTransform, GroupNormalize
# from transforms import GroupMultiScaleCrop
# from transforms import GroupRandomHorizontalFlip
from opts import parser
# Torch
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_
from torch.nn.init import constant_, xavier_uniform_
# GPU/CPU statistics
import GPUtil
import psutil
from tf_model_zoo.compression_layers import *

##########################################################
parser_args = parser.parse_args()

class ChallengeWorker(Worker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        ############################################################
        # Global settings from main.py
        self.best_prec1 = 0
        temp = dataset_config.return_dataset(
            parser_args.dataset, parser_args.modality)
        parser_args.num_class = temp[0]
        parser_args.train_list = temp[1]
        parser_args.val_list = temp[2]
        parser_args.root_path = temp[3]
        parser_args.prefix = temp[4]

        ############################################################
        # Model choosing
        model = ()
        if parser_args.arch == "ECO" or parser_args.arch == "ECOfull":
            from models_eco import TSN
            model = TSN(parser_args.num_class,
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
                parser_args.num_class,
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
        elif  parser_args.arch == 'Dummy':
            from models_eco import TSN
            model = TSN(parser_args.num_class,
                        parser_args.num_segments,
                        parser_args.modality,
                        base_model=parser_args.arch,
                        consensus_type=parser_args.consensus_type,
                        dropout=parser_args.dropout,
                        partial_bn=not parser_args.no_partialbn)

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
            classification_type = 'multilabel'
        else:
            classification_type = 'multiclass'

        self.train_loader = torch.utils.data.DataLoader(
            TSNDataSet(parser_args.root_path,
                       parser_args.train_list,
                       num_segments=parser_args.num_segments,
                       new_length=data_length,
                       modality=parser_args.modality,
                       image_tmpl=parser_args.prefix,
                       classification_type=classification_type,
                       num_labels=parser_args.num_class,
                       transform=torchvision.transforms.Compose([
                           train_augmentation,
                           Stack(roll=True),
                           ToTorchFormatTensor(div=False),
                           normalize,
                       ])),
            batch_size=parser_args.batch_size, shuffle=True,
            num_workers=parser_args.workers, pin_memory=True)
        self.val_loader = torch.utils.data.DataLoader(
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
                           # (Stack(roll=(parser_args.arch == 'C3DRes18') or
                           #     (parser_args.arch == 'ECO') or
                           #     (parser_args.arch == 'ECOfull') or
                           #     (parser_args.arch == 'ECO_2FC'))),
                           # ToTorchFormatTensor(div=(
                           #      parser_args.arch != 'C3DRes18')
                           #      and (parser_args.arch != 'ECO')
                           #     and (parser_args.arch != 'ECOfull')
                           #     and (parser_args.arch != 'ECO_2FC')),
                           normalize,
                       ])),
            batch_size=parser_args.batch_size, shuffle=False,
            num_workers=parser_args.workers, pin_memory=True)

    def compute(self, config, budget, working_directory, *args, **kwargs):  # noqa: C901
        """ Function that runs one model on budget with config and
        gets loss"""
        ############################################################
        # Initialisation
        start_time = time.time()
        ############################################################
        # Parameters
        budget_counter = budget
        ############################################################
        if parser_args.arch == "ECO" or parser_args.arch == "ECOfull" or parser_args.arch == 'Dummy':
            from models_eco import TSN
            model = TSN(parser_args.num_class,
                        parser_args.num_segments,
                        parser_args.modality,
                        base_model=parser_args.arch,
                        consensus_type=parser_args.consensus_type,
                        dropout=config['dropout'],
                        partial_bn=not parser_args.no_partialbn,
                        freeze_eco=parser_args.freeze_eco)
        elif "resnet" in parser_args.arch:
            from models_tsm import TSN
            fc_lr5_temp = (not (parser_args.finetune_model
                                and parser_args.dataset
                                in parser_args.finetune_model))
            model = TSN(parser_args.num_class,
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
                        non_local=parser_args.non_local)

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
        # model = torch.nn.DataParallel(
        #                  model, device_ids=parser_args.gpus).cuda()
        model_dict = model.state_dict()
        ############################################################
        # define loss function (criterion) and optimizer
        if parser_args.loss_type == 'nll':
            criterion = torch.nn.CrossEntropyLoss().cuda()
            if parser_args.print:
                print("Using CrossEntropyLoss")
        else:
            raise ValueError("Unknown loss type")
        # if parser_args.print:
        #     for group in policies:
        #         print('group: {} has {} params,'
        #                 ' lr_mult: {}, decay_mult: {}'.format(
        #                 group['name'],
        #                 len(group['params'],
        #                 group['lr_mult'],
        #                 group['decay_mult']))

        if parser_args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(policies,
                                        config['lr'],
                                        momentum=parser_args.momentum,
                                        weight_decay=parser_args.weight_decay,
                                        nesterov=parser_args.nesterov)
        if parser_args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(policies,
                                         config['lr'])

        model = torch.nn.DataParallel(
            model, device_ids=list(
                range(
                    torch.cuda.device_count()))).cuda()
        ############################################################
        # Model Training with resume option
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
                self.best_prec1 = checkpoint['best_prec1']
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
            try:
                ###########
                # ECO
                if parser_args.arch == "ECO" or parser_args.arch == "ECOfull":
                    new_state_dict = init_ECO(model_dict)
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
                                xavier_uniform_(new_state_dict[k])
                        elif 'bias' in k:
                            if parser_args.print:
                                print("{} init as: 0".format(k))
                            constant_(new_state_dict[k], 0)
                    if parser_args.print:
                        print("------------------------------------")
                    model.load_state_dict(new_state_dict)
                ###########
                # Resnets
                if "resnet" in parser_args.arch:
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
            except Exception:
                print('Failed to load Pretrained, continue with training'
                      'from scratch')
        ############################################################
        # Training
        if parser_args.temporal_pool and not parser_args.resume:
            make_temporal_pool(
                model.module.base_model,
                parser_args.num_segments)

        cudnn.benchmark = True

        if parser_args.evaluate:
            validate(self.val_loader, model, criterion, 0)
            return

        saturate_cnt = 0
        exp_num = 0

        # log_training = open(os.path.join(parser_args.snapshot_pref,
        #   'log.csv'), 'w')
        # with open(os.path.join(parser_args.snapshot_pref,
        #    'parser_args.txt'), 'w') as f:
        #    f.write(str(parser_args))

        # if budget below 1 training still must start, therefore +1
        epoches_to_train = parser_args.start_epoch + int(budget) + 1
        # if budget_counter is below one epoch increase it by 1
        if budget_counter < 1:
            budget_counter += 1
        ############################################################
        # Train and Validation loop
        for epoch in range(parser_args.start_epoch, epoches_to_train):
            if saturate_cnt == parser_args.num_saturate:
                exp_num = exp_num + 1
                saturate_cnt = 0
                if parser_args.print:
                    print("- Learning rate decreases by a factor"
                          " of '{}'".format(10 ** (exp_num)))
            adjust_learning_rate(
                optimizer, epoch, parser_args.lr_steps, exp_num)

            # train for one epoch
            budget_counter -= 1
            train_budget = 1 if budget_counter > 1 else budget_counter
            _ = train(self.train_loader,
                      model,
                      criterion,
                      optimizer,
                      epoch,
                      train_budget)
            ############################################################
            # Validation and saving
            # evaluate on validation set after last epoch
            # TODO: check formula below
            prec1, prec5, loss = validate(
                self.val_loader, model, criterion)
            # Count times where best precision was surpased to prevent
            # overfitting
            is_best = prec1 > self.best_prec1
            if is_best:
                saturate_cnt = 0
            else:
                saturate_cnt = saturate_cnt + 1
            if parser_args.print:
                print("- Validation Prec@1 saturates for {} epochs.".format(
                    saturate_cnt))
            if self.best_prec1 < prec1:
                self.best_prec1 = prec1
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': parser_args.arch,
                    'state_dict': model.state_dict(),
                    'best_prec1': self.best_prec1,
                    'lr': optimizer.param_groups[-1]['lr'],
                }, is_best)

        return {
            "loss": 100. - prec1,
            "info": {
                "train_time": time.time() - start_time,
                "prec1": prec1,
                "prec5": prec5,
                "loss": loss,
            },
        }


def train(train_loader, model, criterion, optimizer, epoch, budget):
    """ Function to train model on train loader with criterion and optimizer
        for budget epochs """
    # Batch and total averaging
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # In PyTorch 0.4, "volatile=True" is deprecated.
    torch.set_grad_enabled(True)

    if parser_args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    loss_summ = 0
    # switch to train mode
    model.train()
    # Time tracking variables
    end = time.time()
    localtime = time.localtime()
    end_time = time.strftime("%Y/%m/%d-%H:%M:%S", localtime)
    # budget is in range 0-1 so a perentage to scale one epoche
    # and discard final batch because it may not be full
    stop_batch = int((budget * len(train_loader))) - 1

    prune_ratio = 0.9
    prune_start = 10
    prune_end = 100
    prune_min_nonzero = 1000
    init_model_pruning(model, prune_start, prune_end, prune_ratio, prune_min_nonzero)

    for i, (input, target) in enumerate(train_loader):
        # break at budget
        if i == stop_batch:
            break
        # measure data loading time
        data_time.update(time.time() - end)

        # target size: [batch_size]
        target = target.cuda(non_blocking=True)  # noqa: W606
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output, output size: [batch_size, num_class]

        output = model(input_var)

        loss = criterion(output, target_var)

        loss = loss / parser_args.iter_size
        loss_summ += loss
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss_summ.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
        # update weights
        loss.backward()
        # show gpu usage all 20. batches
        if i % 20 == 0:
            GPUtil.showUtilization(all=True)
            print(psutil.virtual_memory())

        # scale down gradients when iter size is functioning
        if (i + 1) % parser_args.iter_size == 0:
            optimizer.step()
            optimizer.zero_grad()
            loss_summ = 0
        # Print intermediate results
        if parser_args.print:
            print(('Epoch: [{0}][{1}/{2}], lr: {lr:.7f}\t'
                   'Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                   'UTime {end_time:} \t'
                   'Data {data_time.val:.2f} ({data_time.avg:.2f})\t'
                   'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                   'Prec@1 {top1.val:.2f} ({top1.avg:.2f})\t'
                   'Prec@5 {top5.val:.2f} ({top5.avg:.2f})'.format(
                       epoch, i, stop_batch, batch_time=batch_time,
                       end_time=end_time, data_time=data_time,
                       loss=losses, top1=top1, top5=top5,
                       lr=optimizer.param_groups[-1]['lr'])))

        if parser_args.clip_gradient is not None:
            total_norm = clip_grad_norm_(
                model.parameters(), parser_args.clip_gradient)
            if total_norm > parser_args.clip_gradient:
                if parser_args.print:
                    print("clipping gradient: {} with coef {}".format(
                        total_norm, parser_args.clip_gradient / total_norm))

        # model compression
        print(i)
        prune_model(model, i)
        for module in model.modules():
            if isinstance(module, Compression):
                print('compression rate: ' + str(module.get_compression_ratio()))


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        localtime = time.localtime()
        end_time = time.strftime("%Y/%m/%d-%H:%M:%S", localtime)

    return end_time


def validate(val_loader, model, criterion, logger=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # In PyTorch 0.4, "volatile=True" is deprecated.
    torch.set_grad_enabled(False)

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        # discard final batch
        if i == int(len(val_loader) * parser_args.val_perc) - 1:
            break
        target = target.cuda(non_blocking=True)  # noqa: W606
        input_var = input
        target_var = target
        # show gpu usage all 20. batches
        if i % 20 == 0:
            GPUtil.showUtilization(all=True)
            print(psutil.virtual_memory())
        # compute output
        output = model(input_var)

        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if parser_args.print:
            if i % parser_args.print_freq == 0:
                print(('Test: [{0}/{1}]\t'
                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                       'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                       'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                           i,
                           len(val_loader),
                           batch_time=batch_time,
                           loss=losses,
                           top1=top1,
                           top5=top5)))
    if parser_args.print:
        print(('Testing Results: Prec@1 {:.3f} Prec@5 {:.3f} Loss {:.5f}'
               .format(top1.avg, top5.avg, losses.avg)))

    return top1.avg, top5.avg, losses.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, lr_steps, exp_num):
    """Sets the learning rate to the initial LR decayed
       by 10 every 30 epochs"""
    # decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    decay = 0.1 ** (exp_num)
    lr = parser_args.lr * decay
    decay = parser_args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def init_ECO(model_dict):
    # weight_url_2d = ('https://yjxiong.blob.core.windows.net/'
    #                  'ssn-models/bninception_rgb_kinetics_init'
    #                  '-d4ee618d3399.pth')

    if not os.path.exists(parser_args.finetune_model):

        new_state_dict = {}

    else:
        if parser_args.print:
            print(parser_args.finetune_model)
            print("88" * 40)
        if parser_args.finetune_model is not None:
            pretrained_dict = torch.load(parser_args.finetune_model)
            if parser_args.print:
                print(("=> loading model-finetune: '{}'".format(
                    parser_args.finetune_model)))
        else:
            pretrained_dict = torch.load(
                "pretrained_models/eco_fc_rgb_kinetics.pth.tar")
            if parser_args.print:
                print(("=> loading model-finetune-url: '{}'".format(
                    "pretrained_models/eco_fc_rgb_kinetics.pth.tar")))

        new_state_dict = {
            k: v for k, v in pretrained_dict['state_dict'].items() if
            (k in model_dict) and (v.size() == model_dict[k].size())}
        if parser_args.print:
            print("*" * 50)
            print("Start finetuning ..")

    return new_state_dict


def init_C3DRes18(model_dict):
    if parser_args.pretrained_parts == "scratch":
        new_state_dict = {}
    elif parser_args.pretrained_parts == "3D":
        pretrained_dict = torch.load(
            "pretrained_models/C3DResNet18_rgb_16F_kinetics_v1.pth.tar")
        new_state_dict = {
            k: v for k, v in pretrained_dict['state_dict'].items() if
            (k in model_dict) and (v.size() == model_dict[k].size())}
    else:
        raise ValueError(
            'For C3DRes18, "--pretrained_parts"'
            ' can only be chosen from [scratch, 3D]')

    return new_state_dict


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    filename = '_'.join(
        (parser_args.snapshot_pref, parser_args.modality.lower(),
         "epoch", str(state['epoch']), filename))
    torch.save(state, filename)
    if is_best:
        best_name = '_'.join(
            (parser_args.snapshot_pref,
             parser_args.modality.lower(),
             'model_best.pth.tar'))
        shutil.copyfile(filename, best_name)
