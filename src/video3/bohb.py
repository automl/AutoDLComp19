import os
import shutil
import time

# GPU/CPU statistics
import GPUtil
# Torch
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
# BOHB
from hpbandster.core.worker import Worker
from ops import dataset_config
from ops.load_dataloader import get_train_and_testloader
from ops.load_models import *
# Project
from ops.temporal_shift import make_temporal_pool
from opts import parser
from torch.nn.utils import clip_grad_norm_

##############################################################################


class ChallengeWorker(Worker):
    def __init__(self, parser_args, **kwargs):
        super().__init__(**kwargs)
        ############################################################
        # Global settings from main.py
        self.parser_args = parser_args
        temp = dataset_config.return_dataset(parser_args.dataset, parser_args.modality)
        self.parser_args.num_class = temp[0]
        self.parser_args.train_list = temp[1]
        self.parser_args.val_list = temp[2]
        self.parser_args.root_path = temp[3]
        self.parser_args.prefix = temp[4]
        ############################################################
        # Data loading code
        self.train_loader, self.val_loader = get_train_and_testloader(parser_args)

    ##########################################################################
    def compute(self, config, budget, working_directory, *args, **kwargs):  # noqa: C901
        """ Function that runs one model on budget with config and
        gets loss"""
        ############################################################
        # Initialisation
        start_time = time.time()
        cudnn.benchmark = True
        ############################################################
        # Parameters
        budget_counter = budget
        # learning rate reduce params
        saturate_cnt = 0
        exp_num = 0
        ############################################################
        # Model, optimizer and criterion loading
        model, optimizer = load_model_and_optimizer(self.parser_args, config)
        criterion = load_loss_criterion(self.parser_args)

        # Logging
        if self.parser_args.training:
            with open(
                os.path.join(self.parser_args.working_directory, 'parser_args.txt'), 'w'
            ) as f:
                f.write(str(self.parser_args))
        ############################################################
        # Training
        if self.parser_args.temporal_pool and not self.parser_args.resume:
            make_temporal_pool(model.module.base_model, self.parser_args.num_segments)

        if self.parser_args.evaluate:
            validate(self.val_loader, model, criterion, 0)
            return

        # if budget below 1 training still must start, therefore +1
        epoches_to_train = self.parser_args.start_epoch + int(budget) + 1
        # if budget_counter is below one epoch increase it by 1
        if budget_counter < 1:
            budget_counter += 1
        ############################################################
        # Train and Validation loop
        for epoch in range(self.parser_args.start_epoch, epoches_to_train):
            if saturate_cnt == self.parser_args.num_saturate:
                exp_num = exp_num + 1
                saturate_cnt = 0
                if self.parser_args.print:
                    print(
                        "- Learning rate decreases by a factor"
                        " of '{}'".format(10**(exp_num))
                    )
            adjust_learning_rate(optimizer, epoch, self.parser_args, exp_num)
            # train for one epoch
            budget_counter -= 1
            train_budget = 1 if budget_counter > 1 else budget_counter
            _ = train(
                self.train_loader, model, criterion, optimizer, epoch, train_budget,
                self.parser_args
            )
            ############################################################
            # Validation and saving
            # evaluate on validation set after last epoch
            if self.parser_args.classification_type == 'multiclass':
                prec1, prec5, loss = validate(self.val_loader, model, criterion)
            if self.parser_args.classification_type == 'multilabel':
                prec1, precision, recall, loss = validate(
                    self.val_loader, model, criterion, self.parser_args
                )
            ############################################################
            # Count times where best precision was surpased to prevent
            # overfitting
            is_best = prec1 > self.parser_args.best_prec1
            if is_best:
                saturate_cnt = 0
            else:
                saturate_cnt = saturate_cnt + 1
            ############################################################
            # update best percision and save
            if self.parser_args.print:
                print("- Validation Prec@1 saturates for {} epochs.".format(saturate_cnt))
            if self.parser_args.best_prec1 < prec1:
                self.parser_args.best_prec1 = prec1
                save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'arch': self.parser_args.arch,
                        'state_dict': model.state_dict(),
                        'best_prec1': self.parser_args.best_prec1,
                        'lr': optimizer.param_groups[-1]['lr'],
                    }, is_best, parser_args
                )
        ############################################################
        if self.parser_args.classification_type == 'multiclass':
            return {
                "loss": 100. - prec1,
                "info":
                    {
                        "train_time": time.time() - start_time,
                        "prec1": prec1,
                        "prec5": prec5,
                        "loss": loss,
                    },
            }
        else:
            return {
                "loss": 100. - prec1,
                "info":
                    {
                        "train_time": time.time() - start_time,
                        "f2": prec1,
                        "precision": prec5,
                        "recall": loss,
                    },
            }


##############################################################################
##############################################################################
def train(train_loader, model, criterion, optimizer, epoch, budget, parser_args):
    """ Function to train model on train loader with criterion and optimizer
        for budget epochs """
    ######################################################
    # Variable setups
    # Batch and total averaging
    batch_time, losses = AverageMeter(), AverageMeter()
    data_time, top1 = AverageMeter(), AverageMeter()
    if parser_args.classification_type == 'multiclass':
        top5 = AverageMeter()
        true_pos, false_pos, false_neg, true_neg, precision, recall = (None, ) * 6
    # For multilabel calculate f2 scores
    elif parser_args.classification_type == 'multilabel':
        top5 = None  # fscore has no top-k
        true_pos, false_pos = AverageMeter(), AverageMeter()
        false_neg, true_neg = AverageMeter(), AverageMeter()
        precision, recall = AverageMeter(), AverageMeter()

    loss_summ = 0
    # Time tracking variables
    end = time.time()
    localtime = time.localtime()
    end_time = time.strftime("%Y/%m/%d-%H:%M:%S", localtime)
    # Calculate stop batch for budget
    # budget is in range 0-1 so a perentage to scale one epoche
    stop_batch = int((budget * len(train_loader)))
    ######################################################
    # Training enable and start training
    # In PyTorch 0.4, "volatile=True" is deprecated.
    torch.set_grad_enabled(True)

    if parser_args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    model.train()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if parser_args.classification_type == 'multiclass':
            loss, prec1, prec5 = train_inner(model, optimizer, criterion, input, target, i, parser_args)
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))
        elif parser_args.classification_type == 'multilabel':
            loss, prec1, p, rl = train_inner(model, optimizer, criterion, input, target, i, parser_args)
            top1.update(prec1.item())
            precision.update(p.item())
            recall.update(r.item())
        loss_summ += loss

        ######################################################
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        localtime = time.localtime()
        end_time = time.strftime("%Y/%m/%d-%H:%M:%S", localtime)
        ######################################################
        # Print results
        Printings(
            pr=parser_args.print,
            freq=parser_args.print_freq,
            train=True,
            c_type=parser_args.classification_type,
            batch=i,
            stop_batch=stop_batch,
            batch_time=batch_time,
            end_time=end_time,
            data_time=data_time,
            loss=losses,
            top1=top1,
            top5=top5,
            precision=precision,
            recall=recall,
            lr=optimizer.param_groups[-1]['lr'],
            epoch=epoch,
        )
        ######################################################
        # break at budget
        if i >= stop_batch:
            break
    return end_time


def train_inner(model, optimizer, criterion, input, target, i, parser_args):
    ######################################################
    # fit batch into Model
    # target size: [batch_size]
    target = target.cuda()  # noqa: W606
    input_var = torch.autograd.Variable(input)
    target_var = torch.autograd.Variable(target)
    # compute output, output size: [batch_size, num_class]
    output = model(input_var)
    loss = criterion(output, target_var)
    loss = loss / parser_args.iter_size
    # update average loss
    loss.backward()
    # scale down gradients when iter size is set
    if (i + 1) % parser_args.iter_size == 0:
        optimizer.step()
        optimizer.zero_grad()
        loss_summ = 0
    # clip gradients id set
    if parser_args.clip_gradient is not None:
        total_norm = clip_grad_norm_(model.parameters(), parser_args.clip_gradient)
        if total_norm > parser_args.clip_gradient:
            if parser_args.print:
                print(
                    "clipping gradient: {} with coef {}".format(
                        total_norm, parser_args.clip_gradient / total_norm
                    )
                )
    ######################################################
    # measure accuracy and record loss
    if parser_args.classification_type == 'multiclass':
        # measure accuracy and record loss
        x_val = int(min(output.data.shape[1], 5))
        prec1, precX = accuracy(output.data, target, topk=(1, x_val))
        return loss, prec1, precX
    elif parser_args.classification_type == 'multilabel':
        prec1, p, rl = f2_score(output.data, target, parser_args)
        return loss, prec1, p, rl


##############################################################################
##############################################################################
def validate(val_loader, model, criterion, parser_args, logger=None):
    ######################################################
    # Variable setups
    # Batch and total averaging
    batch_time, losses = AverageMeter(), AverageMeter()
    data_time, top1 = AverageMeter(), AverageMeter()
    if parser_args.classification_type == 'multiclass':
        top5 = AverageMeter()
        true_pos, false_pos, false_neg, true_neg, precision, recall = (None, ) * 6
    # For multilabel calculate f2 scores
    elif parser_args.classification_type == 'multilabel':
        top5 = None  # fscore has no top-k
        true_pos, false_pos = AverageMeter(), AverageMeter()
        false_neg, true_neg = AverageMeter(), AverageMeter()
        precision, recall = AverageMeter(), AverageMeter()
    # Time tracking variables
    end = time.time()
    localtime = time.localtime()
    end_time = time.strftime("%Y/%m/%d-%H:%M:%S", localtime)
    # Calculate stop batch for budget
    # budget is in range 0-1 so a perentage to scale one epoche
    stop_batch = int((parser_args.val_perc * len(val_loader)))
    ######################################################
    # Validation enable and start evaluation
    # In PyTorch 0.4, "volatile=True" is deprecated.
    torch.set_grad_enabled(False)
    # switch to evaluate mode
    model.eval()
    for i, (input, target) in enumerate(val_loader):
        ######################################################
        # fit batch into Model
        # target size: [batch_size]
        target = target.cuda()
        input_var = input
        target_var = target
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        ######################################################
        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        if parser_args.classification_type == 'multiclass':
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))
        if parser_args.classification_type == 'multilabel':
            prec1, p, rl = f2_score(output.data, target)
            top1.update(prec1.item())
            precision.update(p.item())
            recall.update(r.item())
        ######################################################
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        ######################################################
        # Print intermediate results
        Printings(
            pr=parser_args.print,
            freq=parser_args.print_freq,
            train=False,
            c_type=parser_args.classification_type,
            batch=i,
            stop_batch=stop_batch,
            batch_time=batch_time,
            loss=losses,
            top1=top1,
            top5=top5,
            precision=precision,
            recall=recall
        )
        ######################################################
        # break at budget
        if i >= stop_batch:
            break
    ######################################################
    # Print final results
    if parser_args.print:
        if parser_args.classification_type == 'multiclass':
            print(
                (
                    'Testing Results: Prec@1 {:.3f} Prec@5 {:.3f} Loss {:.5f}'.format(
                        top1.avg, top5.avg, losses.avg
                    )
                )
            )
        else:
            print(
                (
                    'Testing Results: F2@1 {:.3f} Prec@1 {:.3f}'
                    ' Rec@1 {:.3f} Loss {:.5f}'.format(
                        top1.avg, precision.avg, recall.avg, losses.avg
                    )
                )
            )

    if parser_args.classification_type == 'multiclass':
        return top1.avg, top5.avg, losses.avg
    if parser_args.classification_type == 'multilabel':
        return top1.avg, precision.avg, recall.avg, losses.avg


##############################################################################
##############################################################################
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


##############################################################################
##############################################################################
def adjust_learning_rate(optimizer, epoch, parser_args, exp_num):
    """Sets the learning rate to the initial LR decayed
       by 10 every 30 epochs"""
    # decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    decay = 0.1**(exp_num)
    lr = parser_args.lr * decay
    decay = parser_args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


##############################################################################
##############################################################################
def accuracy(output, target, topk=(1, )):
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


#############################################################################
##############################################################################
def f2_score(output, target, parser_args):
    """Computes the f2 score, precision and recall"""
    predictions = output
    labels = target
    maximum = predictions.max()
    minimum = predictions.min()
    # strech output of logits
    predictions = -5 + 10 * (predictions - minimum) / (maximum - minimum)
    # wrap probability function
    pred = torch.sigmoid(predictions)
    # TODO: if labels is a percentage it won't work :/
    act_pos_batch = torch.sum(labels)
    act_neg_batch = (parser_args.batch_size * parser_args.num_classes - act_pos_batch)
    true_pos_batch = torch.zeros((1), dtype=torch.float16).cuda()
    false_pos_batch = torch.zeros((1), dtype=torch.float16).cuda()
    # count true and false Positives
    for j in range(parser_args.batch_size):
        # for higher accuracy use batch scaling
        pred = predictions[j]
        maximum = pred.max()
        minimum = pred.min()
        pred = (pred - minimum) / (maximum - minimum)
        a = pred.gt(eval_percentage).byte()
        b1 = labels[j].eq(1).byte()
        b2 = labels[j].eq(0).byte()
        c = a & b1
        d = a & b2
        true_pos_batch += c.sum()
        false_pos_batch += d.sum()
    # update false and ture negatives
    true_pos.update(true_pos_batch)
    false_pos.update(false_pos_batch)
    false_neg.update(act_pos_batch - true_pos_batch)
    true_neg.update(act_neg_batch - false_pos_batch)
    # precision: tp/(tp+fp) percentage of correctly classified predicted
    precision = true_pos.sum / (true_pos.sum + false_pos.sum + 1E-9)
    # recall: tp/(tp+fn) percentage of positives correctly classified
    recall = true_pos.sum / (true_pos.sum + false_neg.sum + 1E-9)
    # F-score with beta=1
    # see Sokolova et al., 2006 "Beyond Accuracy, F-score and ROC:
    # a Family of Discriminant Measures for Performance Evaluation"
    # fscore <- 2*precision.neg*recall.neg/(precision.neg+recall.neg)
    f_score = 2 * precision * recall / (precision + recall + 1E-9)
    return f_score, precision, recall


##############################################################################
##############################################################################
def save_checkpoint(state, is_best, parser_args, filename='checkpoint.pth.tar'):
    if parser_args.training:
        filename = '_'.join(
            (
                parser_args.snapshot_pref, parser_args.modality.lower(), "epoch",
                str(state['epoch']), filename
            )
        )
        torch.save(state, filename)
    if is_best and not parser_args.training:
        best_name = '_'.join(
            (
                parser_args.snapshot_pref, parser_args.modality.lower(),
                'model_best.pth.tar'
            )
        )
        shutil.copyfile(filename, best_name)
    elif is_best and parser_args.training:
        filename = '_'.join(
            (
                parser_args.snapshot_pref, parser_args.modality.lower(),
                'model_best.pth.tar'
            )
        )
        torch.save(state, filename)


##############################################################################
##############################################################################
def Printings(pr, freq, train, c_type, batch, stop_batch, **kwargs):

    if pr and batch % freq == 0:
        # show gpu usage all parser_args.print_freq batches
        GPUtil.showUtilization(all=True)
        ######################################################
        # Training
        top1 = kwargs['top1']
        top5 = kwargs['top5']
        batch_time = kwargs['batch_time']
        # data_time = kwargs['data_time']
        loss = kwargs['loss']
        if train:
            lr = kwargs['lr']
            data_time = kwargs['data_time']
            epoch = kwargs['epoch']
            end_time = kwargs['end_time']
            if c_type == 'multiclass':
                print(
                    (
                        'Epoch: [{0}][{1}/{2}], lr: {lr:.7f}  '
                        'Time {batch_time.val:.2f} ({batch_time.avg:.2f})  '
                        'UTime {end_time:}  '
                        'Data {data_time.val:.2f} ({data_time.avg:.2f})  '
                        'Loss {loss.val:.3f} ({loss.avg:.3f})  '
                        'Prec@1 {top1.val:.2f} ({top1.avg:.2f})  '
                        'Prec@5 {top5.val:.2f} ({top5.avg:.2f})'.format(
                            epoch,
                            batch,
                            stop_batch,
                            batch_time=batch_time,
                            end_time=end_time,
                            data_time=data_time,
                            loss=loss,
                            top1=top1,
                            top5=top5,
                            lr=lr
                        )
                    )
                )
            # Multilabel
            else:
                p = kwargs['precision']
                r = kwargs['recall']
                print(
                    (
                        'Epoch: [{0}][{1}/{2}], lr: {lr:.7f}  '
                        'Time {batch_time.val:.2f} ({batch_time.avg:.2f})  '
                        'UTime {end_time:}  '
                        'Data {data_time.val:.2f} ({data_time.avg:.2f})  '
                        'Loss {loss.val:.3f} ({loss.avg:.3f})  '
                        'F2@1 {top1.val:.3f} ({top1.avg:.3f})  '
                        'Precision@1 {p.val:.3f} ({p.avg:.3f})  '
                        'Recall@1 {r.val:.3f} ({r.avg:.3f})'.format(
                            epoch,
                            batch,
                            stop_batch,
                            batch_time=batch_time,
                            end_time=end_time,
                            data_time=data_time,
                            loss=loss,
                            top1=top1,
                            p=kwargs['precision'],
                            r=kwargs['recall'],
                            lr=lr
                        )
                    )
                )
        ######################################################
        # Validation
        else:
            if c_type == 'multiclass':
                print(
                    (
                        'Test: [{0}/{1}]  '
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                        'Loss {loss.val:.4f} ({loss.avg:.4f})  '
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})  '
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                            batch,
                            stop_batch,
                            batch_time=batch_time,
                            loss=loss,
                            top1=top1,
                            top5=kwargs['top5']
                        )
                    )
                )
            # Multilabel
            else:
                print(
                    (
                        'Test: [{0}/{1}]  '
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                        'Loss {loss.val:.4f} ({loss.avg:.4f})  '
                        'F2@1 {top1.val:.3f} ({top1.avg:.3f})  '
                        'Precision@1 {p.val:.3f} ({p.avg:.3f})  '
                        'Recall@1 {r.val:.3f} ({r.avg:.3f})'.format(
                            batch,
                            stop_batch,
                            batch_time=batch_time,
                            loss=loss,
                            top1=top1,
                            p=kwargs['precision'],
                            r=kwargs['recall']
                        )
                    )
                )
