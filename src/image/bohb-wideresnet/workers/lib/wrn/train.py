# run train.py --dataset cifar10 --model resnet18 --data_augmentation --cutout --length 16

import argparse

import numpy as np
import os
import inspect
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets, transforms
from tqdm import tqdm

from workers.lib.wrn.model.resnet import ResNet18
from workers.lib.wrn.model.wide_resnet import WideResNet
from workers.lib.wrn.util.cutout import Cutout
from workers.lib.wrn.util.misc import CSVLogger
from workers.lib.wrn.util.mixup import mixup_data, mixup_criterion


def get_run_args(frame_args):
    args, _, _, values = frame_args
    config_dict = {arg: values[arg] for arg in args}
    return config_dict


def run_training(save_dir, epochs, learning_rate, alpha, weight_decay, run, dataset='cifar10',
                 model='wideresnet', batch_size=128, data_augmentation=False, cutout=False,
                 n_holes=1, length=8, no_cuda=False, seed=0):
    epochs = int(epochs)
    frame = inspect.currentframe()
    run_config = get_run_args(inspect.getargvalues(frame))

    cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True  # Should make training should go faster for large models

    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    test_id = dataset + '_' + model

    # print(args)

    # Image Preprocessing
    if dataset == 'svhn':
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [109.9, 109.7, 113.8]],
                                         std=[x / 255.0 for x in [50.1, 50.6, 50.8]])
    else:
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    train_transform = transforms.Compose([])
    if data_augmentation:
        train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
        train_transform.transforms.append(transforms.RandomHorizontalFlip())
    train_transform.transforms.append(transforms.ToTensor())
    train_transform.transforms.append(normalize)
    if cutout:
        train_transform.transforms.append(Cutout(n_holes=n_holes, length=length))

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize])

    if dataset == 'cifar10':
        num_classes = 10
        train_dataset = datasets.CIFAR10(root='data/',
                                         train=True,
                                         transform=train_transform,
                                         download=True)

        test_dataset = datasets.CIFAR10(root='data/',
                                        train=False,
                                        transform=test_transform,
                                        download=True)
    elif dataset == 'cifar100':
        num_classes = 100
        train_dataset = datasets.CIFAR100(root='data/',
                                          train=True,
                                          transform=train_transform,
                                          download=True)

        test_dataset = datasets.CIFAR100(root='data/',
                                         train=False,
                                         transform=test_transform,
                                         download=True)
    elif dataset == 'svhn':
        num_classes = 10
        train_dataset = datasets.SVHN(root='data/',
                                      split='train',
                                      transform=train_transform,
                                      download=True)

        extra_dataset = datasets.SVHN(root='data/',
                                      split='extra',
                                      transform=train_transform,
                                      download=True)

        # Combine both training splits (https://arxiv.org/pdf/1605.07146.pdf)
        data = np.concatenate([train_dataset.data, extra_dataset.data], axis=0)
        labels = np.concatenate([train_dataset.labels, extra_dataset.labels], axis=0)
        train_dataset.data = data
        train_dataset.labels = labels

        test_dataset = datasets.SVHN(root='data/',
                                     split='test',
                                     transform=test_transform,
                                     download=True)

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=2)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=2)

    if model == 'resnet18':
        cnn = ResNet18(num_classes=num_classes)
    elif model == 'wideresnet':
        if dataset == 'svhn':
            cnn = WideResNet(depth=16, num_classes=num_classes, widen_factor=8,
                             dropRate=0.4)
        else:
            cnn = WideResNet(depth=28, num_classes=num_classes, widen_factor=10,
                             dropRate=0.3)

    cnn = cnn.to(device)
    criterion = nn.CrossEntropyLoss()
    cnn_optimizer = torch.optim.SGD(cnn.parameters(), lr=learning_rate,
                                    momentum=0.9, nesterov=True, weight_decay=weight_decay)

    if dataset == 'svhn':
        scheduler = MultiStepLR(cnn_optimizer, milestones=[80, 120], gamma=0.1)
    else:
        scheduler = MultiStepLR(cnn_optimizer, milestones=[60, 120, 160], gamma=0.2)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(save_dir, test_id + str(run) + '.csv')
    csv_logger = CSVLogger(args=run_config, fieldnames=['epoch', 'train_acc', 'test_acc'], filename=filename)

    def test(loader):
        cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
        correct = 0.
        total = 0.
        for images, labels in loader:
            if dataset == 'svhn':
                # SVHN labels are from 1 to 10, not 0 to 9, so subtract 1
                labels = labels.type_as(torch.LongTensor()).view(-1) - 1

            images = Variable(images).to(device)
            labels = Variable(labels).to(device)

            pred = cnn(images)

            pred = torch.max(pred.data, 1)[1]
            total += labels.size(0)
            correct += (pred == labels.data).sum().float()

        val_acc = correct.float() / total
        cnn.train()
        return val_acc.detach().cpu().numpy()

    for epoch in range(epochs):

        xentropy_loss_avg = 0.
        correct = 0.
        total = 0.

        progress_bar = tqdm(train_loader)

        for i, (images, labels) in enumerate(progress_bar):
            progress_bar.set_description('Epoch ' + str(epoch))

            if cuda:
                images, labels = images.to(device), labels.to(device)

            if dataset == 'svhn':
                # SVHN labels are from 1 to 10, not 0 to 9, so subtract 1
                labels = labels.type_as(torch.LongTensor()).view(-1) - 1

            images, labels_a, labels_b, lam = mixup_data(images, labels, alpha, cuda)

            images = Variable(images).to(device)
            labels_a = Variable(labels_a).to(device)
            labels_b = Variable(labels_b).to(device)

            cnn.zero_grad()
            pred = cnn(images)

            loss_func = mixup_criterion(labels_a, labels_b, lam)
            xentropy_loss = loss_func(criterion, pred)
            xentropy_loss.backward()
            cnn_optimizer.step()

            xentropy_loss_avg += xentropy_loss.item()

            # Calculate running average of accuracy
            _, predicted = torch.max(pred.data, 1)
            total += labels.size(0)
            correct += lam * predicted.eq(labels_a.data).sum().float() + (1 - lam) * predicted.eq(
                labels_b.data).sum().float()
            accuracy = correct.float() / total

            progress_bar.set_postfix(
                xentropy='%.3f' % (xentropy_loss_avg / (i + 1)),
                acc='%.3f' % accuracy)

        test_acc = test(test_loader)
        tqdm.write('test_acc: %.3f' % (test_acc))

        scheduler.step(epoch)
        accuracy = accuracy.detach().cpu().numpy()

        row = {'epoch': str(epoch), 'train_acc': str(accuracy), 'test_acc': str(test_acc)}
        csv_logger.writerow(row)

    torch.save(cnn.state_dict(), os.path.join(save_dir, test_id + '.pt'))
    csv_logger.close()

    return ({
        'loss': float(1 - test_acc),  # remember: HpBandSter always minimizes!
        'info': {'test accuracy': float(test_acc),
                 'train accuracy': float(accuracy)}
    })


if __name__ == "__main__":
    model_options = ['resnet18', 'wideresnet']
    dataset_options = ['cifar10', 'cifar100', 'svhn']

    parser = argparse.ArgumentParser(description='CNN')
    parser.add_argument('--dataset', '-d', default='cifar10',
                        choices=dataset_options)
    parser.add_argument('--model', '-a', default='wideresnet',
                        choices=model_options)
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    # parser.add_argument('--epochs', type=int, default=200,
    #                     help='number of epochs to train (default: 20)')
    # parser.add_argument('--learning_rate', type=float, default=0.1,
    #                    help='learning rate')
    parser.add_argument('--data_augmentation', action='store_true', default=False,
                        help='augment data by flipping and cropping')
    parser.add_argument('--cutout', action='store_true', default=False,
                        help='apply cutout')
    parser.add_argument('--n_holes', type=int, default=1,
                        help='number of holes to cut out from image')
    parser.add_argument('--length', type=int, default=8,
                        help='length of the holes')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 1)')
    parser.add_argument('--run', type=int, default=1,
                        help='experiment number')
    # parser.add_argument('--alpha', type=float, default=1.,
    #                     help='interpolation strength (uniform=1., ERM=0.)')
    # parser.add_argument('--weight_decay', type=float, default=5e-4,
    #                     help='weight decay')
    args = parser.parse_args()
    run_training(save_dir='0_2', epochs=1, learning_rate=1e-4, alpha=1, weight_decay=1e-4, run=1)
