import os

import torch
from torch.autograd import Variable
from torchvision import datasets, transforms

from workers.lib.wrn.model.wide_resnet import WideResNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(save_path):
    batch_size = 5
    # CIFAR 10 Training
    num_classes = 10
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize])
    test_dataset = datasets.CIFAR10(root='data/',
                                    train=False,
                                    transform=test_transform,
                                    download=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=2)

    cnn = WideResNet(depth=28, num_classes=num_classes, widen_factor=10,
                     dropRate=0.3)
    cnn.load_state_dict(torch.load(os.path.join(save_path, 'cifar10_wideresnet.pt'), map_location=device))
    # Set to test mode
    cnn.eval()
    correct = 0.
    total = 0.
    for images, labels in test_loader:
        images = Variable(images).to(device)
        labels = Variable(labels).to(device)

        pred = cnn(images)

        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels.data).sum().float()

    val_acc = correct.float() / total
    print(val_acc)


if __name__ == "__main__":
    load_model('/data/aad/AutoDLComp19/wideresnet_models/julien/84_0_0')
