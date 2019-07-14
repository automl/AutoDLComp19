from __future__ import print_function
from __future__ import division

from dataset import TSNDataSet
from transforms import Stack, ToTorchFormatTensor, GroupScale
from transforms import GroupCenterCrop, IdentityTransform, GroupNormalize
# from transforms import GroupMultiScaleCrop
# from transforms import GroupRandomHorizontalFlip
import torch, torchvision
import csv
import os
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import numpy as np


def get_train_and_testloader(parser_args):
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
    elif parser_args.arch == "ECOfull_py":
        from models_ecopy import ECOfull
        model = ECOfull(
            num_classes=parser_args.num_class,
            num_segments=parser_args.num_segments,
            modality=parser_args.modality,
            freeze_eco=parser_args.freeze_eco,
            freeze_interval=parser_args.freeze_interval)
    elif parser_args.arch == "ECOfull_efficient_py":
        from models_ecopy import ECOfull_efficient
        model = ECOfull_efficient(
            num_classes=parser_args.num_class,
            num_segments=parser_args.num_segments,
            modality=parser_args.modality,
            freeze_eco=parser_args.freeze_eco,
            freeze_interval=parser_args.freeze_interval)

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
                   num_labels=parser_args.num_class,
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


class DaliVideoLoader(Pipeline):
    def __init__(self, root_path, list_file, num_segments, num_labels, batch_size, transform, shuffle):
        super().__init__(batch_size=batch_size, num_threads=8, device_id=0, seed=16)
        print('init dali pipeline')
        self.pipe = DaliPipe(root_path, list_file, num_segments, batch_size, shuffle)
        print('init label fetcher')
        self.label_fetcher = DaliLabelFetcher(list_file, num_labels)
        print('build dali pipeline')
        self.pipe.build()
        self.transform = transform


    def __iter__(self):
        return self

    def __next__(self):
        x, ids =  self.pipe.run()
        labels = self.label_fetcher(ids)
        return x


class DaliPipe(Pipeline):
    def __init__(self, root_path, list_file, num_segments, batch_size, shuffle):
        super().__init__(batch_size=batch_size, num_threads=8, device_id=0, seed=16)
        filenames = self.init_filenames(root_path, list_file)
        self.input = ops.VideoReader(device="gpu", filenames=filenames, sequence_length=num_segments,
                                     shard_id=0, num_shards=1,
                                     random_shuffle=shuffle, initial_fill=batch_size * num_segments)

    def init_filenames(self, root_path, list_file):
        filenames = []
        with open(list_file, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')
            for row in reader:
                filenames.append(os.path.join(root_path, row[0][1:]))
        return filenames

    def define_graph(self):
        output = self.input(name="Reader")
        return output


class DaliLabelFetcher():
    def __init__(self, list_file, des_num_labels):
        print('fetching dali labels')
        self.data, self.num_labels = self._limit_labels(list_file, des_num_labels)

    def get(self, ids):
        '''
        get the labels as a torch tensor for either a single video id or a list of them
        '''
        if isinstance(ids, Iterable) and not isinstance(ids, str):
            label_tensor = torch.zeros([len(ids), self.num_labels])
            for i, id in enumerate(ids):
                label_tensor[i] = self._get_single_id(id)
        else:
            label_tensor = self._get_single_id(ids)

        return label_tensor

    def _get_single_id(self, id):
        '''
        get the labels as a torch tensor for a single video id (only internally called)
        '''
        label_tensor = torch.zeros(self.num_labels)
        if id in self.data:
            val = self.data[id]
            for idx, perc in zip(val[::2], val[1::2]):
                label_tensor[idx] = perc
            return label_tensor
        else:
            raise IndexError('unknown video id: ' + str(id))

    def _load_file(self, file):
        '''
        load file
        '''
        data = {}

        with open(file, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')
            for row in reader:
                data[row[0]] = [int(elem) - 1 if i % 2 == 0 else float(elem) for i, elem in enumerate(row[2:])]

        return data

    def _get_num_labels(self, data):
        '''
        get total number of different labels
        '''
        num_labels = 0
        for _, val in data.items():
            if val:
                num_labels = max(num_labels, max(val))
        return num_labels + 1

    def _get_most_occ_labels(self, data, des_num_labels):
        '''
        get the x most occurring labels in aascending order
        '''
        num_labels = self._get_num_labels(data)
        if des_num_labels > num_labels:
            return np.array(range(num_labels)), num_labels

        freq = np.zeros(num_labels)
        for _, val in data.items():
            if val:
                freq[val[::2]] += val[1::2]

        index = np.argsort(-freq)[:des_num_labels]
        return np.sort(index), des_num_labels

    def _realign_labels(self, data, index):
        '''
        realign the labels to be limited to the indices stored in 'index'
        '''
        index_dict = {}
        for i, val in enumerate(index):
            index_dict[val] = i

        for key, val in data.items():
            val_new = []
            if val:
                for idx, perc in zip(val[::2], val[1::2]):
                    if idx in index_dict:
                        val_new.append(index_dict[idx])
                        val_new.append(perc)
            data[key] = val_new

        return data

    def _limit_labels(self, list_file, des_num_labels):
        '''
        do not use all labels but only a subset of the most occurring ones
        '''
        data = self._load_file(list_file)
        index, num_labels = self._get_most_occ_labels(data, des_num_labels)
        data = self._realign_labels(data, index)

        return data, num_labels



if __name__ == "__main__":
    root_path = '/data/aad/video_datasets/yfcc100m/'
    list_file = '/data/aad/video_datasets/yfcc100m/train_dali.txt'
    num_segments = 20
    num_labels = 100
    batch_size = 1
    transform = None
    shuffle = True

    vl = DaliVideoLoader(root_path=root_path,
                         list_file=list_file,
                         num_segments=num_segments,
                         num_labels=num_labels,
                         batch_size=batch_size,
                         transform=transform,
                         shuffle=shuffle)















