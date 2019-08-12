# -*- coding: utf-8 -*-
from __future__ import absolute_import
import time
import logging
import torch
from torch.utils.data import Dataset
import numpy as np
import tensorflow as tf
from utils import LOGGER


class TFDataset(Dataset):
    def __init__(self, session, dataset, num_samples, transform_sample=None, transform_label=None):
        super(TFDataset, self).__init__()
        self.session = session
        self.dataset = dataset
        self.transform_sample = transform_sample
        self.transform_label = transform_label

        # Metadata
        self.num_samples = num_samples
        self.num_classes = None
        self.min_shape = None
        self.max_shape = None
        self.median_shape = None
        self.is_multilabel = None

        self.next_element = None
        self.reset()

    def reset(self):
        dataset = self.dataset
        iterator = dataset.make_one_shot_iterator()
        self.next_element = iterator.get_next()
        return self

    def __len__(self):
        return self.num_samples

    def __getitem__(self, _):
        session = self.session if self.session is not None else tf.Session()
        try:
            example, label = session.run(self.next_element)
            # example = torch.as_tensor(example)
            # label = torch.as_tensor(example)
        except tf.errors.OutOfRangeError:
            self.reset()
            raise StopIteration

        example = self.transform_sample(example) \
            if self.transform_sample is not None \
            else example
        label = self.transform_label(label) \
            if self.transform_label is not None \
            else label
        return example, label

    def scan_all(self):
        # Same as scan but extracts the min/max shape and checks
        # if the dataset is multilabeled
        # TODO(Philipp J.): Can we do better than going over the whole
        # to check this?
        session = self.session if self.session is not None else tf.Session()
        min_shape = (np.Inf, np.Inf, np.Inf, np.Inf)
        max_shape = (-np.Inf, -np.Inf, -np.Inf, -np.Inf)
        shape_list = []
        is_multilabel = False
        count = 0
        self.reset()
        while True:
            try:
                example, label = session.run(self.next_element)
            except tf.errors.OutOfRangeError:
                self.reset()
                break
            shape_list.append(example.shape)
            min_shape = np.minimum(min_shape, example.shape)
            max_shape = np.maximum(max_shape, example.shape)
            count += 1
            if np.sum(label) > 1:
                is_multilabel = True
        setattr(self, 'num_classes', label.shape[0])
        setattr(self, 'num_samples', count)
        setattr(self, 'min_shape', min_shape)
        setattr(self, 'max_shape', max_shape)
        setattr(self, 'median_shape', np.median(shape_list, axis=0))
        setattr(self, 'is_multilabel', is_multilabel)
        self.reset()

    def benchmark_transofrmations(self):
        session = self.session if self.session is not None else tf.Session()
        LOGGER.debug('STARTING TRANSFORMATION BENCHMARK')
        if self.transform_sample is None:
            LOGGER.debug('NO TRANSFORMATION TO BENCHMARK')
            return
        idx = 0
        self.reset()
        t_start = time.time()
        while True:
            try:
                example, label = session.run(self.next_element)
                idx += 1
            except tf.errors.OutOfRangeError:
                self.reset()
                break
        LOGGER.debug('SCAN WITHOUT TRANSFORMATIONS TOOK:\t{0:.6g}s'.format(time.time() - t_start))

        idx = 0
        self.reset()
        t_start = time.time()
        while True:
            try:
                example, label = session.run(self.next_element)
                idx += 1
            except tf.errors.OutOfRangeError:
                self.reset()
                break
            example = self.transform_sample(example) \
                if self.transform_sample is not None \
                else example
            label = self.transform_label(label) \
                if self.transform_label is not None \
                else label
        LOGGER.debug('SCAN WITH TRANSFORMATIONS TOOK:\t{0:.6g}s'.format(time.time() - t_start))

    def scan(self, samples=10000000, with_tensors=False, is_batch=False, device=None, half=False):
        shapes, counts, tensors = [], [], []
        for i in range(min(self.num_samples, samples)):
            try:
                example, label = self.__getitem__(i)
            except tf.errors.OutOfRangeError:
                break
            except StopIteration:
                break

            shape = example.shape
            count = np.sum(label, axis=None if not is_batch else -1)
            shapes.append(shape)
            counts.append(count)
            if with_tensors:
                example = torch.Tensor(example)
                label = torch.Tensor(label)

                example.data = example.data.to(device=device)
                if half and example.is_floating_point():
                    example.data = example.data.half()

                label.data = label.data.to(device=device)
                if half and label.is_floating_point():
                    label.data = label.data.half()

                tensors.append([example, label])

        shapes = np.array(shapes)
        counts = np.array(counts) if not is_batch else np.concatenate(counts)

        info = {
            'count': len(counts),
            'is_multilabel': counts.max() > 1.01,
            'example': {
                'shape': [int(v) for v in np.median(shapes, axis=0)],  # 1, width, height, channels
            },
            'label': {
                'min': counts.min(),
                'max': counts.max(),
                'average': counts.mean(),
                'median': np.median(counts),
            }
        }

        if with_tensors:
            return info, tensors
        return info


class TransformDataset(Dataset):
    def __init__(self, dataset, transform=None, index=None):
        self.dataset = dataset
        self.transform = transform
        self.index = index

    def __getitem__(self, index):
        tensors = self.dataset[index]
        tensors = list(tensors)

        if self.transform is not None:
            if self.index is None:
                tensors = self.transform(*tensors)
            else:
                tensors[self.index] = self.transform(tensors[self.index])

        return tuple(tensors)

    def __len__(self):
        return len(self.dataset)


def prefetch_dataset(dataset, num_workers=4, batch_size=32, device=None, half=False):
    if isinstance(dataset, list) and isinstance(dataset[0], torch.Tensor):
        tensors = dataset
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False, drop_last=False,
            num_workers=num_workers, pin_memory=False
        )
        tensors = [t for t in dataloader]
        tensors = [torch.cat(t, dim=0) for t in zip(*tensors)]

    if device is not None:
        tensors = [t.to(device=device) for t in tensors]
    if half:
        tensors = [t.half() if  t.is_floating_point() else t for t in tensors]

    return torch.utils.data.TensorDataset(*tensors)
