# -*- coding: utf-8 -*-
from __future__ import absolute_import
import logging
import time
import torch
from torch.utils.data import Dataset
import numpy as np
import tensorflow as tf
#from torch.nn.modules.hooks import MoveToHook


LOGGER = logging.getLogger(__name__)


class TFDataset(Dataset):
    def __init__(
        self,
        session,
        dataset,
        num_samples=None,
        transform=None
    ):
        super(TFDataset, self).__init__()
        self.session = session
        self.dataset = dataset
        self.transform = transform

        # Metadata
        self.num_samples = num_samples
        self.num_classes = None
        self.min_shape = None
        self.max_shape = None
        self.median_shape = None
        self.mean_shape = None
        self.std_shape = None
        self.is_multilabel = None

        self.next_idx = 0

        self.next_element = None
        self.reset()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, _):
        try:
            example, label = self._tf_exec(self.next_element)
            self.next_idx += 1
            # example = torch.as_tensor(example)
            # label = torch.as_tensor(example)
        except tf.errors.OutOfRangeError:
            self.reset()
            raise StopIteration

        example = self.transform(example) \
            if self.transform is not None \
            else example
        return example, label

    def _tf_exec(self, args):
        # Nice try but ingestion doesn't play nice with eager execution
        return args if tf.executing_eagerly() else self.session.run(args)

    def reset(self):
        dataset = self.dataset
        iterator = dataset.make_one_shot_iterator()
        self.next_element = iterator.get_next()
        self.next_idx = 0
        return self

    def scan(self, max_samples=None):
        # Same as scan but extracts the min/max shape and checks
        # if the dataset is multilabeled
        min_shape = (np.Inf, np.Inf, np.Inf, np.Inf)
        max_shape = (-np.Inf, -np.Inf, -np.Inf, -np.Inf)
        shape_list = []
        is_multilabel = False
        count = 0
        self.reset()
        while count != max_samples:
            try:
                example, label = self._tf_exec(self.next_element)
            except tf.errors.OutOfRangeError:
                self.reset()
                break
            shape_list.append(example.shape)
            min_shape = np.minimum(min_shape, example.shape)
            max_shape = np.maximum(max_shape, example.shape)
            count += 1
            if np.sum(label) > 1:
                is_multilabel = True

        self.num_samples = self.num_samples if max_samples is not None else count
        self.num_classes = label.shape[0]
        self.min_shape = min_shape
        self.max_shape = max_shape
        self.mean_shape = np.mean(shape_list, axis=0)
        self.is_multilabel = is_multilabel
        self.reset()

        return {
            'num_samples': self.num_samples,
            'num_classes': self.num_classes,
            'min_shape': self.min_shape,
            'max_shape': self.max_shape,
            'avg_shape': self.mean_shape,
            'is_multilabel': self.is_multilabel,
        }

