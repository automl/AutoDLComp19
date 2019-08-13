import time
from torch.utils.data import Dataset
import numpy as np
import tensorflow as tf
from utils import LOGGER


class TFDataset(Dataset):
    def __init__(self, session, dataset, num_samples=None, transform_sample=None, transform_label=None):
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

    def scan_all(self, max_samples=None):
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
        while count != max_samples:
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

        setattr(self, 'num_samples', (
            self.num_samples if max_samples is not None else count
        ))
        setattr(self, 'num_classes', label.shape[0])
        setattr(self, 'min_shape', min_shape)
        setattr(self, 'max_shape', max_shape)
        setattr(self, 'median_shape', np.median(shape_list, axis=0))
        setattr(self, 'mean_shape', np.mean(shape_list, axis=0))
        setattr(self, 'std_shape', np.std(shape_list, axis=0))
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
