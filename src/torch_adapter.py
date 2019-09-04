import numpy as np
import tensorflow as tf
import torch
from torch.utils.data import Dataset


class TFAdapterSet(Dataset):
    def __init__(
        self,
        session,
        dataset,
        num_samples,
        batch_size,
        transformations=None,
        num_parallel_calls=None,
        buffer_size=None,
        pin_memory=False,
        drop_last=False,
        shuffle=False,  # currently not working
    ):
        self._session = session
        self._org_dataset = dataset
        self._num_samples = num_samples
        self._batch_size = batch_size
        self._dataset = None
        self._pin_memory = pin_memory
        self._shuffle = shuffle
        self._drop_last = drop_last
        self._num_parallel_calls = tf.data.experimental.AUTOTUNE if num_parallel_calls is None else num_parallel_calls
        self._buffer_size = tf.data.experimental.AUTOTUNE if buffer_size is None else buffer_size
        self._transformations = {'samples': lambda x: x, 'labels': lambda y: y}
        if transformations is not None:
            self._transformations.update(transformations)

        self._update_dataset()

        self._next_element = None
        self.next_idx = 0
        self.reset()

    def _update_dataset(self):
        def transform(x, y):
            ret = (
                # For some reason this does not work if not
                # used from the original dict, meaning:
                # self._transform_samples = transformations['samples']
                # and then invoking
                # self._transform_samples(x)
                # fails
                self._transformations['samples'](x),
                self._transformations['labels'](y),
            )
            return ret

        def tfwrap(x, y):
            ret = tf.py_func(transform, [x, y], [tf.uint8, tf.int64])
            return ret

        self._dataset = self._org_dataset.prefetch(buffer_size=self._buffer_size, )
        self._dataset = self._dataset.map(
            tfwrap, num_parallel_calls=self._num_parallel_calls
        )
        self._dataset = self._dataset.batch(
            batch_size=self._batch_size, drop_remainder=self._drop_last
        )

    @property
    def dataset(self):
        return self._org_dataset

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, val):
        self._batch_size = val
        self._update_dataset()

    @property
    def drop_last(self):
        return self._drop_last

    @drop_last.setter
    def drop_last(self, val):
        self._drop_last = val
        self._update_dataset()

    @property
    def transform_sample(self):
        return self._transformations['samples']

    @transform_sample.setter
    def transform_sample(self, val):
        self.self._transformations['samples'] = lambda x: x if val is None else val
        self._update_dataset()

    @property
    def transform_label(self):
        return self._transformations['labels']

    @transform_label.setter
    def transform_label(self, val):
        self.self._transformations['labels'] = lambda y: y if val is None else val
        self._update_dataset()

    @property
    def num_parallel_calls(self):
        return self._num_parallel_calls

    @num_parallel_calls.setter
    def num_parallel_calls(self, val):
        self._num_parallel_calls = val
        self._update_dataset()

    def __len__(self):
        return self._num_samples

    def __getitem__(self, _):
        try:
            if self.next_idx >= self._num_samples:
                self.reset()
                raise StopIteration
            sample, label = self._session.run(self._next_element)
            self.next_idx += self._batch_size
        except tf.errors.OutOfRangeError:
            self.reset()
            raise StopIteration
        # NOTE(Philipp): Maybe move this into transformations as well
        # though I'm not sure if possible
        sample = torch.from_numpy(sample).pin_memory(
        ) if self._pin_memory else torch.from_numpy(sample)
        label = torch.from_numpy(label).pin_memory(
        ) if self._pin_memory else torch.from_numpy(label)
        return sample, label

    def reset(self):
        iterator = self._dataset.make_one_shot_iterator()
        self._next_element = iterator.get_next()
        self.next_idx = 0
        return self


class TFDataset(Dataset):
    def __init__(
        self,
        session,
        dataset,
        num_samples=None,
        transform_sample=None,
        transform_label=None
    ):
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

        example = self.transform_sample(example) \
            if self.transform_sample is not None \
            else example
        label = self.transform_label(label) \
            if self.transform_label is not None \
            else label
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
        self.median_shape = np.median(shape_list, axis=0)
        self.mean_shape = np.mean(shape_list, axis=0)
        self.std_shape = np.std(shape_list, axis=0)
        self.is_multilabel = is_multilabel = is_multilabel
        self.reset()
