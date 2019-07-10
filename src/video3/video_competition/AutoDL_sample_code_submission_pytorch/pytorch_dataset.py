import random
from pathlib import Path
import pickle

import numpy as np
import torch
import torch.utils.data

class PytorchDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, nr_buffer_shards, transform_sample=None, transform_label=None):
        self._root_dir = Path(root_dir)

        self._is_training = self._root_dir.name == "train"

        self._transform_sample = transform_sample
        self._transform_label = transform_label

        #self._nr_buffer_shards = nr_buffer_shards 
        self._nr_total_shards = len(list(self._root_dir.glob("*.pickle")))
        print('ROOT DIR: ' + str(self._root_dir))
        print('TOTAL SHARDS: ' + str(self._nr_total_shards))
        self._nr_buffer_shards = self._nr_total_shards
        # TODO: random starting shards
        self._shards = [self.get_shard(idx) for idx in range(self._nr_buffer_shards)]

    def __len__(self):
        # The idx passed to __getitem__ will be ignored so output a random length for the
        # pytorch API
        return 10000  # TODO do not hardcode for chucky testset

    def __getitem__(self, idx):
        shard_index = random.randint(0, self._nr_buffer_shards - 1)
        shard = self._shards[shard_index]

        if len(shard) == 0:
            new_shard_index = random.randint(0, self._nr_total_shards - 1)
            shard = self.get_shard(new_shard_index)
            self._shards[shard_index] = shard

        # Labels and samples are stored seperately in the shard for training
        sample_index = random.randint(0, len(shard) - 1)
        sample = shard[sample_index]
        if self._is_training:  # TODO do proper testing
          del shard[sample_index]

        if self._is_training:
            sample, label = sample
            if self._transform_label:
                label = self._transform_sample(sample)

        if self._transform_sample:
            sample = self._transform_sample(sample)

        if self._is_training:
            return sample, label
        return sample

    def get_shard(self, idx):
        shard_path = self._root_dir / "{}.pickle".format(idx)
        return pickle.load(open(shard_path, "rb"))
