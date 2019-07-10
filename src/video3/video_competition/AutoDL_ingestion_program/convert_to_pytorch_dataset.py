import numpy as np
import argparse
import os
import pickle
import torch
import tensorflow as tf
from dataset import AutoDLDataset, AutoDLMetadata

parser = argparse.ArgumentParser('np-converter')
parser.add_argument('--dataset_dir', type=str,
                    default='/home/zelaa/AutoDL/AutoDL-4/starter_autodl/AutoDL_public_data')
parser.add_argument('--dataset_name', type=str,
                    default='Munster')
parser.add_argument('--shard_size', type=int, default=128, help='number of datapoints in shards')
parser.add_argument('--use_lowercase', action='store_true', default=False, 
                   help="if $dataset.data inside $Dataset is lowercase")
args = parser.parse_args()

def main(args):
  dataset_dir = os.path.join(args.dataset_dir, args.dataset_name)
  if args.use_lowercase:
    args.dataset_name = args.dataset_name.lower()
  dataset_dir_tf = os.path.join(dataset_dir, args.dataset_name+'.data')
  dataset_dir_pt = os.path.join(dataset_dir, args.dataset_name+'.data.pytorch')

  for partition in ['train', 'test']:
    data_tf = os.path.join(dataset_dir_tf, partition)
    data_pt = os.path.join(dataset_dir_pt, partition)
    if not os.path.exists(data_pt):
      os.makedirs(data_pt, exist_ok=False)

    dataset = AutoDLDataset(data_tf)
    dataset_ = dataset.get_dataset()
    if partition == "train":
      dataset_ = dataset_.shuffle(buffer_size=int(dataset.get_metadata().size()))
    iterator = dataset_.make_one_shot_iterator()
    next_element = iterator.get_next()

    output_size = dataset.get_metadata().get_output_size()
    sess = tf.Session()
    data_shard = []
    shard_idx = 0
    for idx in range(dataset.get_metadata().size()):
      np_input, np_label = sess.run(next_element)
      data_shard.append((np_input, np_label) if partition == "train" else np_input)

      shard_filled = len(data_shard) >= args.shard_size
      if (idx == dataset.get_metadata().size() - 1) or shard_filled:
        print("Saving Shard {}".format(shard_idx))

        pickle.dump( data_shard, open( os.path.join(data_pt, "{}.pickle".format(shard_idx)), "wb" ) )
        data_shard = []
        shard_idx += 1

if __name__ == "__main__":
  main(args)
