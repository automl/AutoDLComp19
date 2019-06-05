#!/bin/bash

# This script sets up the data / symlinking / ... to run on all tasks

case $(uname -n) in
  kisbat*)
    # Symlink cluster storage of image datasets
    ln -s /data/aad/image_datasets/tf_records/converted_datasets datasets/image
    ;;
  *)
    echo "Not implemented for your cluster / machine yet"
    ;;
esac
