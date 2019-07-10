#!/bin/bash

python download_public_datasets.py
python AutoDL_ingestion_program/convert_to_pytorch_dataset.py --dataset_dir $PWD/AutoDL_public_data --dataset_name Munster --use_lowercase
python AutoDL_ingestion_program/convert_to_pytorch_dataset.py --dataset_dir $PWD/AutoDL_public_data --dataset_name Chucky
python AutoDL_ingestion_program/convert_to_pytorch_dataset.py --dataset_dir $PWD/AutoDL_public_data --dataset_name Hammer
python AutoDL_ingestion_program/convert_to_pytorch_dataset.py --dataset_dir $PWD/AutoDL_public_data --dataset_name Pedro --use_lowercase
python AutoDL_ingestion_program/convert_to_pytorch_dataset.py --dataset_dir $PWD/AutoDL_public_data --dataset_name Decal

