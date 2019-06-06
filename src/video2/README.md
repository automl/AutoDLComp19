
## Overview
This repository provides a PyTorch implementation of state-of-the-art models on video classification.



## Current available models:
* [x] TSM: Temporal Shift Module for Efficient Video Understanding [paper](https://arxiv.org/abs/1811.08383).
* [x] ECO: Efficient Convolutional Network for Online Video Understanding [paper](https://arxiv.org/abs/1804.09066).
* [ ] Timeception for Complex Action Recognition [paper](https://arxiv.org/pdf/1812.01289.pdf).
* [ ] SlowFast Networks for Video Recognition [paper](https://arxiv.org/abs/1812.03982).
* [ ] I3D [paper](https://arxiv.org/abs/1705.07750).

&nbsp;
&nbsp;

## Prerequisites
* [Python 3.5+](https://www.continuum.io/downloads)
* [PyTorch 0.3.0](http://pytorch.org/)

&nbsp;

## Usage

#### 1. Config datasets and pretrained models 
* Download pretrained models from this [link](https://drive.google.com/open?id=1bNUOYQhb4RF0TDKUNGCS3akypYtK92Ao). 
* Config paths in ops/dataset_config.py and make sure that your list correctly address video folders
* Modify path and parameters in run bash files.

#### 3. Train 
##### (i) Train
- To finetune model on UCF101 using TSM method run:
```
./run_TSM_UCF101_finetune.sh local

```
- To finetune model on UCF101 using ECO method and freezing specific layers run:
```
./run_ECO_UCF101_finetune_wfreezing.sh local

```


## Results
The performance with Pre-training on Kinetics:

| method      | UCF101   |     SMv2     | EpicK-Object  | EpicK-Action  |
| ----------  | -------  | ------------ | ------------- | ------------- |
| TSM         |   92.7%  |     --       |    34.9%      |   53.6%       |
| ECOfull(16F)|   92.1%  |     --       |    31.7%      |   58.0%       |
| SlowFast    |   -----  |     --       |    -----      |   -----       |
| Timeception |   -----  |     --       |    -----      |   -----       |



The performance with Pre-training on Youtube-8M:

| method      | UCF101   |     SMv2     | EpicK-Object  | EpicK-Action  |
| ----------  | -------  | ------------ | ------------- | ------------- |
| TSM         |   -----  |     --       |    -----      |   -----       |
| ECOfull(16F)|   -----  |     --       |    -----      |   -----       |
| SlowFast    |   -----  |     --       |    -----      |   -----       |
| Timeception |   -----  |     --       |    -----      |   -----       |



