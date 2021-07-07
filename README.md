# CCBANet

##  Introduction

This repository contains the PyTorch implementation of CCBANet,Cascading Context and Balancing Attention for Polyp Segmentation, MICCAI 2021.

##  Install dependencies

* torch
* torchvision 
* opencv
* Albumentations

##  Usage

####  1. Training

```bash
python train.py --dataset "dataset-name" --batch_size batch-size --load_ckpt "/path-to-check-point" --epoch_start epoch-start
```



####  2. Inference

```bash
python test.py --dataset "dataset-name" --batch_size 1 --load_ckpt "/path-to-check-point"
```



##  Acknowledgement
Part of the code was adpated from [ACSNet:Adaptive Context Selection for Polyp Segmentation](<https://github.com/ReaFly/ACSNet>)
