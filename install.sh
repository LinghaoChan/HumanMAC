#!/bin/bash

mkdir ./checkpoints
mkdir ./data
mkdir ./inference
mkdir ./results

source conda activate
conda create -n humanmac python=3.8
conda activate humanmac

pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirement.txt