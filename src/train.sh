#!/bin/bash

for i in 0 1 2 3 4
do
    python train.py --name exp20 --model unet-resnet152 --lr 0.0001 --n-epochs 250 --fold $i    
done
