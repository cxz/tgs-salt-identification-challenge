#!/bin/bash

#for i in 0 1 2 3 4
#do
#    python train.py --name exp20 --model unet-resnet101 --lr 0.0001 --n-epochs 250 --fold $i --loss focal --focal-gamma 0.5
#done

N_EPOCHS=200
LR=0.0001

#for i in 0 1 2 3 4
#do
#    python train.py --name exp20-resnet101-lovasz --model unet-resnet101 --lr $LR --n-epochs $N_EPOCHS --fold $i --loss lovasz
#done

#for i in 0 1 2 3 4
#do
#    python train.py --name exp20-vgg11-lovasz --model unet-vgg11 --lr $LR --n-epochs $N_EPOCHS --fold $i --loss lovasz
#done

#for i in 0 1 2 3 4
#do
#    python train.py --name exp22-vgg16 --model unet-vgg16 --lr $LR --n-epochs $N_EPOCHS --fold $i --loss lovasz
#done

for i in 3 4 2
do
    python train.py --name exp23-resnet152 --model unet-resnet152 --lr $LR --n-epochs $N_EPOCHS --fold $i --loss lovasz
done

# python predict.py --path ../data/runs/exp23-resnet152
