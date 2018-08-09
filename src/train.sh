#!/bin/bash

#for i in 0 1 2 3 4
#do
#    python train.py --name exp20 --model unet-resnet101 --lr 0.0001 --n-epochs 250 --fold $i --loss focal --focal-gamma 0.5
#done

N_EPOCHS=250
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

#for i in 0 1 2 3 4
#do
#    python train.py --name exp23-resnet152-bce-jaccard --model unet-resnet152 --lr $LR --n-epochs $N_EPOCHS --fold $i --loss bce_jaccard
#done

#for i in 0 1 2 3 4
#do
#    python train.py --name exp24-resnet50-bce-jaccard --model unet-resnet50 --lr $LR --n-epochs $N_EPOCHS --fold $i --loss bce_jaccard --resume
#done

#for i in 0 1 2 3 4
#do
#    python train.py --name exp25-dpn92-bce-jaccard --model unet-dpn92 --lr $LR --n-epochs $N_EPOCHS --fold $i --loss bce_jaccard --resume
#done

#for i in 0 1 2 3 4
#do
#    python train.py --name exp26-unet-incv3 --model unet-incv3 --lr $LR --n-epochs $N_EPOCHS --fold $i --loss bce_jaccard --resume
#done

#for i in 0 1 2 3 4
#do
#    python train.py --name exp27-unet-dpn92-224 --model unet-dpn92 --lr $LR --n-epochs $N_EPOCHS --fold $i --loss bce_jaccard --batch-size 16
#done

#for i in 0 1 2 3 4
#do
#    python train.py --name exp28-resnet101-bce-jaccard --model unet-resnet101 --lr $LR --n-epochs $N_EPOCHS --fold $i --loss bce_jaccard
#done

#for i in 0 1 2 3 4
#do
#    python train.py --name exp29-unet-serefinenet101 --model unet-serefinenet101 --lr $LR --n-epochs $N_EPOCHS --fold $i --loss bce_jaccard
#done

#for i in 0 1 2 3 4
#do
#    python train.py --name exp31-unet-dpn131 --model unet-dpn131 --lr $LR --n-epochs $N_EPOCHS --fold $i --loss bce_jaccard
#done

for i in 0 1 2 3 4
do 
    python train.py --name exp23-dpn92-sample-weight --model unet-dpn92 --lr $LR --n-epochs $N_EPOCHS --fold $i --loss bce_jaccard --weighted-sampler
done


