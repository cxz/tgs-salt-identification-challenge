#!/bin/bash

#for i in 0 1 2 3 4
#do
#    python train.py --name exp20 --model unet-resnet101 --lr 0.0001 --n-epochs 250 --fold $i --loss focal --focal-gamma 0.5
#done

N_EPOCHS=500
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

#for i in 0 1 2 3 4
#expname was 23 by mistake
#do 
#    python train.py --name exp23-dpn92-sample-weight --model unet-dpn92 --lr $LR --n-epochs $N_EPOCHS --fold $i --loss bce_jaccard --weighted-sampler
#done


#for i in 0 1 2 3 4
#do
#    python train.py --name exp33-resnet50-weighted-bce1-dice2 --model unet-resnet50 --lr $LR --n-epochs $N_EPOCHS --fold $i --loss bce_dice
#done

#for i in 0 1 2 3 4
#do
#    python train.py --name exp36-unet-dpn131-weighted-bce1-dice2 --model unet-dpn131 --lr $LR --n-epochs $N_EPOCHS --fold $i --loss bce_dice
#done

#for i in 0 1 2 3 4
#do
#    python train.py --name exp40-resnet152-weighted-bce1-dice3 --model unet-resnet152 --lr $LR --n-epochs $N_EPOCHS --fold $i --loss bce_dice
#done

#for i in 2 3 4 
#do
#    python train.py --name exp45-dpn107 --model unet-dpn107 --lr $LR --n-epochs $N_EPOCHS --fold $i --loss bce_dice --batch-size 24 --weighted-sampler
#    sleep 10
#done

#for i in 0 1 2 3 4
#do
#    #--weighted-sampler
#    python train.py --name exp48-dpn107 --model unet-dpn107 --lr $LR --n-epochs $N_EPOCHS --fold $i --loss lovasz --batch-size 24 --resume
#    sleep 10
#done


#same as 48 for folds 0 and 1.
#for i in 0 1
#do
#    #--weighted-sampler
#    python train.py --name exp49-dpn107 --model unet-dpn107 --lr $LR --n-epochs $N_EPOCHS --fold $i --loss lovasz --batch-size 24 --resume
#    sleep 10
#done
