#!/bin/bash

#for i in 0 1 2 3 4
#do
#    python train.py --name exp20 --model unet-resnet101 --lr 0.0001 --n-epochs 250 --fold $i --loss focal --focal-gamma 0.5
#done

N_EPOCHS=300
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


#too slow, didn't converge
#for i in 0 1 2 3 4
#do
#    python train.py --name exp50-wideresnet38 --model unet-wideresnet38 --lr $LR --n-epochs $N_EPOCHS --fold $i --loss bce_dice --batch-size 24 --weighted-sampler
#    sleep 10
#done


#for i in 1 2 3 4
#do
#    #--weighted-sampler
#    python train.py --name exp51-dpn107 --model unet-dpn107 --lr $LR --n-epochs $N_EPOCHS --fold $i --loss lovasz --batch-size 16 --resume
#    sleep 10
#done


#for i in 0 1 2 3 4
#do
#    #--weighted-sampler
#    #--num-channels 1
#    python train.py --name exp58 --model unet-dpn131 --lr $LR --n-epochs $N_EPOCHS --fold $i --loss bce_dice --batch-size 8
#    sleep 10
#done


#for i in 2 3 4
#do
#    python train.py --name exp66-ud --model unet-dpn107 --lr $LR --n-epochs $N_EPOCHS --fold $i --loss bce_dice --batch-size 24
#    sleep 10
#done

#for i in 0 1 2 3 4
#do
#    python train.py --name exp68-dpn107-small --model unet-dpn107 --lr $LR --n-epochs $N_EPOCHS --fold $i --loss lovasz  --batch-size 4 --resume
#    sleep 10
#done

#for i in 4
#do
#    #python train.py --name tmp2-unetheng --model heng34 --lr $LR --n-epochs 1000 --fold $i --loss lovasz  --batch-size 32 --iter-size 1 --weighted-sampler
#    python train.py --name tmp2-unetheng --model heng34 --lr 0.01 --n-epochs 200 --fold $i --loss lovasz  --batch-size 64 --iter-size 1 --weighted-sampler --resume
#    sleep 10
#done

#for i in 4 5 6 7 8 9
#do
#    python train.py --name exp75 --model unet-dpn107 --lr $LR --n-epochs 150 --fold 4 --loss lovasz --batch-size 16 --iter-size 2
#    sleep 10
#    cp ../data/runs/exp75/model_4.pth ../data/runs/exp75/model_4_snapshotx_$i.pth
#done


# removed cutout, changed border_mode, resuming with lr=1e-6
for i in 4
do
    python train.py --name exp77 --model unet-dpn107 --lr $LR --n-epochs 150 --fold 4 --loss lovasz --batch-size 4 --iter-size 8
done


