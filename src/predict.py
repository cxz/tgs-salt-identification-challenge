"""
"""

import os
import argparse
from pathlib import Path
import glob
import json

import numpy as np
import cv2
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from albumentations import Compose, Normalize
from albumentations import HorizontalFlip, PadIfNeeded

from metric import iou_metric
import dataset
import models
import utils


def predict(model, ids, transform, kind):    
    if kind == 'test':
        loader = dataset.make_test_loader(ids, transform=transform)
    else:
        loader = dataset.make_val_loader(ids, transform=transform)
    preds = np.zeros((len(ids), 101, 101, 1), dtype=np.float32)
    pred_idx = 0
    with torch.no_grad():
        for batch_num, (inputs, paths) in enumerate(tqdm(loader, desc='predict')):
            inputs = utils.cuda(inputs)
            outputs = model(inputs)            
            for p in outputs:
                pred_mask = F.sigmoid(p).data.cpu().numpy()                
                preds[pred_idx, ..., 0] = pred_mask[0, 13:-14, 13:-14] # remove padding from loader
                pred_idx += 1
    return preds


def predict_tta(model, ids, output, kind='test'):
    size = dataset.SIZE
    base_transform = Compose([PadIfNeeded(min_height=size, min_width=size)])
    preds1 = predict(model, ids, transform=base_transform, kind=kind)
    
    flip_lr = Compose([HorizontalFlip(p=1), PadIfNeeded(min_height=size, min_width=size)])
    preds2 = predict(model, ids, transform=flip_lr, kind=kind)
    preds2 = preds2[:, :, ::-1, :]
    
    preds = np.mean([preds1, preds2], axis=0)
    np.save(output, preds)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--path', type=str, default='experiment folder', help='')
    #arg('--model_path', type=str, default='data/models/UNet', help='')
    #arg('--model_type', type=str, default='unet-resnet101', help='', choices=['unet-resnet101'])
    #arg('--output_path', type=str, help='', default='runs/inference')
    arg('--batch-size', type=int, default=32)
    #arg('--fold', type=int, default=-1, choices=[0, 1, 2, 3, -1], help='-1: all folds')    
    arg('--workers', type=int, default=8)
    args = parser.parse_args()
    
    
    with open(os.path.join(args.path, 'params.json'), 'r') as f:
        config = json.loads(f.read())
        
    model_type = config['model']

    test_ids = dataset.get_test_ids()
    for fold in range(5):
        print('processing fold ', fold)
        model_path = os.path.join(args.path, f"model_{fold}.pth")
        model = models.get_model(model_path, model_type=model_type)
        model.eval()
        print('loaded.')
        
        print('predicting val set')
        val_output = os.path.join(args.path, f"val_preds_fold{fold}.npy")
        _, val_ids = dataset.get_fold(fold)        
        predict_tta(model, val_ids, val_output, kind='val')
        
        print('predicting test set')
        test_output = os.path.join(args.path, f"test_preds_fold{fold}.npy")        
        predict_tta(model, test_ids, test_output)
