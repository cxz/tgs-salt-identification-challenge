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
from albumentations import HorizontalFlip, PadIfNeeded, VerticalFlip, Resize

from metric import iou_metric
import dataset
import models
import utils


def predict(model, ids, transform, kind, batch_size=32):
    loader = dataset.make_loader(
        ids,
        shuffle=False,
        mode=kind,
        batch_size=batch_size,
        transform=transform)
    preds = np.zeros((len(ids), 101, 101, 1), dtype=np.float32)
    pred_idx = 0
    with torch.no_grad():
        for batch_num, (inputs, paths) in enumerate(tqdm(loader, desc='predict')):
            inputs = utils.cuda(inputs)
            outputs = model(inputs)            
            for p in outputs:
                pred_mask = F.sigmoid(p).data.cpu().numpy()                
                #preds[pred_idx, ..., 0] = pred_mask[0, 13:-14, 13:-14] # remove padding from loader
                preds[pred_idx, ..., 0] = cv2.resize(pred_mask[0, 27:-27, 27:-27], (101, 101)) # remove padding from loader
                pred_idx += 1
    return preds


def predict_tta(model, ids, output, kind='test', upside_down=False, batch_size=32):
    size = dataset.SIZE
    base_transform = Compose([
        Resize(202, 202, interpolation=cv2.INTER_NEAREST), #experiment with input 256
        VerticalFlip(p=int(upside_down)),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_REPLICATE),
        Normalize()])
    preds1 = predict(model, ids, transform=base_transform, kind=kind, batch_size=batch_size)
    
    flip_lr = Compose([
        Resize(202, 202, interpolation=cv2.INTER_NEAREST), #experiment with input 256
        VerticalFlip(p=int(upside_down)), HorizontalFlip(p=1),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_REPLICATE),
        Normalize()])
    preds2 = predict(model, ids, transform=flip_lr, kind=kind, batch_size=batch_size)
    preds2 = preds2[:, :, ::-1, :]
    
    preds = np.mean([preds1, preds2], axis=0)
    if upside_down:
        preds = preds[:, ::-1, :, :]
    np.save(output, preds)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--path', type=str, default='experiment folder', help='')
    arg('--batch-size', type=int, default=32)
    arg('--fold', type=int, default=-1, choices=[0, 1, 2, 3, 4, -1], help='-1: all folds')    
    arg('--workers', type=int, default=8)
    args = parser.parse_args()

    with open(os.path.join(args.path, 'params.json'), 'r') as f:
        config = json.loads(f.read())
        
    model_type = config['model']

    test_ids = dataset.get_test_ids()
    
    folds = list(range(5)) if args.fold == -1 else [args.fold]
    for fold in folds:
        print('processing fold ', fold)
        model_path = os.path.join(args.path, f"model_{fold}.pth")
        model = models.get_model(model_path, model_type=model_type)
        model.eval()
        print('loaded.')

        print('predicting val set')
        val_output = os.path.join(args.path, f"val_preds_fold{fold}.npy")
        _, val_ids = dataset.get_split(fold)
        predict_tta(model, val_ids, val_output, kind='val', batch_size=args.batch_size)

        print('predicting test set')
        test_output = os.path.join(args.path, f"test_preds_fold{fold}.npy")        
        predict_tta(model, test_ids, test_output, batch_size=args.batch_size)
