""" merge predictions and generate submission.
"""

import os
import sys

import glob
from pathlib import Path
import argparse

import cv2
import numpy as np
import pandas as pd
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

from super_pool import SuperPool

def load_train_mask(image_id):
    mask = cv2.imread(os.path.join('../input/train/masks', '%s.png' % image_id), 0)
    return (mask / 255.0).astype(np.uint8)


def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.mean(metric)


# ref.: https://www.kaggle.com/stainsby/fast-tested-rle
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def generate_submission(out_csv, preds):    
    pool = SuperPool()
    sample_df = pd.read_csv('../input/sample_submission.csv')
    test_ids = sample_df.id.values
    
    preds_encoded = pool.map(rle_encode, [x.T for x in preds])
    rows = [(image_id, p) for image_id, p in zip(test_ids, preds_encoded)]

    sub = pd.DataFrame(rows, columns=['id', 'rle_mask'])
    sub.to_csv(out_csv, index=False)

    
def main(write_submission=True):
    experiments = [
        '../data/subm023', 
        '../data/subm022', 
        '../data/subm020', 
        '../data/subm019'
    ]
    
    preds = np.zeros((18000, 101, 101, 1), dtype=np.float32)
    
    for fold in range(5):
        print('processing fold ', fold)
                        
        # merge fold predictions for val set
        val_preds = []
        for exp in experiments:
            val_preds.append(np.load(os.path.join(exp, f"val_preds_fold{fold}.npy")))
        
        # simple average
        val_preds = np.mean(val_preds, axis=0)
        
        # find threshold
        _, filenames = dataset.get_split(fold)
        masks = np.array([load_train_mask(image_id) for image_id in filenames])
        
        thres = np.linspace(0.0, 0.9, 20)
        thres_ioc = [iou_metric_batch(masks, np.int32(val_preds > t)) for t in thres]
        best_thres_tta = thres[np.argmax(thres_ioc)]
        print(f"fold {fold} -- iou: ", best_thres_tta, max(thres_ioc))  
        
        fold_preds = []
        for exp in experiments:
            fold_preds.append(np.load(os.path.join(exp, f"test_preds_fold{fold}.npy")))
            
        # simple average
        fold_preds = np.mean(fold_preds, axis=0)
        fold_preds_thresholded = (fold_preds > best_thres_tta).astype(np.uint8)
        preds += fold_preds_thresholded
    
    # majority voting
    final = np.round(preds/5.0).astype(np.uint8)
    
    if write_submission:
        output_csv = '../submissions/subm_023.csv'
        print('writing to ', output_csv)
        
        generate_submission(output_csv, final)
        print('done.')
    
if __name__ == '__main__':
    main()

    
    