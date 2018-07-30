"""
"""

import os
import argparse
from pathlib import Path

import numpy as np
import cv2
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F

from albumentations import Compose, Normalize, PadIfNeeded, HorizontalFlip

import models
import dataset
import utils

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


def get_model(model_path, model_type='UNet11'):
    num_classes = 1
    model = models.get_model(model_path, model_type)
    model.eval()    
    return model


def predict(model, filenames, transform, output_path):    
    loader = dataset.make_test_loader(filenames, transform=transform)
    preds = np.zeros((len(filenames), 101, 101, 1), dtype=np.float32)
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


def predict_tta(model, filenames, output_path):
    size = dataset.SIZE
    base_transform = [PadIfNeeded(min_height=size, min_width=size)]
    preds1 = predict(model, filenames, Compose(base_transform), output_path)
    
    flip_lr = HorizontalFlip(p=1)
    preds2 = predict(model, filenames, Compose(base_transform + [flip_lr]), output_path)
    preds2 = preds2[:, :, ::-1, :]
    
    preds = np.mean([preds1, preds2], axis=0)
           
    output = os.path.join(output_path, 'preds.npy')
    np.save(output, preds)
    
def generate_submission():
    #pred = np.mean([fold0, fold1, fold2, fold3, fold4], axis=0)
    #sample_df = pd.read_csv('../input/sample_submission.csv')
    #test_ids = sample_df.id.values
    #rows = []
    #for image_id, p in zip(test_ids, pred):
    #    rows.append([image_id, rle_encode((p>.3).T)])
    #
    #sub = pd.DataFrame(rows, columns=['id', 'rle_mask'])
    #sub.to_csv('../submissions/tmp.csv', index=False)
    print('done.')

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model_path', type=str, default='data/models/UNet', help='')
    arg('--model_type', type=str, default='unet-resnet101', help='', choices=['unet-resnet101'])
    arg('--output_path', type=str, help='', default='runs/inference')
    arg('--batch-size', type=int, default=32)
    #arg('--fold', type=int, default=-1, choices=[0, 1, 2, 3, -1], help='-1: all folds')    
    arg('--workers', type=int, default=8)

    args = parser.parse_args()

    output_path = Path(args.output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    filenames = dataset.get_test_filenames()
    
    model = get_model(args.model_path, args.model_type)
    print(f"{args.model_path} loaded.")

    predict_tta(model, filenames, output_path)

