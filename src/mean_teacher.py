import os
import argparse
import json
import uuid
from pathlib import Path
from validation import validation_binary
import time
import numpy as np
import tqdm

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.backends.cudnn as cudnn
import torch.backends.cudnn
from loss import LossBinary, FocalLoss, LossLovasz, BCEDiceJaccardLoss
import dataset
import models
import utils
import random
import validation
from loss2 import make_loss

from models import get_model
from predict import predict_tta


global_step = 0

LR = 1e-6
INITIAL_LR = 1e-5
CONSISTENCY = 100
CONSISTENCY_RAMPUP = 5
LR_RAMPDOWN_EPOCHS = 100
LR_RAMPUP = 2
EPOCHS = 50

BATCH_SIZE=32
LABELED_BATCH_SIZE=16

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))

def cuda(x):
    return x.cuda(async=True) if torch.cuda.is_available() else x


def create_model(model_name, snapshot=None):
    model = get_model(None, model_name)

    ema_model = get_model(None, model_name)

    if snapshot is not None:
        state = torch.load(str(snapshot))
        state = {key.replace('module.', ''): value for key, value in state['model'].items()}
        model.load_state_dict(state)
        ema_model.load_state_dict(state)

    #
    for param in ema_model.parameters():
        param.detach_()

    model = nn.DataParallel(model).cuda()
    ema_model = nn.DataParallel(ema_model).cuda()

    return model, ema_model


def create_data_loaders(fold, batch_size, workers):
    train_ids, val_ids = dataset.get_split(fold)

    labeled_size = len(train_ids)
    unlabeled_size = 18000
    sampler = dataset.TwoStreamBatchSampler(
        range(labeled_size),  # labeled ids
        list(range(labeled_size, labeled_size+unlabeled_size)),  # unlabeled ids
        batch_size,  # total batch size (labeled + unlabeled)
        LABELED_BATCH_SIZE  # labeled batch size # TODO: was .5
    )

    train_loader = dataset.DataLoader(
        dataset=dataset.MeanTeacherTGSDataset(train_ids, transform=dataset.train_transform(), mode='train'),
        # shuffle=True,
        num_workers=workers,
        batch_sampler=sampler,
        pin_memory=torch.cuda.is_available()
    )

    valid_loader = dataset.make_loader(
        val_ids,
        transform=dataset.val_transform(),
        shuffle=False,
        # batch_size=len(device_ids),
        batch_size=batch_size,  # len(device_ids),
        workers=workers)

    return train_loader, valid_loader


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def adjust_learning_rate(optimizer, epoch, step_in_epoch, total_steps_in_epoch):
    lr = LR
    epoch = epoch + step_in_epoch / total_steps_in_epoch

    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    lr = linear_rampup(epoch, LR_RAMPUP) * (LR - INITIAL_LR) + INITIAL_LR

    # Cosine LR rampdown from https://arxiv.org/abs/1608.03983 (but one cycle only)
    if LR_RAMPDOWN_EPOCHS:
        assert LR_RAMPDOWN_EPOCHS >= EPOCHS
        lr *= cosine_rampdown(epoch, LR_RAMPDOWN_EPOCHS)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return CONSISTENCY * sigmoid_rampup(epoch, CONSISTENCY_RAMPUP)


def save(model, optimizer, model_path, epoch, step, valid_best):    
    # epoch_path = "{}_epoch{}.pth".format(str(model_path), epoch)
    torch.save({
        'model': model.state_dict(),
        'epoch': epoch,
        'valid_best': valid_best,
        'optimizer': optimizer.state_dict(),
        'step': step,        
    }, str(model_path))
    

def train(train_loader, model, ema_model, optimizer, epoch):
    global global_step

    # consistency loss: pixel-wise bce
    # both labelled and unlabelled samples contribute to consistency loss
    consistency_criterion = torch.nn.BCELoss(weight=None, reduce=True)

    # for segmentation task, dice_loss
    #bce_weight = 1
    #dice_weight = 2
    #segmentation_criterion = make_loss(bce_weight, dice_weight)
    segmentation_criterion = LossLovasz()

    model.train()
    ema_model.train()

    mini_batch_size = BATCH_SIZE
    steps_per_epoch = len(train_loader) * mini_batch_size

    tq = tqdm.tqdm(total=(steps_per_epoch))
    tq.set_description('Epoch {}'.format(epoch))

    losses = []
    segmentation_losses = []
    consistency_losses = []

    smooth_mean = 10

    for i, ((main_input, ema_input, kind), targets) in enumerate(train_loader):
        #adjust_learning_rate(optimizer, epoch, i, len(train_loader))

        inputs = cuda(main_input)
        ema_inputs = cuda(ema_input)

        with torch.no_grad():
            targets = cuda(targets)

        ema_out = ema_model(ema_inputs)
        model_out = model(inputs)
        
        # segmentation loss only for labeled samples. target for unlabeled should be ignored.
        loss1 = segmentation_criterion(model_out[:LABELED_BATCH_SIZE], targets[:LABELED_BATCH_SIZE])
        loss2 = consistency_criterion(F.sigmoid(model_out), F.sigmoid(ema_out))
        consistency_weight = .5  # TODO: was .5
        loss = loss1 + consistency_weight * loss2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print(global_step, ' ', loss1.item(), ' ', loss2.item())

        global_step += 1
        ema_decay = 0.999
        update_ema_variables(model, ema_model, ema_decay, global_step)

        tq.update(main_input.size(0))
        losses.append(loss.item())
        segmentation_losses.append(loss1.item())
        consistency_losses.append(loss2.item())

        mean_loss = np.mean(losses[-smooth_mean:])
        mean_segmentation = np.mean(segmentation_losses[-smooth_mean:])
        mean_consistency = np.mean(consistency_losses[-smooth_mean:])
        tq.set_postfix(loss='{:.5f}'.format(mean_loss),
                       consistency='{:.5f}'.format(mean_consistency),
                       segmentation='{:.5f}'.format(mean_segmentation))

    tq.close()


from torch.optim.lr_scheduler import CosineAnnealingLR

def main(fold=0, batch_size=BATCH_SIZE, lr=1e-5, workers=4):
    global global_step
    train_loader, valid_loader = create_data_loaders(fold, batch_size, workers)

    # model_path = f'../data/runs/exp66-ud/model_{fold}.pth'
    model_path = f'../data/runs/tmp2-unetheng/model_{fold}.pth'
    
    # create model and ema_model
    # model, ema_model = create_model('unet-dpn107', model_path)
    model, ema_model = create_model('heng34', model_path)    

    criterion = LossLovasz()
    optimizer = Adam(model.parameters(), lr=lr)
    # optimizer = SGD(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, verbose=True, patience=10, min_lr=1e-7, factor=0.5)
    # scheduler = CosineAnnealingLR(optimizer, T_max=50)

    cudnn.benchmark = True

    valid_best = None
    valid_best_ema = None

    for epoch in range(EPOCHS):
        valid_metrics = validation.validation_binary(model, criterion, valid_loader)
        valid_metrics_ema = validation.validation_binary(ema_model, criterion, valid_loader)
        
        if valid_best is None:
            valid_best = valid_metrics['val_iou']
            valid_best_ema = valid_metrics_ema['val_iou']
            
        if valid_metrics['val_iou'] > valid_best:
            valid_best = valid_metrics['val_iou'] 
            path = "{}.student.pth".format(model_path[:-4])
            save(model, optimizer, path, epoch, global_step, valid_best)
            
        if valid_metrics_ema['val_iou'] > valid_best_ema:
            valid_best_ema = valid_metrics_ema['val_iou'] 
            path = "{}.teacher.pth".format(model_path[:-4])
            save(ema_model, optimizer, path, epoch, global_step, valid_best_ema)

        train(train_loader, model, ema_model, optimizer, epoch)

        scheduler.step(valid_metrics_ema['val_loss'])


def predict(path, kind='student', batch_size=BATCH_SIZE, fold=-1, workers=8):

    with open(os.path.join(path, 'params.json'), 'r') as f:
        config = json.loads(f.read())

    model_type = config['model']

    test_ids = dataset.get_test_ids()
    folds = list(range(5)) if fold == -1 else [fold]
    for fold in folds:
        print('processing fold ', fold)
        model_path = os.path.join(path, f"model_{fold}.{kind}.pth")
        model = models.get_model(model_path, model_type=model_type)
        model.eval()
        print('loaded.')

        print('predicting val set')
        val_output = os.path.join(path, f"val_preds_fold{fold}.npy")
        _, val_ids = dataset.get_split(fold)
        predict_tta(model, val_ids, val_output, kind='val', upside_down=True)

        print('predicting test set')
        test_output = os.path.join(path, f"test_preds_fold{fold}.npy")
        predict_tta(model, test_ids, test_output, kind='test', upside_down=True)


if __name__ == '__main__':
    main(fold=4, lr=1e-5)
    # predict('../data/runs/tmp', kind='student')
    #predict('../data/runs/exp66-ud', kind='teacher')
