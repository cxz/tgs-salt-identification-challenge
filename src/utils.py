import json
from datetime import datetime
from pathlib import Path

import random
import numpy as np
import pandas as pd

import torch
import tqdm
from torch import nn


def fold_snapshot(output_dir, fold):
    fname = os.path.join(output_dir, f"model_{fold}.pth")
    return fname if os.path.exists(fname) else None

def cuda(x):
    return x.cuda(async=True) if torch.cuda.is_available() else x


def get_learning_rate(optimizer):
    return optimizer.param_groups[0]['lr']


def save(model, optimizer, model_path, epoch, step, valid_best):    
    # epoch_path = "{}_epoch{}.pth".format(str(model_path), epoch)
    torch.save({
        'model': model.state_dict(),
        'epoch': epoch,
        'valid_best': valid_best,
        'optimizer': optimizer.state_dict(),
        'step': step,        
    }, str(model_path))


def train(experiment, output_dir, args, model, criterion, scheduler, train_loader, valid_loader, validation, optimizer, n_epochs=None, fold=None, batch_size=None, snapshot=None):
    if snapshot:
        state = torch.load(snapshot)
        epoch = state['epoch']
        step = state['step']
        valid_best = state['valid_best']
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        print('Restored model, epoch {}, step {:,}, valid_best {}'.format(epoch, step, valid_best))
    else:
        epoch = 1
        step = 0
        valid_best = None
    
    model_path = output_dir / 'model_{fold}.pth'.format(fold=fold)
    
    scores_fname = output_dir / 'scores.csv'
    scores = scores = pd.read_csv(scores_fname).values.tolist() if scores_fname.exists() else []

    steps_per_epoch = len(train_loader) * batch_size
    smooth_mean = 10
    
    for epoch in range(epoch, n_epochs + 1):
        model.train()
        tq = tqdm.tqdm(total=(steps_per_epoch)
        lr = get_learning_rate(optimizer)
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        losses = []
        tl = train_loader
        try:
            mean_loss = 0
            for i, (inputs, targets) in enumerate(tl):
                inputs = cuda(inputs)

                with torch.no_grad():
                    targets = cuda(targets)
                    
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                step += 1

                tq.update(inputs.size(0))
                losses.append(loss.item())
                mean_loss = np.mean(losses[-smooth_mean:])
                tq.set_postfix(loss='{:.5f}'.format(mean_loss))

            tq.close()

            valid_metrics = validation(model, criterion, valid_loader)            
            scores.append([
                "{:01d}".format(fold), 
                "{:03d}".format(epoch),
                "{:.4f}".format(valid_metrics['val_loss']), 
                "{:.4f}".format(valid_metrics['val_iou'])
            ])
            scores_df = pd.DataFrame(scores, columns=['fold', 'epoch', 'val_loss', 'val_iou'])
            scores_df.to_csv(str(scores_fname), index=False)            
                        
            if valid_best is None or valid_metrics['val_iou'] > valid_best:
                valid_best = valid_metrics['val_iou']
                save(model, optimizer, model_path, epoch, step, valid_best)

            scheduler.step(valid_metrics['val_loss'])
        except KeyboardInterrupt:
            tq.close()
            # print('Ctrl+C, saving snapshot')
            # save(model, optimizer, model_path, epoch, step, valid_best)
            print('done.')
            return
