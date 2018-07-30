import json
from datetime import datetime
from pathlib import Path

import random
import numpy as np
import pandas as pd

import torch
import tqdm
from torch import nn


def cuda(x):
    return x.cuda(async=True) if torch.cuda.is_available() else x


def write_event(log, epoch, step: int, **data):
    data['epoch'] = epoch
    data['step'] = step
    data['dt'] = datetime.now().isoformat()    
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()


def save(model, optimizer, model_path, epoch, step):    
    # epoch_path = "{}_epoch{}.pth".format(str(model_path), epoch)
    torch.save({
        'model': model.state_dict(),
        'epoch': epoch,
        'optimizer': optimizer.state_dict(),
        'step': step,        
    }, str(model_path))


def train(experiment, output_dir, args, model, criterion, scheduler, train_loader, valid_loader, validation, optimizer, n_epochs=None, fold=None):
    # lr = args.lr
    n_epochs = n_epochs or args.n_epochs
    
    if args.resume:
        state = torch.load(args.resume)
        epoch = state['epoch']
        step = state['step']
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        print('Restored model, epoch {}, step {:,}'.format(epoch, step))
    else:
        epoch = 1
        step = 0
    
    model_path = output_dir / 'model_{fold}.pth'.format(fold=fold)
    
    scores_fname = output_dir / 'scores.csv'
    if scores_fname.exists():
        scores = pd.read_csv(scores_fname).values.tolist()
    else:
        scores = []
            
    smooth_mean = 10
    valid_best = None
    
    for epoch in range(epoch, n_epochs + 1):
        model.train()
        tq = tqdm.tqdm(total=(len(train_loader) * args.batch_size))
        tq.set_description('Epoch {}, lr {}'.format(epoch, None))
        losses = []
        tl = train_loader
        try:
            mean_loss = 0
            for i, (inputs, targets) in enumerate(tl):
                inputs = cuda(inputs)

                with torch.no_grad():
                    targets = [cuda(t) for t in targets]
                    
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                optimizer.zero_grad()
                batch_size = inputs.size(0)
                loss.backward()
                optimizer.step()
                step += 1
                tq.update(batch_size)
                losses.append(loss.item())
                mean_loss = np.mean(losses[-smooth_mean:])
                tq.set_postfix(loss='{:.5f}'.format(mean_loss))
                
            tq.close()

            valid_metrics = validation(model, criterion, valid_loader)            
            scores.append([
                "{}".format(fold), 
                "{:03d}".format(epoch),
                "{:.4f}".format(valid_metrics['val_loss']), 
                "{:.4f}".format(valid_metrics['val_iou'])
            ])
            scores_df = pd.DataFrame(scores, columns=['fold', 'epoch', 'val_loss', 'val_iou'])
            scores_df.to_csv(str(scores_fname), index=False)            
                        
            if valid_best is None or valid_metrics['val_iou'] > valid_best:
                valid_best = valid_metrics['val_iou']
                save(model, optimizer, model_path, epoch, step)

            scheduler.step(valid_metrics['val_loss'])
        except KeyboardInterrupt:
            tq.close()
            # print('Ctrl+C, saving snapshot')
            # save(model, optimizer, model_path, epoch, step)
            print('done.')
            return
