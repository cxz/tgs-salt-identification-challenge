import os
import argparse
import json
import uuid
from pathlib import Path
from validation import validation_binary

import torch
from torch import nn
from torch.optim import Adam
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.backends.cudnn as cudnn
import torch.backends.cudnn

from loss import LossBinary, FocalLoss, LossLovasz, BCEDiceJaccardLoss, LossHinge

import dataset
import models
import utils
import random


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--name', type=str)
    arg('--jaccard-weight', default=0.25, type=float)
    arg('--device-ids', type=str, default='0', help='For example 0,1 to run on two GPUs')
    arg('--fold', type=int, help='fold', default=0)
    arg('--output-dir', default='../data/runs', help='checkpoint root')
    arg('--batch-size', type=int, default=32)
    arg('--iter-size', type=int, default=1)
    arg('--n-epochs', type=int, default=100)
    arg('--lr', type=float, default=0.0001)
    arg('--workers', type=int, default=4)
    arg('--seed', type=int, default=0)
    arg('--model', type=str, default=models.archs[0], choices=models.archs)
    arg('--loss', type=str, default='focal', choices=['focal', 'lovasz', 'bjd', 'bce_jaccard', 'bce_dice', 'cos_dice', 'hinge'])
    arg('--focal-gamma', type=float, default=.5)
    arg('--num-channels', type=int, default=3)
    arg('--weighted-sampler', action="store_true")
    arg('--ignore-empty-masks', action='store_true')
    arg('--remove-suspicious', action='store_true')
    arg('--resume', action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if not args.name:
        experiment = uuid.uuid4().hex        
    else:
        experiment = args.name 
        
    output_dir = Path(args.output_dir) / experiment
    output_dir.mkdir(exist_ok=True, parents=True)
    output_dir.joinpath('params.json').write_text(json.dumps(vars(args), indent=True, sort_keys=True))
    
    # in case --resume is provided it will be loaded later
    model = models.get_model('../data/runs/exp77/model_4.pth', args.model)

    if torch.cuda.is_available():
        if args.device_ids:
            device_ids = list(map(int, args.device_ids.split(',')))
        else:
            device_ids = None
        model = nn.DataParallel(model, device_ids=device_ids).cuda()

    train_ids, val_ids = dataset.get_split(args.fold)

    cudnn.benchmark = True

    train_loader = dataset.make_loader(
        train_ids,
        num_channels=args.num_channels,
        transform=dataset.train_transform(),
        shuffle=True,
        weighted_sampling=args.weighted_sampler,
        ignore_empty_masks=args.ignore_empty_masks,
        remove_suspicious=args.remove_suspicious,
        batch_size=args.batch_size,
        workers=args.workers)

    valid_loader = dataset.make_loader(
        val_ids,
        num_channels=args.num_channels,
        transform=dataset.val_transform(),
        shuffle=False,
        #batch_size=len(device_ids),
        batch_size=args.batch_size, # len(device_ids),
        workers=args.workers)

    # optimizer = Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    optimizer = Adam(model.parameters(), lr=args.lr)

    # loss = LossBinary(jaccard_weight=args.jaccard_weight)
    # loss = LossBinaryMixedDiceBCE(dice_weight=0.5, bce_weight=0.5)
    if args.loss == 'focal':
        loss = FocalLoss(args.focal_gamma)
    elif args.loss == 'lovasz':
        loss = LossLovasz()
    elif args.loss == 'bjd':               
        loss = BCEDiceJaccardLoss({'bce': 0.25, 'jaccard': None, 'dice': 0.75})
    elif args.loss == 'bce_jaccard':
        loss = LossBinary(args.jaccard_weight)
    elif args.loss == 'bce_dice':
        import loss2
        bce_weight = 1
        dice_weight = 2
        loss = loss2.make_loss(bce_weight, dice_weight)
    elif args.loss == 'cos_dice':
        import loss2
        loss = loss2.make_cos_dice_loss()
    elif args.loss == 'hinge':
        loss = LossHinge()
        
    else:
        raise NotImplementedError

    validation = validation_binary
    scheduler = ReduceLROnPlateau(optimizer, verbose=True, min_lr=1e-7, factor=0.5)
    snapshot = utils.fold_snapshot(output_dir, args.fold) if args.resume else None

    utils.train(
        experiment=experiment,
        output_dir=output_dir,
        optimizer=optimizer,
        args=args,
        model=model,
        criterion=loss,
        scheduler=scheduler,
        train_loader=train_loader,
        valid_loader=valid_loader,
        validation=validation,
        fold=args.fold,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,        
        snapshot=snapshot,
        iter_size=args.iter_size
    )


if __name__ == '__main__':
    main()
