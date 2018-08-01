"""
"""

import argparse
import json
import uuid
from pathlib import Path
from validation import validation_binary

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.backends.cudnn as cudnn
import torch.backends.cudnn

from loss import FocalLoss
import dataset
import models
import utils
import random


def get_model(model_path, model):
    return None


def make_loader(ids, transform, shuffle=False, batch_size=32, workers=4):
    return None


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--name', type=str)
    arg('--jaccard-weight', default=0.25, type=float)
    arg('--device-ids', type=str, default='0', help='For example 0,1 to run on two GPUs')
    arg('--fold', type=int, help='fold', default=0)
    arg('--output-dir', default='../data/runs', help='checkpoint root')
    arg('--batch-size', type=int, default=32)
    arg('--n-epochs', type=int, default=100)
    arg('--lr', type=float, default=0.0001)
    arg('--workers', type=int, default=4)
    arg('--seed', type=int, default=0)
    arg('--resume', type=str)
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
    model = get_model()

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
        transform=dataset.train_transform(),
        shuffle=True,
        batch_size=args.batch_size,
        workers=args.workers)

    valid_loader = dataset.make_loader(
        val_ids,
        transform=dataset.val_transform(),
        shuffle=False,
        batch_size=args.batch_size,
        workers=args.workers)

    optimizer = Adam(model.parameters(), lr=args.lr)
    loss = FocalLoss(args.focal_gamma)
    validation = validation_binary
    scheduler = ReduceLROnPlateau(optimizer, verbose=True, min_lr=1e-6, factor=0.5)

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
        fold=args.fold
    )


if __name__ == '__main__':
    main()