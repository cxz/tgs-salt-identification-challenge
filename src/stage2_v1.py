import os
import itertools

import torch
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import KFold

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler

from albumentations.torch.functional import img_to_tensor
from albumentations import HorizontalFlip, ShiftScaleRotate, Normalize, ElasticTransform, Compose, PadIfNeeded, \
    RandomCrop, Cutout, IAAAdditiveGaussianNoise
from albumentations import VerticalFlip

from validation import validation_binary
from loss import LossBinary, FocalLoss, LossLovasz, BCEDiceJaccardLoss

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.backends.cudnn as cudnn
import torch.backends.cudnn

from pathlib import Path
import utils


SIZE = 128
PAT = '../input'


class TGSDataset(Dataset):
    def __init__(self, ids: list, num_channels=3, transform=None, mode='train'):
        self.transform = transform
        self.mode = mode
        self.num_channels = num_channels
        if mode != 'test':
            self.ids_ = pd.read_csv(os.path.join('../input', 'train.csv')).id.values
            self.local_ids = ids
            self.real_idx = dict([(id_, pos) for pos, id_ in enumerate(self.ids_)])
            self.extra = np.load('../data/cache/X_train_stage1oof_851.npy')
        else:
            self.ids_ = pd.read_csv(os.path.join(PATH, 'sample_submission.csv')).id.values
            self.local_ids = ids
            self.real_idx = dict([(id_, pos) for pos, id_ in enumerate(self.ids_)])

    def __len__(self):
        return len(self.local_ids)

    def load_image(self, path):
        img_mean = 0.5  # 0.47194558584317564
        img_std = 1  # 0.1088611608890351
        img = cv2.imread(str(path))[:, :, :self.num_channels]
        img = img.astype(np.float32) / 255
        # TODO: need to adjust to new normalization code.
        raise
        img -= img_mean
        img /= img_std
        return img

    def get_image_fname(self, image_id):
        subdir = 'test' if self.mode == 'test' else 'train'
        return os.path.join('../input', subdir, 'images', '%s.png' % image_id)

    def load_mask(self, image_id):
        path = os.path.join('../input', 'train', 'masks', '%s.png' % image_id)
        mask = cv2.imread(path, 0)
        return (mask / 255.0).astype(np.uint8)

    def load_image_extra(self, image_id):
        return self.load_image(self.get_image_fname(image_id))

    def __getitem__(self, idx):
        image_id = self.local_ids[idx]
        # data = {'image': self.load_image_extra(image_id)}
        data = {}

        if True:
            x1 = self.load_image_extra(image_id)
            x2 = self.extra[self.real_idx[self.local_ids[idx]]]
            # print(x1.shape, x2.shape)
            x = np.zeros((101, 101, self.num_channels + self.extra.shape[3]), dtype=np.float32)
            x[..., :self.num_channels] = x1[..., :self.num_channels]
            x[..., self.num_channels:] = x2
            data = {'image': x}

        if self.mode != 'test':
            data['mask'] = self.load_mask(image_id)

        augmented = self.transform(**data)
        image_tensor = img_to_tensor(augmented['image']).reshape(self.num_channels + self.extra.shape[3], SIZE, SIZE)

        if self.mode != 'test':
            return image_tensor, torch.from_numpy(augmented['mask']).reshape(1, SIZE, SIZE).float()
        else:
            return image_tensor, self.get_image_fname(image_id)

import dataset


import torch
import torch.nn as nn

from torchvision.models import resnet50


def conv3x3(num_in, num_out):
    '''Creates a 3x3 convolution building block module.

    Args:
      num_in: number of input feature maps
      num_out: number of output feature maps

    Returns:
      The 3x3 convolution module.
    '''

    return nn.Conv2d(num_in, num_out, kernel_size=3, padding=1, bias=False)


class ConvRelu(nn.Module):
    '''Convolution followed by ReLU activation building block.
    '''

    def __init__(self, num_in, num_out):
        '''Creates a `ConvReLU` building block.

        Args:
          num_in: number of input feature maps
          num_out: number of output feature maps
        '''

        super().__init__()

        self.block = nn.Sequential(
            conv3x3(num_in, num_out),
            nn.ReLU(inplace=True))

    def forward(self, x):
        '''The networks forward pass for which autograd synthesizes the backwards pass.

        Args:
          x: the input tensor

        Returns:
          The networks output tensor.
        '''

        return self.block(x)


class DecoderBlock(nn.Module):
    '''Decoder building block upsampling resolution by a factor of two.
    '''

    def __init__(self, num_in, num_out):
        '''Creates a `DecoderBlock` building block.

        Args:
          num_in: number of input feature maps
          num_out: number of output feature maps
        '''

        super().__init__()

        self.block = ConvRelu(num_in, num_out)

    def forward(self, x):
        '''The networks forward pass for which autograd synthesizes the backwards pass.

        Args:
          x: the input tensor

        Returns:
          The networks output tensor.
        '''

        return self.block(nn.functional.upsample(x, scale_factor=2, mode='nearest'))


class UNet(nn.Module):
    def __init__(self, num_classes=6, num_filters=32, pretrained=True):
        super().__init__()

        self.resnet = resnet50(pretrained=pretrained)

        self.enc0 = nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu, self.resnet.maxpool)
        self.enc1 = self.resnet.layer1  # 256
        self.enc2 = self.resnet.layer2  # 512
        self.enc3 = self.resnet.layer3  # 1024
        self.enc4 = self.resnet.layer4  # 2048

        self.center = DecoderBlock(2048, num_filters * 8)

        self.dec0 = DecoderBlock(2048 + num_filters * 8, num_filters * 8)
        self.dec1 = DecoderBlock(1024 + num_filters * 8, num_filters * 8)
        self.dec2 = DecoderBlock(512 + num_filters * 8, num_filters * 2)
        self.dec3 = DecoderBlock(256 + num_filters * 2, num_filters * 2 * 2)
        self.dec4 = DecoderBlock(num_filters * 2 * 2, num_filters)
        self.dec5 = ConvRelu(num_filters, num_filters)

        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)
        self.hidden1 = nn.Linear(6, 1)

    def forward(self, x):
        enc0 = self.enc0(x[:, :3, :, :])
        enc1 = self.enc1(enc0)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        center = self.center(nn.functional.max_pool2d(enc4, kernel_size=2, stride=2))

        dec0 = self.dec0(torch.cat([enc4, center], dim=1))
        dec1 = self.dec1(torch.cat([enc3, dec0], dim=1))
        dec2 = self.dec2(torch.cat([enc2, dec1], dim=1))
        dec3 = self.dec3(torch.cat([enc1, dec2], dim=1))
        dec4 = self.dec4(dec3)
        dec5 = self.dec5(dec4)

        out = self.final(dec5)  # [n, num_classes, 128, 128]
        out = torch.sum(out * x[:, 3:, :, :], dim=1)  # [n, 128, 128]
        return out

import models

if __name__ == '__main__':
    fold = 1
    model = models.get_model('tmp/model_{}.pth'.format(fold), 'unet-dpn107')

    # model.forward(torch.from_numpy(np.zeros((1, 9, 128, 128), dtype=np.float32)))
    device_ids = None
    model = nn.DataParallel(model, device_ids=device_ids).cuda()
    cudnn.benchmark = True

    optimizer = Adam([p for p in model.parameters() if p.requires_grad], lr=1e-6)
    loss = loss = LossLovasz()
    validation = validation_binary
    scheduler = ReduceLROnPlateau(optimizer, verbose=True, min_lr=1e-7, factor=0.5, patience=5)

    output_dir = Path('tmp')
    output_dir.mkdir(exist_ok=True, parents=True)

    train_ids, val_ids = dataset.get_split(fold)
    train_loader = DataLoader(
        dataset=TGSDataset(train_ids, num_channels=3, transform=dataset.val_transform(), mode='train'),
        shuffle=True,
        num_workers=4,
        batch_size=24,
        sampler=None,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        dataset=TGSDataset(val_ids, num_channels=3, transform=dataset.val_transform(), mode='train'),
        shuffle=False,
        num_workers=4,
        batch_size=1,
        sampler=None,
        pin_memory=torch.cuda.is_available()
    )

    utils.train(
        experiment=None,
        output_dir=output_dir,
        optimizer=optimizer,
        args=None,
        model=model,
        criterion=loss,
        scheduler=scheduler,
        train_loader=train_loader,
        valid_loader=val_loader,
        validation=validation,
        fold=fold,
        batch_size=24,
        n_epochs=200,
        snapshot=None
    )