import os
import torch
import numpy as np
import pandas as pd
import cv2
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from albumentations.torch.functional import img_to_tensor
from albumentations import HorizontalFlip, ShiftScaleRotate, Normalize, ElasticTransform, Compose, PadIfNeeded, RandomCrop

SIZE = 128
PATH = '../input'

def get_split(fold):
    train_df = pd.read_csv(os.path.join(PATH, 'train.csv'))
    train_ids = train_df.id.values

    folds = pd.read_csv(os.path.join(PATH, 'folds.csv'))
    fold_dict = folds.set_index('id').to_dict()['fold']

    train_file_names = [os.path.join(PATH, 'train', 'images', '%s.png' % train_id)
                        for train_id in train_ids
                        if fold_dict[train_id] != fold]

    val_file_names = [os.path.join(PATH, 'train', 'images', '%s.png' % train_id)
                      for train_id in train_ids
                      if fold_dict[train_id] == fold]

    return train_file_names, val_file_names

def get_test_filenames():
    sample_df = pd.read_csv(os.path.join(PATH, 'sample_submission.csv'))
    test_ids = sample_df.id.values
    return [os.path.join(PATH, 'test', 'images', '%s.png' % test_id) for test_id in test_ids]


def train_transform(p=1):
    return Compose([
        PadIfNeeded(min_height=SIZE, min_width=SIZE),
        #PadIfNeeded(min_height=SIZE*2, min_width=SIZE*2),
        #RandomCrop(SIZE, SIZE),
        HorizontalFlip(p=0.5),
        ElasticTransform(p=0.1),
        ShiftScaleRotate(
            p=0.5,
            shift_limit=.15,
            scale_limit=.15,
            interpolation=cv2.INTER_CUBIC,
            border_mode=cv2.BORDER_REFLECT_101),
        # Normalize(p=1)
    ], p=p)


def val_transform(before=[], p=1):
    return Compose(before + [
        PadIfNeeded(min_height=SIZE, min_width=SIZE),
        # Normalize(p=1)
    ], p=p)


def get_fold(fold):
    return get_split(fold)


def make_train_loader(filenames, batch_size=32, workers=4):
    train_loader = make_loader(
        filenames,
        shuffle=True,
        transform=train_transform(p=1),
        batch_size=batch_size,
        workers=workers)
    return train_loader


def make_val_loader(filenames, transform=val_transform(p=1), batch_size=32, workers=4):
    val_loader = make_loader(
        filenames,
        transform=transform,
        batch_size=batch_size,
        workers=workers)
    return val_loader

def make_test_loader(filenames, transform=None, batch_size=32, workers=4):
    return make_loader(
        filenames,
        transform=transform,
        mode='test',
        shuffle=False,
        batch_size=batch_size,
        workers=workers)    

def make_loader(file_names, shuffle=False, transform=None, mode='train', batch_size=1, workers=2):
    assert transform is not None
    return DataLoader(
        dataset=TGSDataset(file_names, transform=transform, mode=mode),
        shuffle=shuffle,
        num_workers=workers,
        batch_size=batch_size,
        pin_memory=torch.cuda.is_available()
    )

import skimage

class TGSDataset(Dataset):
    def __init__(self, file_names: list, transform=None, mode='train'):
        self.file_names = file_names
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_file_name = self.file_names[idx]
        data = {'image': load_image(img_file_name)}

        if self.mode != 'test':
            data['mask'] = load_mask(img_file_name)

        augmented = self.transform(**data)
        image_tensor = img_to_tensor(augmented['image']).reshape(3, SIZE, SIZE)

        if self.mode != 'test':
            mask = augmented['mask']
            y_cls = np.array([1, np.sum(mask) > 0], dtype=np.uint8)
            targets = (
                # torch.from_numpy(mask).reshape(SIZE, SIZE).long(),
                torch.from_numpy(mask).reshape(1, SIZE, SIZE).float(),
                torch.from_numpy(y_cls).float()
            )
            # return img_to_tensor(image).view(3, 128, 128), torch.from_numpy(mask).view(1, 128, 128).float()
            return image_tensor, targets
        else:
            return image_tensor, str(img_file_name)


def load_image(path):
    if not os.path.exists(path):
        raise(path)
    img = cv2.imread(str(path))
    if img is None:
        raise
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC).reshape(128, 128, 3)
    img= img.astype(np.float32) / 255.0
    img -= 0.5
    return img


def load_mask(path):
    if not os.path.exists(path):
        raise(path)
    mask_folder = 'masks'    
    mask = cv2.imread(str(path).replace('images', mask_folder), 0)
    #mask = cv2.resize(mask, (128, 128), interpolation=cv2.INTER_CUBIC)
    #mask = mask.astype(np.float32) / 255.0
    return (mask/255.0).astype(np.uint8)

if __name__ == '__main__':
    a, b = get_fold(0)
    loader = make_val_loader(b)
    for inputs, target in loader:
        print('.')
        break