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

    fold_ids = [train_id for train_id in train_ids if fold_dict[train_id] != fold]
    val_ids = [train_id for train_id in train_ids if fold_dict[train_id] == fold]
    return fold_ids, val_ids


def get_test_ids():
    sample_df = pd.read_csv(os.path.join(PATH, 'sample_submission.csv'))
    return sample_df.id.values


def train_transform(p=1):
    return Compose([
        PadIfNeeded(min_height=SIZE, min_width=SIZE),
        HorizontalFlip(p=0.5),
        ElasticTransform(p=0.1),
        ShiftScaleRotate(
            p=0.5,
            rotate_limit=0.0,
            shift_limit=.15,
            scale_limit=.15,
            interpolation=cv2.INTER_CUBIC,
            border_mode=cv2.BORDER_REFLECT_101),
    ], p=p)


def val_transform(p=1):
    return Compose([
        PadIfNeeded(min_height=SIZE, min_width=SIZE),
    ], p=p)


def make_test_loader(ids, transform=None, batch_size=32, workers=4):
    return make_loader(
        ids,
        transform=transform,
        mode='test',
        shuffle=False,
        batch_size=batch_size,
        workers=workers)    


def make_loader(ids, shuffle=False, transform=None, mode='train', batch_size=32, workers=4):
    assert transform is not None
    return DataLoader(
        dataset=TGSDataset(ids, transform=transform, mode=mode),
        shuffle=shuffle,
        num_workers=workers,
        batch_size=batch_size,
        pin_memory=torch.cuda.is_available()
    )


class TGSDataset(Dataset):
    def __init__(self, ids: list, transform=None, mode='train'):
        self.transform = transform
        self.mode = mode
        if mode != 'test':
            self.ids_ = pd.read_csv(os.path.join(PATH, 'train.csv')).id.values
            self.local_ids = ids
            self.real_idx = dict([(id_, pos) for pos, id_ in enumerate(self.ids_)])
            # self.file_names = [os.path.join(PATH, 'train', 'images', '%s.png' % id_) for id_ in self.ids_]
            # self.X_z = np.load('../data/cache/X_train_z.npy')
            # self.X_amount = np.log1p([np.sum(x) for x in np.load('../data/cache/X_train3_stage1oof.npy')])/10.0
            # self.X_filters = np.load('../data/cache/X_train_filters.npy')

        else:
            self.ids_ = pd.read_csv(os.path.join(PATH, 'sample_submission.csv')).id.values
            self.local_ids = ids
            self.real_idx = dict([(id_, pos) for pos, id_ in enumerate(self.ids_)])
            # self.file_names = [os.path.join(PATH, 'test', 'images', '%s.png' % self.id_) for id_ in ids]
            # self.X_z = np.load('../data/cache/X_test_z.npy')
            # self.X_amount = np.log1p([np.sum(x) for x in np.load('../data/cache/X_test3_stage1oof.npy')])/10.0
            # self.X_filters = np.load('../data/cache/X_test_filters.npy')

    def __len__(self):
        return len(self.local_ids)

    def load_image(self, path):
        img = cv2.imread(str(path))
        img = img.astype(np.float32) / 255.0
        img -= 0.5
        return img

    def get_image_fname(self, idx):
        id_ = self.local_ids[idx]
        subdir = 'test' if self.mode == 'test' else 'train'
        return os.path.join(PATH, subdir, 'images', '%s.png' % id_)

    def load_mask(self, path):
        mask_folder = 'masks'
        mask = cv2.imread(str(path).replace('images', mask_folder), 0)
        return (mask / 255.0).astype(np.uint8)

    def load_image_extra(self, idx):
        real_idx = self.real_idx[self.local_ids[idx]]
        image = self.load_image(self.get_image_fname(idx))
        return image

    def __getitem__(self, idx):
        fname = self.get_image_fname(idx)
        data = {'image': self.load_image_extra(idx)}

        if self.mode != 'test':
            data['mask'] = self.load_mask(fname)

        augmented = self.transform(**data)
        image_tensor = img_to_tensor(augmented['image']).reshape(3, SIZE, SIZE)

        if self.mode != 'test':
            mask = augmented['mask']
            # y_cls = np.array([1, np.sum(mask) > 0], dtype=np.uint8)
            targets = (
                # torch.from_numpy(mask).reshape(SIZE, SIZE).long(),
                torch.from_numpy(mask).reshape(1, SIZE, SIZE).float(),
                # torch.from_numpy(y_cls).float()
                None
            )
            # return img_to_tensor(image).view(3, 128, 128), torch.from_numpy(mask).view(1, 128, 128).float()
            return image_tensor, targets
        else:
            return image_tensor, str(fname)


if __name__ == '__main__':
    a, b = get_split(0)
    print(len(a), len(b))
    loader = make_loader(b)
    for inputs, target in loader:
        print(inputs.shape, target[0].shape)
        # npy = inputs.data.cpu().numpy()
        # for i in range(3):
        #    print(np.min (npy[0, ..., i]))
        #    print(np.max (npy[0, ..., i]))
        #    print(np.mean(npy[0, ..., i]))
        break