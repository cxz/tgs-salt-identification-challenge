import os
import torch
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import KFold

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from albumentations.torch.functional import img_to_tensor
from albumentations import HorizontalFlip, ShiftScaleRotate, Normalize, ElasticTransform, Compose, PadIfNeeded, RandomCrop

SIZE = 128
PATH = '../input'

def generate_folds_by_depth():
    n_fold = 5
    depths = pd.read_csv(os.path.join(PATH, 'depths.csv'))
    depths.sort_values('z', inplace=True)
    depths.drop('z', axis=1, inplace=True)
    depths['fold'] = (list(range(n_fold))*depths.shape[0])[:depths.shape[0]]
    print(depths.head())
    depths.to_csv(os.path.join(PATH, 'folds.csv'), index=False)

#def generate_folds_by_density():
#    def load_train_mask(image_id):
#        mask = cv2.imread(os.path.join('../input/train/masks', '%s.png' % image_id), 0)
#        return (mask / 255.0).astype(np.uint8)
#    
#    n_fold = 5
#    df = pd.read_csv(os.path.join('../input', 'train.csv'))
#    df['amounts'] = [np.sum(load_train_mask(x)) for x in df.id.values]    
#    df.sort_values('amounts', inplace=True)
#    df['fold'] = (list(range(n_fold))*df.shape[0])[:df.shape[0]]
#    print(df.head())
#    df[['id', 'fold']].to_csv(os.path.join(PATH, 'folds_density.csv'), index=False)    
    
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


def train_transform():
    return Compose([
        PadIfNeeded(min_height=SIZE, min_width=SIZE),
        HorizontalFlip(
            p=0.5),
        ElasticTransform(
            p=0.25,
            alpha=1,
            sigma=30,          # TODO
            alpha_affine=30),  # TODO
        ShiftScaleRotate(
            p=0.5,
            rotate_limit=.15,   # TODO
            shift_limit=.25,    # TODO
            scale_limit=.25,    # TODO
            interpolation=cv2.INTER_CUBIC,
            border_mode=cv2.BORDER_REFLECT_101),
    ], p=1)


def val_transform():
    return Compose([
        PadIfNeeded(min_height=SIZE, min_width=SIZE),
    ], p=1)


def make_loader(ids, shuffle=False, transform=None, mode='train', batch_size=32, workers=4, weighted_sampling=False, remove_suspicious=False):
    assert transform is not None
    
    sampler = None

    if mode == 'train' and remove_suspicious:
        suspicious = set(pd.read_csv('../data/cache/suspicious.csv').image_id.values)
        filtered_ids = [id_ for id_ in ids if id_ not in suspicious]
    else:
        filtered_ids = ids

    if mode == 'train' and weighted_sampling:
        def load_mask(image_id):
            path = os.path.join(PATH, 'train', 'masks', '%s.png' % image_id)
            mask = cv2.imread(path, 0)
            return (mask / 255.0).astype(np.uint8)

        masks = [load_mask(x) for x in filtered_ids]

        weights = [1.25 if 50 <= np.sum(m) <= 500 else 1 for m in masks]        
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(filtered_ids))
        shuffle = False  # mutualy exclusive

    return DataLoader(
        dataset=TGSDataset(filtered_ids, transform=transform, mode=mode),
        shuffle=shuffle,
        num_workers=workers,
        batch_size=batch_size,
        sampler=sampler,
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
            # self.extra = np.load('../data/cache/X_train_stage1oof_843.npy')
        else:
            self.ids_ = pd.read_csv(os.path.join(PATH, 'sample_submission.csv')).id.values
            self.local_ids = ids
            self.real_idx = dict([(id_, pos) for pos, id_ in enumerate(self.ids_)])

    def __len__(self):
        return len(self.local_ids)

    def load_image(self, path):
        img = cv2.imread(str(path))
        img = img.astype(np.float32) / 255.0
        img -= 0.5
        return img

    def get_image_fname(self, image_id):
        subdir = 'test' if self.mode == 'test' else 'train'
        return os.path.join(PATH, subdir, 'images', '%s.png' % image_id)

    def load_mask(self, image_id):
        path = os.path.join(PATH, 'train', 'masks', '%s.png' % image_id)
        mask = cv2.imread(path, 0)
        return (mask / 255.0).astype(np.uint8)

    def load_image_extra(self, image_id):
        return self.load_image(self.get_image_fname(image_id))

    def __getitem__(self, idx):
        image_id = self.local_ids[idx]
        data = {'image': self.load_image_extra(image_id)}
        
        if False:
            #x1 = self.load_image_extra(image_id)
            x2 = self.extra[self.real_idx[self.local_ids[idx]]]
            #x = np.zeros((101, 101, 1 + self.extra.shape[3]), dtype=np.float32)
            #x[..., 0] = x1[..., 0]
            #x[..., 1:] = x2
            # qdata = {'image': x2}

        if self.mode != 'test':
            data['mask'] = self.load_mask(image_id)

        augmented = self.transform(**data)
        image_tensor = img_to_tensor(augmented['image']).reshape(3, SIZE, SIZE)

        if self.mode != 'test':
            return image_tensor, torch.from_numpy(augmented['mask']).reshape(1, SIZE, SIZE).float()
        else:
            return image_tensor, self.get_image_fname(image_id)


if __name__ == '__main__':
    print('empty.')
    a, b = get_split(0)
    print(len(a), len(b))
    #loader = make_loader(b, transform=val_transform())
    #for inputs, target in loader:
    #    inputs = inputs.data.cpu().numpy()
    #    target = target.data.cpu().numpy()
    #    print(inputs.shape, np.max(inputs), target.shape, np.max(target))
            
