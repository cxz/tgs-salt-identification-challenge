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
from albumentations import HorizontalFlip, ShiftScaleRotate, Normalize, ElasticTransform, Compose, PadIfNeeded, RandomCrop, Cutout, OneOf, IAAAdditiveGaussianNoise, GaussNoise, RandomContrast
from albumentations import VerticalFlip
from albumentations import Resize
from albumentations import Normalize

SIZE = 256
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

    # depth = pd.read_csv(os.path.join(PATH, 'depths.csv'))
    # depth_dict = depth.set_index('id').to_dict()['z']

    fold_ids = [train_id for train_id in train_ids if fold_dict[train_id] != fold]
    val_ids = [train_id for train_id in train_ids if fold_dict[train_id] == fold]
    return fold_ids, val_ids


def get_test_ids():
    sample_df = pd.read_csv(os.path.join(PATH, 'sample_submission.csv'))
    return sample_df.id.values


def train_transform(upside_down=False):
    return Compose([
        Resize(202, 202, 
               interpolation=cv2.INTER_NEAREST),
        PadIfNeeded(min_height=SIZE, 
                    min_width=SIZE, 
                    border_mode=cv2.BORDER_REPLICATE),
        VerticalFlip(p=int(upside_down)),
        HorizontalFlip(
            p=0.5),
        Cutout(
            p=0.1,
            num_holes=1,
            max_h_size=10,
            max_w_size=10
        ),
        #OneOf([
        #    IAAAdditiveGaussianNoise(),
        #    #GaussNoise(),
        #    #RandomContrast(),
        #], p=0.5),
        ElasticTransform(
            p=0.25,
            alpha=1,
            sigma=30,          # TODO
            alpha_affine=30),  # TODO
        ShiftScaleRotate(
            p=0.25,
            rotate_limit=.15,   # TODO
            shift_limit=.15,    # TODO
            scale_limit=.15,    # TODO
            interpolation=cv2.INTER_CUBIC,
            #border_mode=cv2.BORDER_REFLECT_101),
            border_mode=cv2.BORDER_REPLICATE),
        Normalize(),
    ], p=1)


def val_transform(upside_down=False):    
    return Compose([
        Resize(202, 202, 
               interpolation=cv2.INTER_NEAREST),
        VerticalFlip(p=int(upside_down)),
        PadIfNeeded(min_height=SIZE, 
                    min_width=SIZE,
                    border_mode=cv2.BORDER_REPLICATE),
        Normalize()
    ], p=1)


def make_loader(
        ids,
        shuffle=False,
        num_channels=3,
        transform=None, mode='train',
        batch_size=32, workers=4,
        ignore_empty_masks=False,
        weighted_sampling=False,
        weighted_sampling_small_masks=False,
        remove_suspicious=False):

    assert transform is not None
    
    sampler = None

    if mode == 'train' and remove_suspicious:
        suspicious = set(pd.read_csv('../data/cache/suspicious.csv').image_id.values)
        filtered_ids = [id_ for id_ in ids if id_ not in suspicious]
    else:
        filtered_ids = ids

    if mode == 'train' and ignore_empty_masks:
        def load_mask(image_id):
            path = os.path.join(PATH, 'train', 'masks', '%s.png' % image_id)
            mask = cv2.imread(path, 0)
            return (mask / 255.0).astype(np.uint8)

        masks = [load_mask(x) for x in filtered_ids]
        weights = [0 if np.sum(m) <= 5 else 1 for m in masks]
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(filtered_ids))
        shuffle = False  # mutualy exclusive

    if mode == 'train' and weighted_sampling_small_masks:
        def load_mask(image_id):
            path = os.path.join(PATH, 'train', 'masks', '%s.png' % image_id)
            mask = cv2.imread(path, 0)
            return (mask / 255.0).astype(np.uint8)

        masks = [load_mask(x) for x in filtered_ids]

        weights = [1.25 if 50 <= np.sum(m) <= 500 else 1 for m in masks]        
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(filtered_ids))
        shuffle = False  # mutualy exclusive
        
    if mode == 'train' and weighted_sampling:
        def load_mask(image_id):
            path = os.path.join(PATH, 'train', 'masks', '%s.png' % image_id)
            mask = cv2.imread(path, 0)
            return (mask / 255.0).astype(np.uint8)

        masks = [load_mask(x) for x in filtered_ids]

        weights = [1 if 0 == np.sum(m) else 1.6315 for m in masks] # 38% of empty masks
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(filtered_ids))
        shuffle = False  # mutualy exclusive        

    return DataLoader(
        dataset=TGSDataset(filtered_ids, num_channels=num_channels, transform=transform, mode=mode),
        shuffle=shuffle,
        num_workers=workers,
        batch_size=batch_size,
        sampler=sampler,
        pin_memory=torch.cuda.is_available()
    )


class TGSDataset(Dataset):
    def __init__(self, ids: list, num_channels=3, transform=None, mode='train'):
        self.transform = transform
        self.mode = mode
        self.num_channels = num_channels
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
        img_mean = 0.5  # 0.47194558584317564
        img_std = 1  # 0.1088611608890351
        img = cv2.imread(str(path))[:, :, :self.num_channels]
        img = img.astype(np.float32) #/ 255
        #img -= mean
        #img /= std
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
        image_tensor = img_to_tensor(augmented['image']).reshape(self.num_channels, SIZE, SIZE)

        if self.mode != 'test':
            return image_tensor, torch.from_numpy(augmented['mask']).reshape(1, SIZE, SIZE).float()
        else:
            return image_tensor, self.get_image_fname(image_id)


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices
    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        for (primary_batch, secondary_batch) in  zip(grouper(primary_iter, self.primary_batch_size), grouper(secondary_iter, self.secondary_batch_size)):
            yield primary_batch + secondary_batch

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


class MeanTeacherTGSDataset(Dataset):
    def __init__(self, labeled_ids: list, transform=None, mode='train'):
        self.transform = transform
        self.mode = mode

        self.ids_ = pd.read_csv(os.path.join(PATH, 'train.csv')).id.values
        self.local_ids = labeled_ids
        self.real_idx = dict([(id_, pos) for pos, id_ in enumerate(self.ids_)])

        self.unlabeled_ids_ = pd.read_csv(os.path.join(PATH, 'sample_submission.csv')).id.values

    def __len__(self):
        return len(self.local_ids) + len(self.unlabeled_ids_)

    def load_image(self, path):
        img = cv2.imread(str(path))
        img = img.astype(np.float32) # / 255.0
        #img -= 0.5
        return img

    def get_image_fname(self, image_id, labeled):
        subdir = 'train' if labeled else 'test'
        return os.path.join(PATH, subdir, 'images', '%s.png' % image_id)

    def load_mask(self, image_id):
        path = os.path.join(PATH, 'train', 'masks', '%s.png' % image_id)
        mask = cv2.imread(path, 0)
        return (mask / 255.0).astype(np.uint8)

    def get_image_id(self, idx):
        is_labeled = idx < len(self.local_ids)
        if is_labeled:
            return self.local_ids[idx]
        else:
            return self.unlabeled_ids_[idx - len(self.local_ids)]

    def __getitem__(self, idx):
        is_labeled = idx < len(self.local_ids)
        image_id = self.get_image_id(idx)
        image = self.load_image(self.get_image_fname(image_id, is_labeled))
        data = dict()
        data['image'] = image
        if is_labeled:
            data['mask'] = self.load_mask(image_id)
        else:
            data['mask'] = np.zeros((SIZE, SIZE), dtype=np.uint8)

        augmented = self.transform(**data)
        image_tensor = img_to_tensor(augmented['image']).reshape(3, SIZE, SIZE)

        # TODO: add noise to teacher input
        teacher_tensor = img_to_tensor(augmented['image']).reshape(3, SIZE, SIZE)
        return (image_tensor, teacher_tensor, int(is_labeled)), torch.from_numpy(augmented['mask']).reshape(1, SIZE, SIZE).float()


if __name__ == '__main__':
    if False:
        print('empty.')
        a, b = get_split(0)
        print('train, val: ', len(a), len(b))

        labeled_size = len(a)
        unlabeled_size = 18000
        batch_size = 24
        sampler = TwoStreamBatchSampler(
            list(range(labeled_size)),  # labeled ids
            list(range(labeled_size, labeled_size+unlabeled_size)),  # unlabeled ids
            batch_size,  # total batch size (labeled + unlabeled)
            batch_size // 2  # labeled batch size
        )

        print('len sampler: ', len(sampler))

        dl = DataLoader(
            dataset=MeanTeacherTGSDataset(a, transform=val_transform(), mode='train'),
            batch_sampler=sampler,
            num_workers=1,
            pin_memory=torch.cuda.is_available()
        )

        for batch_idx, (inputs, target) in enumerate(dl):
            print(batch_idx)
            #print(type(inputs))
            #print(inputs[0].shape, inputs[1].shape, inputs[2].shape)
            #print(target.shape)
            break
    fold_ids, val_ids = get_split(0)
    loader = make_loader(fold_ids, transform=train_transform())
    for x, y in loader:
        x_ = x.data.cpu().numpy()
        print(np.max(x_), np.mean(x_))
        break




