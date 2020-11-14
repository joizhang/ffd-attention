import os

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from torch.utils.data import Dataset, DataLoader

from preprocessing.constants import CELEB_DF
from training.datasets.transform import create_transform


def create_generalization_transform():
    return A.Compose([
        A.Blur(blur_limit=(5, 10), p=0.7),
        A.OneOf([A.RandomBrightnessContrast(), A.FancyPCA(), A.HueSaturationValue()], p=0.7),
        A.OpticalDistortion(distort_limit=(1., 2.), border_mode=cv2.BORDER_CONSTANT, p=0.5)
    ])


class CelebDFV2Dataset(Dataset):

    def __init__(self, data_root, df: DataFrame, mode, transform: A.Compose):
        self.data_root = data_root
        self.df = df
        self.mode = mode
        self.transform = transform
        self.generalization_transform = create_generalization_transform()

    def __getitem__(self, index):
        video, img_file, label, ori_video, frame = self.df.iloc[index].values
        # image
        img_path = os.path.join(self.data_root, 'crops', video, img_file)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # mask
        if label == 1:
            mask_path = os.path.join(self.data_root, 'diffs', video, '{}_diff.png'.format(img_file[:-4]))
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        # data augmentation
        transformed = self.transform(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"]
        mask = mask.unsqueeze(0) / 255.

        return {'images': image, 'labels': label, 'masks': mask}

    def __len__(self):
        r = self.df.shape[0]
        return r


def get_celeba_df_dataloader(model, args):
    train_df = pd.read_csv(f'data/{CELEB_DF}/data_{CELEB_DF}_train.csv')
    train_transform = create_transform(model.default_cfg)
    train_data = CelebDFV2Dataset(data_root=args.data_dir, df=train_df, mode='train', transform=train_transform)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=(train_sampler is None),
                              sampler=train_sampler, num_workers=args.workers, pin_memory=True, drop_last=True)

    val_df = pd.read_csv(f'data/{CELEB_DF}/data_{CELEB_DF}_val.csv')
    val_transform = create_val_test_transform(model.default_cfg)
    val_data = CelebDFV2Dataset(data_root=args.data_dir, df=val_df, mode='validation', transform=val_transform)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                            pin_memory=True, drop_last=False)

    return train_sampler, train_loader, val_loader


def get_celeba_df_test_dataloader(model, args):
    assert os.path.exists(f'data/{CELEB_DF}/data_{CELEB_DF}_test.csv'), 'Please firstly prepare data_test.csv.'
    df = pd.read_csv(f'data/{CELEB_DF}/data_{CELEB_DF}_test.csv')
    test_transform = create_val_test_transform(model.default_cfg)
    test_data = CelebDFV2Dataset(args.data_dir, df=df, mode='test', transform=test_transform)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                             pin_memory=True, drop_last=False)
    return test_loader
