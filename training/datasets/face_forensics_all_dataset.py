import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from constants import DF, F2F, FSW, NT
from constants import FACE_FORENSICS
from training.datasets.face_forensics_dataset import FaceForensicsDataset
from training.datasets.transform_v2 import create_train_transform, create_val_test_transform

MANIPULATION_TYPES = [DF, F2F, FSW, NT]


def get_face_forensics_all_dataloader(model_cfg, args):
    version = args.compression_version
    train_transform = create_train_transform(model_cfg)
    train_df_list = []
    for fake_type in MANIPULATION_TYPES:
        train_df = pd.read_csv(f'data/{FACE_FORENSICS}/{version}/data_{FACE_FORENSICS}_{fake_type}_train.csv')
        train_df['fake_type'] = fake_type
        train_df_list.append(train_df)
    train_df_concat = pd.concat(train_df_list)
    train_data = FaceForensicsDataset(data_root=args.data_dir, df=train_df_concat, mode='train',
                                      transform=train_transform, fake_type='All',
                                      compression_version=version, use_generalization=True)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=(train_sampler is None),
                              sampler=train_sampler, num_workers=args.workers, pin_memory=True, drop_last=True)

    val_transform = create_val_test_transform(model_cfg)
    val_df_list = []
    for fake_type in MANIPULATION_TYPES:
        val_df = pd.read_csv(f'data/{FACE_FORENSICS}/{version}/data_{FACE_FORENSICS}_{fake_type}_val.csv')
        val_df['fake_type'] = fake_type
        val_df_list.append(val_df)
    val_df_concat = pd.concat(val_df_list)
    val_data = FaceForensicsDataset(data_root=args.data_dir, df=val_df_concat, mode='validation',
                                    transform=val_transform, fake_type='All', compression_version=version)
    val_loader = DataLoader(val_data, batch_size=30, shuffle=False, num_workers=args.workers,
                            pin_memory=True, drop_last=False)

    return train_sampler, train_loader, val_loader


def get_face_forensics_all_test_dataloader(model_cfg, args):
    version = args.compression_version
    test_transform = create_val_test_transform(model_cfg)
    test_df_list = []
    for fake_type in MANIPULATION_TYPES:
        test_df = pd.read_csv(f'data/{FACE_FORENSICS}/{version}/data_{FACE_FORENSICS}_{fake_type}_test.csv')
        test_df['fake_type'] = fake_type
        test_df_list.append(test_df)
    test_df_concat = pd.concat(test_df_list)
    # test_df = test_df.iloc[11281:]
    test_data = FaceForensicsDataset(data_root=args.data_dir, df=test_df_concat, mode='test', transform=test_transform,
                                     fake_type='All', compression_version=version)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                             pin_memory=True, drop_last=True)
    return test_loader
