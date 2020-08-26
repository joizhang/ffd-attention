import os
from glob import glob
from pathlib import Path
from pandas import DataFrame

import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

"""
Real face images: FFHQ, CelebA
Identity and expression swap: FaceForensics++(FaceSwap and Deepfake)
Attributes manipulation: FaceAPP, StarGAN
Entire face synthesis: PGGAN, StyleGAN
"""
CLASSES = {'Real': 0, 'Fake_Partial': 1, 'Fake_Entire': 2}

DFFD = {
    'ffhq': 'Real',
    # 'celeba': 'Real',
    'faceapp': 'Fake_Partial',
    # 'stargan': 'Fake_Partial',
    # 'pggan_v1': 'Fake_Entire',
    # 'pggan_v2': 'Fake_Entire',
    'stylegan_celeba': 'Fake_Entire',
    'stylegan_ffhq': 'Fake_Entire'
}


class DffdDataset(Dataset):

    def __init__(self, data_root, mode, transform=None, mask_transform=None):
        self.data_root = data_root
        self.mode = mode
        self.transform = transform
        self.mask_transform = mask_transform
        self.data = []
        self._prepare_data()

    def _prepare_data(self):
        mode_paths = []
        for mode_path in glob(os.path.join(self.data_root, "*/{}".format(self.mode))):
            data_type = Path(mode_path).parent.name
            if data_type in DFFD and DFFD[data_type] in CLASSES:
                mode_paths.append(mode_path)
                label = CLASSES[DFFD[Path(mode_path).parent.name]]
                for file_name in os.listdir(mode_path):
                    image_path = os.path.join(mode_path, file_name)
                    mask_path = os.path.join("{}_mask".format(mode_path), file_name)
                    self.data.append((image_path, label, mask_path))
        print(self.mode, ":", mode_paths)

    def __getitem__(self, index):
        image_path, label, mask_path = self.data[index]
        # image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        # mask
        mask = None
        if label == 0:
            mask = np.zeros((image.size(1), image.size(2)), dtype=np.uint8)
        elif label == 1:
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            else:
                mask = np.zeros(image.numpy().transpose(1, 2, 0).shape)
        elif label == 2:
            mask = np.ones((image.size(1), image.size(2)), dtype=np.uint8) * 255
        mask = self.mask_transform(mask)
        return {'images': image, 'labels': label, 'masks': mask}

    def __len__(self):
        return len(self.data)


class CelebDFV2Dataset(Dataset):

    def __init__(self, data_root, df: DataFrame, mode, transform, ):
        self.data_root = data_root
        self.df = df
        self.mode = mode
        self.transform = transform

    def __getitem__(self, index):
        video, img_file, label, ori_video, frame = self.df.iloc[index].values
        img_path = os.path.join(self.data_root, 'crops', video, img_file)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        return image, label

    def __len__(self):
        r = self.df.shape[0]
        return r
