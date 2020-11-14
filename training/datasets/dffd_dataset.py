import os
from glob import glob
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from training.datasets.transform import create_transform

"""
1. Real face images: FFHQ, CelebA
2. Identity and expression swap: FaceForensics++(FaceSwap and Deepfake)
3. Attributes manipulation: FaceAPP, StarGAN
4. Entire face synthesis: PGGAN, StyleGAN
"""
CLASSES = {
    'Real': 0,
    'Expression_Swap': 1,
    'Identity_Swap': 2,
    'Attribute_Manipulation': 3,
    'Entire_Face_Synthesis': 4
}

DFFD = {
    'ffhq': 'Real',
    # 'celeba': 'Real',

    'faceapp': 'Attribute_Manipulation',
    # 'stargan': 'Attribute_Manipulation',
    # 'pggan_v1': 'Entire_Face_Synthesis',
    # 'pggan_v2': 'Entire_Face_Synthesis',
    # 'stylegan_celeba': 'Entire_Face_Synthesis',
    'stylegan_ffhq': 'Entire_Face_Synthesis'
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
        elif label == 1 or label == 2 or label == 3:
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            else:
                mask = np.zeros(image.numpy().transpose(1, 2, 0).shape)
            label = 1
        elif label == 4:
            mask = np.ones((image.size(1), image.size(2)), dtype=np.uint8) * 255
            label = 1
        mask = self.mask_transform(mask)
        return {'images': image, 'labels': label, 'masks': mask}

    def __len__(self):
        return len(self.data)


def get_dffd_dataloader(model, args, mode, shuffle=True, num_workers=1):
    mask_transform, transform = create_transform(model.default_cfg)
    dataset = DffdDataset(args.data_dir, mode, transform=transform, mask_transform=mask_transform)
    dataloader = DataLoader(dataset, args.batch_size, shuffle, num_workers=num_workers, pin_memory=True,
                            drop_last=False)
    return dataloader
