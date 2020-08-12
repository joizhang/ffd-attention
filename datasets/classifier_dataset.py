import os
from glob import glob
from pathlib import Path

import cv2
from torch.utils.data import Dataset

"""
Real face images: FFHQ, CelebA
Identity and expression swap: FaceForensics++(FaceSwap and Deepfake)
Attributes manipulation: FaceAPP, StarGAN
Entire face synthesis: PGGAN, StyleGAN
"""
DFFD = {'ffhq': 'Real', 'faceapp': 'Fake', 'stargan': 'Fake', 'pggan': 'Fake'}


class DffdDataset(Dataset):

    def __init__(self, data_root, mode, transform, classes=None, seed=111):
        """
        Args:
            classes (dict): {'Real': 0, 'Fake_Entire': 1, 'Fake_Partial': 2}
        """
        self.data_root = data_root
        self.mode = mode
        self.transform = transform
        self.classes = classes
        self.data = []
        for mode_path in glob(os.path.join(data_root, "*/{}".format(mode))):
            print(mode_path)
            label = self.classes[DFFD[Path(mode_path).parent.name]]
            file_names = os.listdir(mode_path)
            for file_name in file_names:
                self.data.append((os.path.join(mode_path, file_name), label))

    def __getitem__(self, index):
        image_path, label = self.data[index]
        # img = self.load_img(image_path)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.data)
