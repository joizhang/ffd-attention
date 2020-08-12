import os
import unittest

import torch
from torch import hub
from torch import nn
from torch.backends import cudnn
from torch.utils.data import dataset
from torchvision import transforms, datasets
from torchsummary import summary

from models.xception import xception
from tools.model_utils import validate
from config import Config

torch.backends.cudnn.benchmark = True

CONFIG = Config()
hub.set_dir(CONFIG['TORCH_HOME'])


class XceptionTestCase(unittest.TestCase):

    def test_xception(self):
        gpu = 0
        torch.cuda.set_device(gpu)
        model = xception(pretrained=True)
        model = model.cuda()
        input_size = (3, 299, 299)
        summary(model, input_size=input_size)
        criterion = nn.CrossEntropyLoss().cuda()

        valdir = os.path.join(CONFIG['IMAGENET_HOME'], 'val')
        self.assertEqual(True, os.path.exists(valdir))
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])),
            batch_size=10, shuffle=False,
            num_workers=2, pin_memory=True)

        validate(val_loader, model, criterion, input_size)


if __name__ == '__main__':
    unittest.main()
