import os
import unittest

import torch
from torch import hub
from torch import nn
from torch.backends import cudnn
from torch.utils.data import dataset
from torchvision import transforms, datasets
from torchsummary import summary

from training.models import xception
from training.tools.model_utils import validate
from config import Config

torch.backends.cudnn.benchmark = True

CONFIG = Config()

hub.set_dir(CONFIG['TORCH_HOME'])

os.environ["CUDA_VISIBLE_DEVICES"] = CONFIG['CUDA_VISIBLE_DEVICES']


class XceptionTestCase(unittest.TestCase):

    def test_summary_xception(self):
        self.assertTrue(torch.cuda.is_available())
        model = xception(pretrained=True)
        model = model.cuda()
        input_size = (3, 299, 299)
        summary(model, input_size=input_size)

    def test_xception(self):
        model = xception(pretrained=True)
        model = model.cuda()
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
            num_workers=1, pin_memory=True)

        validate(val_loader, model, criterion)


if __name__ == '__main__':
    unittest.main()
