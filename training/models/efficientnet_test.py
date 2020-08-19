import os
import unittest

import torch
from PIL import Image
from timm.models.efficientnet import efficientnet_b3
from torch import hub
from torch import nn
from torch.backends import cudnn
from torch.utils.data import dataset
from torchsummary import summary
from torchvision import transforms, datasets

from config import Config
from training.tools.model_utils import validate

torch.backends.cudnn.benchmark = True

CONFIG = Config()

hub.set_dir(CONFIG['TORCH_HOME'])

os.environ["CUDA_VISIBLE_DEVICES"] = CONFIG['CUDA_VISIBLE_DEVICES']


class EfficientNetTestCase(unittest.TestCase):

    def test_summary_efficientnet(self):
        self.assertTrue(torch.cuda.is_available())
        model = efficientnet_b3(pretrained=True, num_classes=1000, in_chans=3)
        model = model.cuda()
        input_size = (3, 224, 224)
        summary(model, input_size=input_size)

    def test_efficientnet(self):
        model = efficientnet_b3(pretrained=True, num_classes=1000, in_chans=3)
        model = model.cuda()
        criterion = nn.CrossEntropyLoss().cuda()

        valdir = os.path.join(CONFIG['IMAGENET_HOME'], 'val')
        self.assertEqual(True, os.path.exists(valdir))
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256, Image.BICUBIC),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])),
            batch_size=50, shuffle=False,
            num_workers=1, pin_memory=False)

        validate(val_loader, model, criterion)


if __name__ == '__main__':
    unittest.main()
