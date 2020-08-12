import os
import unittest

import torch
from timm.models.resnet import resnet34
from torch import hub
from torch import nn
from torch.backends import cudnn
from torch.utils.data import dataset
from torchsummary import summary
from torchvision import transforms, datasets

from config import Config
from tools.model_utils import validate

torch.backends.cudnn.benchmark = True

CONFIG = Config()
hub.set_dir(CONFIG['TORCH_HOME'])


class ResNetTestCase(unittest.TestCase):

    def test_resnet(self):
        gpu = 0
        torch.cuda.set_device(gpu)
        model = resnet34(pretrained=True)
        model = model.cuda()
        summary(model, input_size=(3, 224, 224))
        criterion = nn.CrossEntropyLoss().cuda()

        valdir = os.path.join(CONFIG['IMAGENET_HOME'], 'val')
        self.assertEqual(True, os.path.exists(valdir))
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])),
            batch_size=50, shuffle=False,
            num_workers=1, pin_memory=True)

        validate(val_loader, model, criterion)


if __name__ == '__main__':
    unittest.main()
