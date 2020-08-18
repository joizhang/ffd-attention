import unittest

import torch
from torch import hub
from torch.backends import cudnn

from config import Config
from training.models.vgg_map import vgg16_map

torch.backends.cudnn.benchmark = True

CONFIG = Config()
hub.set_dir(CONFIG['TORCH_HOME'])


class VGGMapTestCase(unittest.TestCase):

    def test_vgg16_map(self):
        model = vgg16_map(pretrained=False)


if __name__ == '__main__':
    unittest.main()
