import os
import unittest

import torch
from pytorch_toolbelt.utils import count_parameters
from torch import hub
from torch.backends import cudnn
from torchsummary import summary

from config import Config
from training.models.mesonet import meso4, meso_inception4

CONFIG = Config()
hub.set_dir(CONFIG['TORCH_HOME'])
os.environ["CUDA_VISIBLE_DEVICES"] = CONFIG['CUDA_VISIBLE_DEVICES']

torch.backends.cudnn.benchmark = True


class MesoNetTestCase(unittest.TestCase):

    def test_meso4(self):
        self.assertTrue(torch.cuda.is_available())
        model = meso4()
        model = model.cuda()
        input_size = model.default_cfg['input_size']
        summary(model, input_size=input_size)

    def test_count_meso4_params(self):
        self.assertTrue(torch.cuda.is_available())
        model = meso4()
        print(count_parameters(model))

    def test_meso_inception4(self):
        self.assertTrue(torch.cuda.is_available())
        model = meso_inception4()
        model = model.cuda()
        input_size = model.default_cfg['input_size']
        summary(model, input_size=input_size)


if __name__ == '__main__':
    unittest.main()