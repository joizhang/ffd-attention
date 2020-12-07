import os
import unittest

import torch
from torch import hub
from torch.backends import cudnn
from torchsummary import summary

from config import Config
from training.models import xception_reg

CONFIG = Config()
hub.set_dir(CONFIG['TORCH_HOME'])
os.environ["CUDA_VISIBLE_DEVICES"] = CONFIG['CUDA_VISIBLE_DEVICES']
torch.backends.cudnn.benchmark = True


class XceptionTestCase(unittest.TestCase):

    def test_summary_xception_reg(self):
        self.assertTrue(torch.cuda.is_available())
        model = xception_reg(pretrained=True)
        model = model.cuda()
        input_size = model.default_cfg['input_size']
        summary(model, input_size=input_size, batch_size=10)


if __name__ == '__main__':
    unittest.main()
