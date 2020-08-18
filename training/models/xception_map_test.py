import os
import unittest

import torch
from torch import hub
from torch.backends import cudnn
from torchsummary import summary

from config import Config
from training.models import xception_map

torch.backends.cudnn.benchmark = True

CONFIG = Config()

hub.set_dir(CONFIG['TORCH_HOME'])

os.environ["CUDA_VISIBLE_DEVICES"] = CONFIG['CUDA_VISIBLE_DEVICES']


class XceptionMapTestCase(unittest.TestCase):

    def test_summary_xception_map(self):
        self.assertTrue(torch.cuda.is_available())
        model = xception_map(pretrained=True)
        model = model.cuda()
        input_size = (3, 299, 299)
        summary(model, input_size=input_size)

    def test_xception_map(self):
        model = xception_map(pretrained=False)
        x = torch.randn((1, 3, 299, 299), dtype=torch.float32).cuda()
        model = model.cuda()
        outputs = model(x)
        self.assertEqual(torch.Size([1, 1000]), outputs[0].shape)


if __name__ == '__main__':
    unittest.main()
