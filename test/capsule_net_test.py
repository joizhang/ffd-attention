import os
import unittest

import torch
from torch import hub
from torch.backends import cudnn
from torchsummary import summary

from config import Config
from training.models.capsule_net import CapsuleNet

CONFIG = Config()
hub.set_dir(CONFIG['TORCH_HOME'])
os.environ["CUDA_VISIBLE_DEVICES"] = CONFIG['CUDA_VISIBLE_DEVICES']

torch.backends.cudnn.benchmark = True


class CapsuleNetTestCase(unittest.TestCase):

    def test_summary_capsule_net(self):
        self.assertTrue(torch.cuda.is_available())
        model = CapsuleNet(num_class=2, gpu_id=0)
        model = model.cuda()
        input_size = (3, 224, 224)
        summary(model, input_size=input_size)