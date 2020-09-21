import os
import unittest

import timm
import torch
from torch.backends import cudnn
from torchsummary import summary

from training.models.xception_aligned import aligned_xception

torch.backends.cudnn.benchmark = True

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


class XceptionTestCase(unittest.TestCase):

    def test_gluon_xception(self):
        self.assertTrue(torch.cuda.is_available())
        model = timm.create_model('gluon_xception71')
        model = model.cuda()
        input_size = (3, 299, 299)
        summary(model, input_size=input_size)

    def test_summary_xception(self):
        self.assertTrue(torch.cuda.is_available())
        model = aligned_xception(pretrained=True)

        # inputs = torch.rand(1, 3, 299, 299)
        # output = model(inputs)
        # print(output[0].size(), output[1].size())

        model = model.cuda()
        input_size = (3, 299, 299)
        summary(model, input_size=input_size)


if __name__ == '__main__':
    unittest.main()