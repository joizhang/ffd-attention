import os

import torch
from torch import hub
from torch import nn
from torch.backends import cudnn
from torch.utils.data import dataset
from torchvision import transforms, datasets

from config import Config
from training.models import xception
from training.tools.model_utils import validate

torch.backends.cudnn.benchmark = True

CONFIG = Config()
hub.set_dir(CONFIG['TORCH_HOME'])

if __name__ == '__main__':
    pass
