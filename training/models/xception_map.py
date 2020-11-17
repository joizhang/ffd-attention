import torch.nn.functional as F
from torch import nn

from training.models.resnet import Bottleneck
from training.models.xception import default_cfgs, SeparableConv2d, Xception
from training.tools.model_utils import load_pretrained

__all__ = ['xception_reg', 'xception_butd', 'xception_se']


class RegAttention(nn.Module):

    def __init__(self, in_channels=728, out_channels=1):
        super(RegAttention, self).__init__()
        # Reg Map
        self.map = SeparableConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        mask = self.map(x)
        mask = self.sigmoid(mask)
        x = x * mask
        return x, mask


class BottomUpTopDownAttention(nn.Module):

    def __init__(self, in_channels=728, out_channels=1):
        super(BottomUpTopDownAttention, self).__init__()
        # Bottom up top down
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.softmax1_blocks = nn.Sequential(
            # Bottleneck has expansion coefficient 4, so in_channels divided by 4
            Bottleneck(inplanes=in_channels, planes=in_channels // 4),
            Bottleneck(inplanes=in_channels, planes=in_channels // 4),
        )
        self.interpolation1 = nn.UpsamplingBilinear2d(size=(19, 19))
        self.softmax2_blocks = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        mask = self.maxpool1(x)
        mask = self.softmax1_blocks(mask)
        mask = self.interpolation1(mask) + x
        mask = self.softmax2_blocks(mask)
        x = (1 + mask) * x
        return x, mask


class SEAttention(nn.Module):
    # TODO
    def __init__(self, in_channels=728, out_channels=728):
        super(SEAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduction_channels = max(in_channels // 16, 8)
        self.fc1 = nn.Conv2d(in_channels, reduction_channels, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduction_channels, out_channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class XceptionMap(Xception):

    def __init__(self, num_classes, in_chans, attn_type, **kwargs):
        """ Constructor
        Args:
            num_classes (int): number of classes
            attn_type (nn.Module): Include reg(Direct Regression), butd(Bottom-up Top-down), SE(Squeeze and Excitation),
        """
        super(XceptionMap, self).__init__(num_classes=num_classes, in_chans=in_chans)

        self.mask_attn = attn_type()

    def forward_features(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        # Middle flow
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        # Attention mechanism
        x, mask = self.mask_attn(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        # Exit flow
        x = self.block12(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        return x, mask

    def forward(self, x):
        x, mask = self.forward_features(x)
        x = self.global_pool(x).flatten(1)
        if self.drop_rate:
            F.dropout(x, self.drop_rate, training=self.training)
        x = self.fc(x)
        return x, mask


def _xception(pretrained=False, num_classes=1000, in_chans=3, attn_type=None, **kwargs):
    default_cfg = default_cfgs['xception']
    model = XceptionMap(num_classes=num_classes, in_chans=in_chans, attn_type=attn_type, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans, strict=False)
    return model


def xception_reg(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """[On the Detection of Digital Face Manipulation](https://arxiv.org/abs/1910.01717)"""
    return _xception(pretrained, num_classes, in_chans, RegAttention, **kwargs)


def xception_butd(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """[Residual Attention Network for Image Classification](https://arxiv.org/abs/1704.06904)"""
    return _xception(pretrained, num_classes, in_chans, BottomUpTopDownAttention, **kwargs)


def xception_se(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """[Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)"""
    return _xception(pretrained, num_classes, in_chans, SEAttention, **kwargs)
