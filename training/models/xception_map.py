from torch import nn
import torch.nn.functional as F

from training.models.xception import default_cfgs, SeparableConv2d, Xception
from training.tools.model_utils import load_pretrained
from training.models import Bottleneck

__all__ = ['xception_map']


class XceptionMap(Xception):

    def __init__(self, num_classes, in_chans, attn_type, **kwargs):
        """ Constructor
        Args:
            attn_type (str): Include reg, butd(bottom-up top-down), map,
            num_classes (int): number of classes
        """
        super(XceptionMap, self).__init__(num_classes=num_classes, in_chans=in_chans, **kwargs)

        self.attn_type = attn_type
        # Reg Map
        self.map = SeparableConv2d(728, 1, 3, stride=1, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        # Bottom up top down
        in_channels, out_channels = 728, 728
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.softmax1_blocks = nn.Sequential(
            # Bottleneck has expansion coefficient 4, so out_channels divided by 4
            Bottleneck(inplanes=in_channels, planes=out_channels // 4),
            Bottleneck(inplanes=in_channels, planes=out_channels // 4),
        )
        self.interpolation1 = nn.UpsamplingBilinear2d(size=(19, 19))
        self.softmax2_blocks = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid()
        )

    def mask_reg(self, x):
        mask = self.map(x)
        mask = self.sigmoid(mask)
        x = x * mask
        return x, mask, 0

    def mask_bottom_up_top_down(self, x):
        mask = self.maxpool1(x)
        mask = self.softmax1_blocks(mask)
        mask = self.interpolation1(mask) + x
        mask = self.softmax2_blocks(mask)
        x = (1 + mask) * x
        return x, mask, 0

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
        if self.attn_type == 'reg':
            x, mask, vec = self.mask_reg(x)
        else:
            x, mask, vec = self.mask_bottom_up_top_down(x)
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
        return x, mask, vec

    def forward(self, x):
        x, mask, vec = self.forward_features(x)
        x = self.global_pool(x).flatten(1)
        if self.drop_rate:
            F.dropout(x, self.drop_rate, training=self.training)
        x = self.fc(x)
        return x, mask, vec


def xception_map(pretrained=False, num_classes=1000, in_chans=3, attn_type='reg', **kwargs):
    default_cfg = default_cfgs['xception']
    model = XceptionMap(num_classes=num_classes, in_chans=in_chans, attn_type=attn_type, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans, strict=False)
    return model
