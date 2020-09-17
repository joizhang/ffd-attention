import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):

    def __init__(self, low_level_inplanes, num_classes):
        super(Decoder, self).__init__()

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU()

        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.last_conv = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        )
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self._init_weight()

    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = self.upsample1(x)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)
        x = self.upsample2(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_decoder(low_level_inplanes, num_classes):
    return Decoder(low_level_inplanes, num_classes)
