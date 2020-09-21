import torch.nn as nn

from training.models import build_aspp
from training.models import build_decoder
from training.models import aligned_xception


class DeepLabV3Plus(nn.Module):

    def __init__(self, backbone='resnet', output_stride=16, num_classes=2, freeze_bn=True):
        super(DeepLabV3Plus, self).__init__()

        self.num_features = 2048
        if backbone == 'resnet':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError

        self.backbone = aligned_xception(output_stride=output_stride)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Linear(self.num_features, num_classes)
        )
        self.aspp = build_aspp(in_channels=2048)
        self.decoder = build_decoder(low_level_inplanes, num_classes)
        self.freeze_bn = freeze_bn

    def forward(self, inputs):
        x, low_level_feat = self.backbone(inputs)
        x = self.classifier(x)
        mask = self.aspp(x)
        mask = self.decoder(mask, low_level_feat)
        return x, mask

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone, self.classifier]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
