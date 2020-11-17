import torch
import torch.nn as nn


class Decoder(nn.Module):

    def __init__(self, inplanes, planes):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(inplanes, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, planes, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.decoder(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_decoder(inplanes=1, planes=1):
    return Decoder(inplanes, planes)
