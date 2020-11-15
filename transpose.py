import torch
from torch import nn

x = torch.randn(1, 1, 16, 16)
# downsample = nn.Conv2d(1, 1, 3, stride=2, padding=1)
# h = downsample(input)
# print(h.size())

# nn.Up
# deconv1 = nn.ConvTranspose2d(1, 1, 4, 1, 0, bias=False)
# deconv2 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
# x = deconv1(x)
# x = deconv2(x)
# print(x.size())


decoder = nn.Sequential(
    nn.ConvTranspose2d(1, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
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

    nn.ConvTranspose2d(8, 1, kernel_size=3, stride=1, padding=1, output_padding=0),
    nn.Softmax(dim=1)
)

x = decoder(x)
print(x.size())