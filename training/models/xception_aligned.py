import math
import logging

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

__all__ = ['aligned_xception']

default_cfgs = {
    'xception': {
        'url': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-cadene/xception-43020ad28.pth',
        'input_size': (3, 299, 299),
        'pool_size': (10, 10),
        'crop_pct': 0.8975,
        'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5),
        'std': (0.5, 0.5, 0.5),
        'num_classes': 1000,
        'first_conv': 'conv1',
        'classifier': 'fc'
        # The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
    }
}


def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, [pad_beg, pad_end, pad_beg, pad_end])
    return padded_inputs


# Calculate symmetric padding for a convolution
def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1, **_) -> int:
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


class SeparableConv2d(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        padding = get_padding(kernel_size, stride, dilation)
        self.convdw = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding, dilation, groups=inplanes,
                                bias=bias)
        self.bn = nn.BatchNorm2d(num_features=inplanes)
        self.convpw = nn.Conv2d(inplanes, planes, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.convdw(x)
        x = self.bn(x)
        x = self.convpw(x)
        return x


class Block(nn.Module):

    def __init__(self, inplanes, planes, reps, stride=1, dilation=1,
                 start_with_relu=True, grow_first=True, is_last=False):
        super(Block, self).__init__()

        if planes != inplanes or stride != 1:
            self.skip = nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False)
            self.skipbn = nn.BatchNorm2d(planes)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = inplanes
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(inplanes, planes, 3, 1, dilation))
            rep.append(nn.BatchNorm2d(planes))
            filters = planes

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, 1, dilation))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(inplanes, planes, 3, 1, dilation))
            rep.append(nn.BatchNorm2d(planes))

        if not start_with_relu:
            rep = rep[1:]

        if stride != 1:
            rep.append(self.relu)
            rep.append(SeparableConv2d(planes, planes, 3, 2))
            rep.append(nn.BatchNorm2d(planes))

        if stride == 1 and is_last:
            rep.append(self.relu)
            rep.append(SeparableConv2d(planes, planes, 3, 1))
            rep.append(nn.BatchNorm2d(planes))

        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x = x + skip
        return x


class AlignedXception(nn.Module):
    """
    Modified Alighed Xception
    """

    def __init__(self, in_chans, output_stride, **kwargs):
        super(AlignedXception, self).__init__()

        if output_stride == 16:
            entry_block3_stride = 2
            middle_block_dilation = 1
            exit_block_dilations = (1, 2)
        elif output_stride == 8:
            entry_block3_stride = 1
            middle_block_dilation = 2
            exit_block_dilations = (2, 4)
        else:
            raise NotImplementedError

        # Entry flow
        self.conv1 = nn.Conv2d(in_chans, 32, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=False, grow_first=True)
        self.block3 = Block(256, 728, 2, entry_block3_stride, start_with_relu=True, grow_first=True, is_last=True)

        # Middle flow
        self.block4 = Block(728, 728, 3, 1, middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block5 = Block(728, 728, 3, 1, middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block6 = Block(728, 728, 3, 1, middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block7 = Block(728, 728, 3, 1, middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block8 = Block(728, 728, 3, 1, middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block9 = Block(728, 728, 3, 1, middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, 3, 1, middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, 3, 1, middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block12 = Block(728, 728, 3, 1, middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block13 = Block(728, 728, 3, 1, middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block14 = Block(728, 728, 3, 1, middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block15 = Block(728, 728, 3, 1, middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block16 = Block(728, 728, 3, 1, middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block17 = Block(728, 728, 3, 1, middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block18 = Block(728, 728, 3, 1, middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block19 = Block(728, 728, 3, 1, middle_block_dilation, start_with_relu=True, grow_first=True)

        # Exit flow
        self.block20 = Block(728, 1024, 2, 1, exit_block_dilations[0],
                             start_with_relu=True, grow_first=False, is_last=True)
        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, dilation=exit_block_dilations[1])
        self.bn3 = nn.BatchNorm2d(1536)
        self.conv4 = SeparableConv2d(1536, 1536, 3, 1, dilation=exit_block_dilations[1])
        self.bn4 = nn.BatchNorm2d(1536)
        self.conv5 = SeparableConv2d(1536, 2048, 3, 1, dilation=exit_block_dilations[1])
        self.bn5 = nn.BatchNorm2d(2048)

        # Init weights
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        # add relu here
        x = self.relu(x)
        low_level_feat = x
        x = self.block2(x)
        x = self.block3(x)

        # Middle flow
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)

        # Exit flow
        x = self.block20(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        return x, low_level_feat


def _load_pretrained(model, cfg=None, strict=True):
    if cfg is None:
        cfg = getattr(model, 'default_cfg')
    if cfg is None or 'url' not in cfg or not cfg['url']:
        logging.warning("Pretrained model URL is invalid, using random initialization.")
        return
    state_dict = model_zoo.load_url(cfg['url'], progress=False, map_location='cpu')
    model_dict = {}

    for k, v in state_dict.items():
        if k in state_dict:
            # if 'pointwise' in k:
            #     v = v.unsqueeze(-1).unsqueeze(-1)
            if k.startswith('block11'):
                model_dict[k] = v
                model_dict[k.replace('block11', 'block12')] = v
                model_dict[k.replace('block11', 'block13')] = v
                model_dict[k.replace('block11', 'block14')] = v
                model_dict[k.replace('block11', 'block15')] = v
                model_dict[k.replace('block11', 'block16')] = v
                model_dict[k.replace('block11', 'block17')] = v
                model_dict[k.replace('block11', 'block18')] = v
                model_dict[k.replace('block11', 'block19')] = v
            elif k.startswith('block12'):
                model_dict[k.replace('block12', 'block20')] = v
            elif k.startswith('bn3'):
                model_dict[k] = v
                model_dict[k.replace('bn3', 'bn4')] = v
            elif k.startswith('conv4'):
                model_dict[k.replace('conv4', 'conv5')] = v
            elif k.startswith('bn4'):
                model_dict[k.replace('bn4', 'bn5')] = v
            else:
                model_dict[k] = v
    state_dict.update(model_dict)
    del state_dict['conv4.pointwise.weight']
    model.load_state_dict(state_dict, strict=strict)


def aligned_xception(pretrained=False, in_chans=3, output_stride=16, **kwargs):
    default_cfg = default_cfgs['xception']
    model = AlignedXception(in_chans, output_stride, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        _load_pretrained(model, default_cfg, strict=False)
    return model
