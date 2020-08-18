import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.hub import load_state_dict_from_url
from training.models import VGG, Block
from training.models.vgg import make_layers, cfgs

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGGMap(VGG):

    def __init__(self, features, num_classes=1000, init_weights=True, templates=0):
        """
        Constructor
        """
        super(VGGMap, self).__init__(features, num_classes, init_weights)

        self.templates = templates
        self.map_conv1 = Block(256, 128, 2, 2, start_with_relu=True, grow_first=False)
        self.map_linear = nn.Linear(128, 10)
        self.relu = nn.ReLU(inplace=True)

        print(self.features)
        # self.conv_1_1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        # self.conv_1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        # self.conv_2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        # self.conv_2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        # self.conv_3_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        # self.conv_3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        # self.conv_3_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        # self.conv_4_1 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        # self.conv_4_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        # self.conv_4_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        # self.conv_5_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        # self.conv_5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        # self.conv_5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)

        if init_weights:
            self._initialize_weights()

    def mask_template(self, x):
        vec = self.map_conv1(x)
        vec = self.relu(vec)
        vec = F.adaptive_avg_pool2d(vec, (1, 1))
        vec = vec.view(vec.size(0), -1)
        vec = self.map_linear(vec)
        mask = torch.mm(vec, self.templates.reshape(10, 784))
        mask = mask.reshape(x.shape[0], 1, 28, 28)
        x = x * mask
        return x, mask, vec

    def forward(self, x):
        x = F.relu(self.conv_1_1(x))
        x = F.relu(self.conv_1_2(x))
        x = F.max_pool2d(x, 2, 2)  # B*64*112*112
        x = F.relu(self.conv_2_1(x))
        x = F.relu(self.conv_2_2(x))
        x = F.max_pool2d(x, 2, 2)  # B*128*56*56
        x = F.relu(self.conv_3_1(x))
        x = F.relu(self.conv_3_2(x))
        x = F.relu(self.conv_3_3(x))
        x = F.max_pool2d(x, 2, 2)  # B*256*28*28
        x, mask, vec = self.mask_template(x)

        # map_ = torch.sigmoid(x[:,-1,:,:].unsqueeze(1))
        # x = x[:,0:-1,:,:]
        # x = map_.expand_as(x).mul(x)

        x = F.relu(self.conv_4_1(x))
        x = F.relu(self.conv_4_2(x))
        x = F.relu(self.conv_4_3(x))
        x = F.max_pool2d(x, 2, 2)  # B*512*14*14
        x = F.relu(self.conv_5_1(x))
        x = F.relu(self.conv_5_2(x))
        x = F.relu(self.conv_5_3(x))

        # print(x.shape)
        x = self.avgpool(x)
        # print(x.shape)
        x = torch.flatten(x, 1)
        # print(x.shape)
        # input()
        x = self.classifier(x)

        return x, mask, vec
        # return x, 0, 0

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGGMap(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


# def vgg16(pretrained=False, progress=True, templates=0, num_classes=2, **kwargs):
#     model = VGGMap(templates, num_classes)
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls['vgg16'], progress=progress)
#         state_dict_new = {}
#         for name, weights in state_dict.items():
#             # print(name)
#             if 'features' in name:
#                 state_dict_new['conv_1_1.weight'] = state_dict['features.0.weight']
#                 state_dict_new['conv_1_2.weight'] = state_dict['features.2.weight']
#                 state_dict_new['conv_2_1.weight'] = state_dict['features.5.weight']
#                 state_dict_new['conv_2_2.weight'] = state_dict['features.7.weight']
#                 state_dict_new['conv_3_1.weight'] = state_dict['features.10.weight']
#                 state_dict_new['conv_3_2.weight'] = state_dict['features.12.weight']
#                 state_dict_new['conv_3_3.weight'] = state_dict['features.14.weight']
#                 state_dict_new['conv_4_1.weight'] = state_dict['features.17.weight']
#                 state_dict_new['conv_4_2.weight'] = state_dict['features.19.weight']
#                 state_dict_new['conv_4_3.weight'] = state_dict['features.21.weight']
#                 state_dict_new['conv_5_1.weight'] = state_dict['features.24.weight']
#                 state_dict_new['conv_5_2.weight'] = state_dict['features.26.weight']
#                 state_dict_new['conv_5_3.weight'] = state_dict['features.28.weight']
#                 state_dict_new['conv_1_1.bias'] = state_dict['features.0.bias']
#                 state_dict_new['conv_1_2.bias'] = state_dict['features.2.bias']
#                 state_dict_new['conv_2_1.bias'] = state_dict['features.5.bias']
#                 state_dict_new['conv_2_2.bias'] = state_dict['features.7.bias']
#                 state_dict_new['conv_3_1.bias'] = state_dict['features.10.bias']
#                 state_dict_new['conv_3_2.bias'] = state_dict['features.12.bias']
#                 state_dict_new['conv_3_3.bias'] = state_dict['features.14.bias']
#                 state_dict_new['conv_4_1.bias'] = state_dict['features.17.bias']
#                 state_dict_new['conv_4_2.bias'] = state_dict['features.19.bias']
#                 state_dict_new['conv_4_3.bias'] = state_dict['features.21.bias']
#                 state_dict_new['conv_5_1.bias'] = state_dict['features.24.bias']
#                 state_dict_new['conv_5_2.bias'] = state_dict['features.26.bias']
#                 state_dict_new['conv_5_3.bias'] = state_dict['features.28.bias']
#             else:
#                 state_dict_new[name] = state_dict[name]
#         del state_dict_new['classifier.6.weight']
#         del state_dict_new['classifier.6.bias']
#         del state_dict_new['conv_3_3.weight']
#         del state_dict_new['conv_3_3.bias']
#         model.load_state_dict(state_dict_new, False)
#         # model = torch.load('./pretrain_vgg16.pth')
#         # print("loaded")
#     else:
#         model.apply(init_weights)
#     return model


def vgg16_map(pretrained=False, progress=True, **kwargs):
    return _vgg('vgg16', 'D', False, pretrained, progress, **kwargs)


def vgg16_bn_map(pretrained=False, progress=True, **kwargs):
    return _vgg('vgg16', 'D', True, pretrained, progress, **kwargs)
